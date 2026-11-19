"""Cascade base classifier for diffusion denoised smoothing (CIFAR-10).

Drop this file into `cifar10/` of ethz-privsec/diffusion_denoised_smoothing.

Base classifier f(x + eps), eps ~ N(0, sigma^2 I) in [0, 1] pixel space:
  1. Cheap path : Cohen et al. noise-trained ResNet-110 applied directly to x + eps.
  2. If max softmax of cheap path >= tau  -> return cheap prediction.
     Otherwise                            -> expensive path: one-step (or multi-step)
                                             diffusion denoising + ViT (Carlini et al.).

Routing depends only on (x + eps) and a tau fixed in advance, so f is still a fixed
function of the noisy input and Cohen's CERTIFY remains valid unchanged.

Modes:
  dds     : always expensive path (baseline, same noise formulation as cascade)
  cheap   : always cheap path (Cohen baseline)
  cascade : cheap first, expensive only when not confident
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from transformers import AutoModelForImageClassification

from archs.cifar_resnet import resnet as resnet_cifar

DIFFUSION_CKPT = "cifar10/cifar10_uncond_50M_500K.pt"
VIT_NAME = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD = [0.2023, 0.1994, 0.2010]


class Args:  # identical to DRM.py
    image_size = 32
    num_channels = 128
    num_res_blocks = 3
    num_heads = 4
    num_heads_upsample = -1
    attention_resolutions = "16,8"
    dropout = 0.3
    learn_sigma = True
    sigma_small = False
    class_cond = False
    diffusion_steps = 4000
    noise_schedule = "cosine"
    timestep_respacing = ""
    use_kl = False
    predict_xstart = False
    rescale_timesteps = True
    rescale_learned_sigmas = True
    use_checkpoint = False
    use_scale_shift_norm = True


class NormalizeLayer(nn.Module):
    """Like Cohen's: mean/std are plain tensors (NOT in state_dict), so checkpoints
    from locuslab/smoothing (keys '1.*' only) load with strict=True."""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


def _load(path):
    """torch>=2.6 defaults to weights_only=True, which can reject old (2019) checkpoints."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")


def load_cohen_resnet110(ckpt_path: str) -> nn.Module:
    """Load a checkpoint from locuslab/smoothing (Sequential(normalize, resnet110))."""
    model = nn.Sequential(NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STD),
                          resnet_cifar(depth=110, num_classes=10))
    ckpt = _load(ckpt_path)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    return model


def sigma_to_timestep(diffusion, sigma: float) -> int:
    """Smallest t with sqrt(1-abar_t)/sqrt(abar_t) >= 2*sigma (same rule as certify.py)."""
    target, t, real = sigma * 2, 0, 0.0
    while real < target:
        t += 1
        real = diffusion.sqrt_one_minus_alphas_cumprod[t] / diffusion.sqrt_alphas_cumprod[t]
    return t


class CascadeRobustModel(nn.Module):
    def __init__(self, sigma: float, mode: str = "cascade", tau: float = 0.9,
                 cheap_ckpt: str = None, multistep: bool = False, amp: bool = False):
        super().__init__()
        assert mode in ("dds", "cheap", "cascade")
        self.sigma, self.mode, self.tau, self.multistep = sigma, mode, tau, multistep
        # amp: fp16 autocast for the expensive path (T4 tensor cores). Still a fixed deterministic
        # function of x+eps, so the certificate is unaffected; use the SAME flag for dds and cascade.
        self.amp = amp

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys()))
        self.diffusion = diffusion
        self.t = sigma_to_timestep(diffusion, sigma)
        if mode != "cheap":
            model.load_state_dict(_load(DIFFUSION_CKPT))
            self.model = model.eval().cuda()
            self.classifier = AutoModelForImageClassification.from_pretrained(VIT_NAME).eval().cuda()
        if mode != "dds":
            assert cheap_ckpt is not None, "--cheap_ckpt is required for cheap/cascade"
            self.cheap = load_cohen_resnet110(cheap_ckpt).eval().cuda()

        self.reset_stats()

    # ---------- bookkeeping ----------
    def reset_stats(self):
        self.n_total = 0
        self.n_expensive = 0

    # ---------- paths ----------
    @torch.no_grad()
    def cheap_logits(self, x_noisy):
        return self.cheap(x_noisy)

    @torch.no_grad()
    def expensive_logits(self, x_noisy):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            return self._expensive_logits(x_noisy).float()

    def _expensive_logits(self, x_noisy):
        a =float(self.diffusion.sqrt_alphas_cumprod[self.t])   # numpy float64 -> python float
        x_t = a * (2 * x_noisy - 1)                      # x_t = sqrt(abar) * (x_in + delta)
        t_batch = torch.full((len(x_t),), self.t, device=x_t.device, dtype=torch.long)
        if self.multistep:
            out = x_t
            for i in range(self.t)[::-1]:
                tb = torch.full((len(x_t),), i, device=x_t.device, dtype=torch.long)
                out = self.diffusion.p_sample(self.model, out, tb, clip_denoised=True)["sample"]
        else:
            out = self.diffusion.p_sample(self.model, x_t, t_batch, clip_denoised=True)["pred_xstart"]
        imgs = F.interpolate(out, (224, 224), mode="bicubic", antialias=True)
        return self.classifier(imgs).logits

    # ---------- base classifier used by Smooth ----------
    @torch.no_grad()
    def forward(self, x, t=None):
        """x: clean images in [0,1], already repeated by Smooth._sample_noise.
        Returns one-hot 'logits' (Smooth only uses argmax)."""
        x_noisy = x + torch.randn_like(x) * self.sigma
        return self.forward_noisy(x_noisy)

    @torch.no_grad()
    def forward_noisy(self, x_noisy):
        n = len(x_noisy)
        self.n_total += n
        if self.mode == "dds":
            self.n_expensive += n
            pred = self.expensive_logits(x_noisy).argmax(1)
        elif self.mode == "cheap":
            pred = self.cheap_logits(x_noisy).argmax(1)
        else:
            probs = self.cheap_logits(x_noisy).softmax(1)
            conf, pred = probs.max(1)
            hard = conf < self.tau
            if hard.any():
                self.n_expensive += int(hard.sum())
                pred[hard] = self.expensive_logits(x_noisy[hard]).argmax(1)
        return F.one_hot(pred, 10).float()
