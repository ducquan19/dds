"""Step 2: measure the cheap path on noisy CIFAR-10 calibration images.

Default split = held-out TEST images (idx % certify_skip != 0), disjoint from the certified
images. Train images are NOT used by default: both models were trained on them, which makes
the cheap path look over-confident and the calibrated risk optimistic.

For each image: draw ONE noise sample (keeps samples i.i.d. across images for the
statistical test in calibrate_tau.py) plus K extra samples for analysis.
Saves cheap confidence / prediction, expensive (DDS) prediction and label.

Noise is drawn per image from a CPU generator seeded by (seed, image idx), so the SAME noise can be
regenerated. --reuse_exp <previous probe.npz> then skips the expensive path entirely and only runs
the cheap model (seconds per candidate), reusing the stored DDS predictions.

Example:
  python cifar10/probe.py --sigma 0.25 --cheap_ckpt models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar \
      --num_images 5000 --K 4 --out probe_s0.25.npz
"""
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms

from DRM_cascade import CascadeRobustModel


def main(a):
    torch.manual_seed(a.seed)
    prev = np.load(a.reuse_exp) if a.reuse_exp else None
    model = CascadeRobustModel(a.sigma, mode="cheap" if prev is not None else "cascade", tau=1.01,
                               cheap_ckpt=a.cheap_ckpt, amp=a.amp, cheap_norm=a.cheap_norm)
    if a.split == "train":
        # NOTE: both the Cohen ResNet and the ViT were trained on these images -> optimistic risk.
        ds = datasets.CIFAR10(a.data_dir, train=True, download=False, transform=transforms.ToTensor())
        pool = torch.arange(len(ds))
    else:
        # held-out TEST images disjoint from the certified ones (certify uses idx % skip == 0)
        ds = datasets.CIFAR10(a.data_dir, train=False, download=False, transform=transforms.ToTensor())
        pool = torch.tensor([i for i in range(len(ds)) if i % a.certify_skip != 0])
    # dedicated generator: model construction (e.g. ViT init in full mode, absent in --reuse_exp mode)
    # consumes the global RNG, which previously made the two modes pick different images
    idx = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(a.seed))[: a.num_images]].tolist()
    print(f"split={a.split}: {len(idx)} calibration images (pool {len(pool)})", flush=True)

    if prev is not None:
        assert int(prev.get("idx_version", 1)) == 2, \
            "reuse_exp: base probe was made by an older probe.py (different image sampling); delete it and rerun"
        assert int(prev["seed"]) == a.seed and int(prev["K"]) == a.K and float(prev["sigma"]) == a.sigma,             "reuse_exp: seed/K/sigma differ from the stored probe"
        assert np.array_equal(prev["img"][:: 1 + a.K], np.array(idx)), "reuse_exp: different calibration images"
    rec = {k: [] for k in ["img", "label", "conf", "cheap", "exp"]}
    xs, ys, ids = [], [], []

    def flush():
        x = torch.stack(xs).repeat_interleave(1 + a.K, 0)
        eps = torch.cat([torch.randn((1 + a.K,) + tuple(xs[0].shape),
                                     generator=torch.Generator().manual_seed(a.seed * 1_000_003 + 7_919 + j))
                         for j in ids])
        x_noisy = (x + eps * a.sigma).cuda()
        probs = model.cheap_logits(x_noisy).softmax(1)
        conf, cheap = probs.max(1)
        if prev is None:
            exp = torch.cat([model.expensive_logits(c).argmax(1) for c in x_noisy.split(a.batch_size)]).cpu().numpy()
        else:
            exp = np.zeros(len(x_noisy), dtype=np.int64)   # filled from the stored probe below
        rec["img"].append(np.repeat(ids, 1 + a.K))
        rec["label"].append(np.repeat(ys, 1 + a.K))
        rec["conf"].append(conf.cpu().numpy())
        rec["cheap"].append(cheap.cpu().numpy())
        rec["exp"].append(exp)
        xs.clear(); ys.clear(); ids.clear()

    per_flush = max(1, a.batch_size // (1 + a.K))
    for n, i in enumerate(idx):
        x, y = ds[i]
        xs.append(x); ys.append(y); ids.append(i)
        if len(xs) == per_flush:
            flush()
        if n % 500 == 0:
            print(f"{n}/{len(idx)}", flush=True)
    if xs:
        flush()

    out = {k: np.concatenate(v) for k, v in rec.items()}
    out["first"] = np.zeros(len(out["img"]), dtype=bool)
    out["first"][:: 1 + a.K] = True          # the i.i.d. sample of each image
    if prev is not None:
        out["exp"] = prev["exp"]
    np.savez(a.out, sigma=a.sigma, seed=a.seed, K=a.K, idx_version=2, **out)

    f = out["first"]
    print(f"saved {a.out}")
    print(f"cheap acc {np.mean(out['cheap'][f] == out['label'][f]):.4f} | "
          f"dds acc {np.mean(out['exp'][f] == out['label'][f]):.4f} | "
          f"agreement {np.mean(out['cheap'][f] == out['exp'][f]):.4f}")
    for tau in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        acc = out["conf"][f] >= tau
        dis = np.mean(acc & (out["cheap"][f] != out["exp"][f]))
        print(f"tau={tau:.2f}  accept={acc.mean():.3f}  P(accept & disagree)={dis:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--cheap_ckpt", type=str, required=True)
    p.add_argument("--num_images", type=int, default=5000)
    p.add_argument("--split", choices=["test_holdout", "train"], default="test_holdout")
    p.add_argument("--certify_skip", type=int, default=20,
                   help="must equal --skip of certify_cascade.py; those images are excluded")
    p.add_argument("--K", type=int, default=4, help="extra noise samples per image (analysis only)")
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--data_dir", type=str, default="data/cifar10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="fp16 autocast on the expensive path")
    p.add_argument("--reuse_exp", type=str, default=None, help="previous probe .npz: reuse its DDS predictions")
    p.add_argument("--cheap_norm", choices=["auto", "normalize", "center"], default="auto")
    p.add_argument("--out", type=str, required=True)
    main(p.parse_args())
