"""Record-once certification: one GPU pass gives dds / cheap / cascade(any tau) certificates.

For each certified test image, draw N0+N noise samples ONCE (CPU generator seeded by image idx,
so the exact same noise can be regenerated later) and store for every sample:
  exp   : prediction of the expensive path (one-step diffusion + ViT)      [uint8]
  cheap : prediction of the cheap path                                    [uint8]
  conf  : max-softmax of the cheap path                                   [float16]
plus measured GPU seconds of each path for this image. analyze_record.py then runs Cohen's
CERTIFY for every mode/tau on CPU. Each mode still sees N0+N i.i.d. noise samples, so every
certificate is valid; sharing the noise across modes makes the comparison paired.

--only_cheap re-records just the cheap path with another checkpoint on the SAME noise
(~cost of the cheap path only), to evaluate new cheap models without re-running diffusion.

Example (2 GPUs):
  CUDA_VISIBLE_DEVICES=0 python cifar10/record.py --sigma 0.25 --cheap_ckpt $CKPT --amp --shard 0 --num_shards 2 --out_dir out/rec_0.25
  CUDA_VISIBLE_DEVICES=1 python cifar10/record.py --sigma 0.25 --cheap_ckpt $CKPT --amp --shard 1 --num_shards 2 --out_dir out/rec_0.25
"""
import argparse
import os
import time

import numpy as np
import torch
from torchvision import datasets, transforms

from DRM_cascade import CascadeRobustModel


def noise_for(idx, n, shape, sigma, seed):
    g = torch.Generator().manual_seed(seed * 1_000_003 + idx)
    return torch.randn((n,) + tuple(shape), generator=g) * sigma


def main(a):
    model = CascadeRobustModel(a.sigma, mode="cheap" if a.only_cheap else "cascade", tau=1.01,
                               cheap_ckpt=a.cheap_ckpt, amp=a.amp, cheap_norm=a.cheap_norm)
    ds = datasets.CIFAR10(a.data_dir, train=False, download=True, transform=transforms.ToTensor())
    todo = [i for i in range(len(ds)) if i % a.skip == 0][: a.max_images][a.shard:: a.num_shards]
    os.makedirs(a.out_dir, exist_ok=True)
    n_tot = a.N0 + a.N
    done = 0
    t_start = time.time()
    for i in todo:
        if a.time_budget and time.time() - t_start > a.time_budget * 3600:
            print(f"[shard {a.shard}] time budget {a.time_budget} h reached after {done}/{len(todo)}; "
                  f"rerun to resume", flush=True)
            break
        path = os.path.join(a.out_dir, f"{i}.npz")
        if os.path.exists(path):                      # resumable
            done += 1
            continue
        x, label = ds[i]
        xn = x.unsqueeze(0) + noise_for(i, n_tot, x.shape, a.sigma, a.seed)
        conf, cheap, exp = [], [], []
        t_cheap = t_exp = 0.0
        for chunk in xn.split(a.batch_size):
            c = chunk.cuda(non_blocking=True)
            torch.cuda.synchronize(); t0 = time.time()
            cf, cp = model.cheap_logits(c).softmax(1).max(1)
            torch.cuda.synchronize(); t_cheap += time.time() - t0
            conf.append(cf.half().cpu()); cheap.append(cp.to(torch.uint8).cpu())
            if not a.only_cheap:
                t0 = time.time()
                ep = model.expensive_logits(c).argmax(1)
                torch.cuda.synchronize(); t_exp += time.time() - t0
                exp.append(ep.to(torch.uint8).cpu())
        rec = dict(idx=i, label=label, sigma=a.sigma, N0=a.N0, N=a.N, seed=a.seed, amp=a.amp,
                   conf=torch.cat(conf).numpy(), cheap=torch.cat(cheap).numpy(), t_cheap=t_cheap)
        if not a.only_cheap:
            rec.update(exp=torch.cat(exp).numpy(), t_exp=t_exp)
        np.savez(path + ".tmp.npz", **rec)
        os.replace(path + ".tmp.npz", path)          # atomic: no half-written files on preemption
        done += 1
        if done % 10 == 0 or done == len(todo):
            print(f"[shard {a.shard}] {done}/{len(todo)}  t_cheap {t_cheap:.1f}s  t_exp {t_exp:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--cheap_ckpt", type=str, required=True)
    p.add_argument("--only_cheap", action="store_true")
    p.add_argument("--cheap_norm", choices=["auto", "normalize", "center"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--skip", type=int, default=20)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--N0", type=int, default=100)
    p.add_argument("--N", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="data/cifar10")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--time_budget", type=float, default=None,
                   help="hours; stop cleanly before starting a new image after this (Kaggle 12 h limit)")
    main(p.parse_args())
