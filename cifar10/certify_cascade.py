"""Step 4: certify with dds / cheap / cascade base classifiers.

Same CERTIFY as cifar10/certify.py (core.Smooth, Clopper-Pearson), plus two extra
columns: fraction of noise samples sent to the expensive path, and wall-clock time.

Examples (500 test images, N=10k, matching Dual RS's reproduced Carlini files):
  python cifar10/certify_cascade.py --mode dds     --sigma 0.25 --skip 20 --N 10000 --alpha 0.0005 --outfile out/dds_0.25.tsv
  python cifar10/certify_cascade.py --mode cheap   --sigma 0.25 --skip 20 --N 10000 --alpha 0.0005 --cheap_ckpt <ckpt> --outfile out/cheap_0.25.tsv
  python cifar10/certify_cascade.py --mode cascade --tau 0.95 --sigma 0.25 --skip 20 --N 10000 --alpha 0.0005 --cheap_ckpt <ckpt> --outfile out/cascade_0.25.tsv
"""
import argparse
import csv
import datetime
import os
import time

import torch
from torchvision import datasets, transforms

from core import Smooth
from DRM_cascade import CascadeRobustModel


def main(a):
    torch.manual_seed(a.seed + a.shard)
    model = CascadeRobustModel(a.sigma, mode=a.mode, tau=a.tau, cheap_ckpt=a.cheap_ckpt,
                               multistep=a.multistep, amp=a.amp)
    ds = datasets.CIFAR10(a.data_dir, train=False, download=True, transform=transforms.ToTensor())
    smoothed = Smooth(model, 10, a.sigma, model.t)

    # images to certify: idx % skip == 0, optionally truncated, then split across shards (GPUs)
    todo = [i for i in range(len(ds)) if i % a.skip == 0][: a.max_images]
    todo = todo[a.shard:: a.num_shards]

    done, correct = set(), 0
    if a.resume and os.path.exists(a.outfile):        # continue an interrupted run
        rows = list(csv.DictReader(open(a.outfile), delimiter="\t"))
        done = {int(r["idx"]) for r in rows}
        correct = sum(r["label"] == r["predict"] for r in rows)
        f = open(a.outfile, "a")
        print(f"resume: {len(done)} images already done", flush=True)
    else:
        f = open(a.outfile, "w")
        print("idx\tlabel\tpredict\tradius\tcorrect\texp_frac\ttime", file=f, flush=True)
    total = len(done)
    for i in todo:
        if i in done:
            continue
        x, label = ds[i]
        x = x.cuda()
        model.reset_stats()
        torch.cuda.synchronize(); t0 = time.time()
        pred, radius = smoothed.certify(x, a.N0, a.N, a.alpha, a.batch_size)
        torch.cuda.synchronize(); dt = time.time() - t0
        correct += int(pred == label); total += 1
        frac = model.n_expensive / max(model.n_total, 1)
        print(f"{i}\t{label}\t{pred}\t{radius:.3}\t{correct}\t{frac:.4f}\t"
              f"{datetime.timedelta(seconds=dt)}", file=f, flush=True)
        if total % 10 == 0:
            print(f"[shard {a.shard}] {total}/{len(todo)}  last {dt:.1f}s  exp_frac {frac:.3f}", flush=True)
    print(f"mode={a.mode} sigma={a.sigma} clean acc of g: {correct / max(total, 1):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dds", "cheap", "cascade"], default="cascade")
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--cheap_ckpt", type=str, default=None)
    p.add_argument("--multistep", action="store_true")
    p.add_argument("--skip", type=int, default=20)
    p.add_argument("--N0", type=int, default=100)
    p.add_argument("--N", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.0005)
    p.add_argument("--data_dir", type=str, default="data/cifar10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outfile", type=str, required=True)
    p.add_argument("--max_images", type=int, default=None, help="certify only the first K selected images")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--amp", action="store_true", help="fp16 autocast on the expensive path")
    p.add_argument("--resume", action="store_true", help="append to outfile, skip finished idx")
    main(p.parse_args())
