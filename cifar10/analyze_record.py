"""CPU-only: Cohen CERTIFY for dds / cheap / cascade(tau...) from records made by record.py.

Same procedure as core.Smooth.certify: top class from the first N0 predictions, Clopper-Pearson
lower bound on p_A from the next N, radius = sigma * Phi^-1(pA_lower), abstain if < 0.5.
Time per image = cost model  t_cheap + exp_frac * t_exp  (both measured per image in record.py).

Examples:
  python cifar10/analyze_record.py out/rec_0.25 --taus 0.5 0.7 0.9 0.99 --tau_file out/tau_0.25.txt
  # new cheap model recorded with --only_cheap into out/rec_0.25_smoothadv:
  python cifar10/analyze_record.py out/rec_0.25 --cheap_rec out/rec_0.25_smoothadv --tau_file ...
"""
import argparse
import glob
import os

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

RADII = [0.0, 0.25, 0.5, 0.75, 1.0]


def certify(pred, n0, n, alpha, sigma):
    c_a = int(np.bincount(pred[:n0], minlength=10).argmax())
    n_a = int(np.sum(pred[n0:n0 + n] == c_a))
    p_lo = proportion_confint(n_a, n, alpha=2 * alpha, method="beta")[0]
    return (-1, 0.0) if p_lo < 0.5 else (c_a, sigma * norm.ppf(p_lo))


def main(a):
    files = sorted(glob.glob(os.path.join(a.rec, "*.npz")), key=lambda f: int(os.path.basename(f)[:-4]))
    assert files, f"no records in {a.rec}"
    a.sigma = float(np.load(files[0])["sigma"]) if a.sigma is None else a.sigma
    taus = list(a.taus)
    if a.tau_file:
        taus.append(float(open(a.tau_file).read()))
    modes = ["dds", "cheap"] + [f"tau={t:g}" for t in taus]
    res = {m: [] for m in modes}   # rows: (idx, label, pred, radius, exp_frac, sec)
    for f in files:
        r = np.load(f)
        exp, label, n0, n = r["exp"], int(r["label"]), int(r["N0"]), int(r["N"])
        if a.cheap_rec:
            r2 = np.load(os.path.join(a.cheap_rec, os.path.basename(f)))
            cheap, conf, t_cheap = r2["cheap"], r2["conf"].astype(np.float32), float(r2["t_cheap"])
        else:
            cheap, conf, t_cheap = r["cheap"], r["conf"].astype(np.float32), float(r["t_cheap"])
        t_exp = float(r["t_exp"])
        preds = {"dds": (exp, 1.0), "cheap": (cheap, 0.0)}
        for t, m in zip(taus, modes[2:]):
            hard = conf < t
            preds[m] = (np.where(hard, exp, cheap), float(hard.mean()))
        for m, (p, ef) in preds.items():
            c, rad = certify(p, n0, n, a.alpha, a.sigma)
            sec = t_exp if m == "dds" else t_cheap + ef * t_exp
            res[m].append((int(r["idx"]), label, c, rad, ef, sec))

    n_img = len(files)
    print(f"{n_img} images | sigma={a.sigma} | alpha={a.alpha}" + (f" | cheap from {a.cheap_rec}" if a.cheap_rec else ""))
    print("mode".ljust(14) + "".join(f"{'r=' + str(x):>8}" for x in RADII) + "     ACR  sec/img  speedup  exp_frac")
    t_dds = np.mean([row[5] for row in res["dds"]])
    base = {row[0]: row for row in res["dds"]}
    for m in modes:
        rows = res[m]
        ok = lambda row, rad: row[2] == row[1] and row[3] >= rad
        accs = [100 * np.mean([ok(row, rad) for row in rows]) for rad in RADII]
        acr = np.mean([row[3] if row[2] == row[1] else 0.0 for row in rows])
        sec = np.mean([row[5] for row in rows])
        print(m.ljust(14) + "".join(f"{x:8.1f}" for x in accs)
              + f"  {acr:6.3f}  {sec:7.2f}  {t_dds / sec:6.2f}x  {np.mean([row[4] for row in rows]):8.3f}")
        if a.paired and m != "dds":   # images certified by dds at r but not by this mode, and vice versa
            for rad in [0.25, 0.5, 0.75]:
                lost = sum(ok(base[row[0]], rad) and not ok(row, rad) for row in rows)
                won = sum(ok(row, rad) and not ok(base[row[0]], rad) for row in rows)
                print(f"{'':14}  r={rad}: lost {lost}, gained {won} images vs dds")
        if a.tsv_dir:
            os.makedirs(a.tsv_dir, exist_ok=True)
            with open(os.path.join(a.tsv_dir, f"{m.replace('=', '')}_{a.sigma:.2f}.tsv"), "w") as fo:
                print("idx\tlabel\tpredict\tradius\tcorrect\texp_frac\ttime", file=fo)
                for row in rows:
                    print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.3}\t{int(row[2] == row[1])}\t{row[4]:.4f}\t"
                          f"0:00:{row[5]:09.6f}", file=fo)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("rec", type=str)
    p.add_argument("--cheap_rec", type=str, default=None)
    p.add_argument("--sigma", type=float, default=None, help="default: read from the records")
    p.add_argument("--alpha", type=float, default=0.0005)
    p.add_argument("--taus", type=float, nargs="*", default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    p.add_argument("--tau_file", type=str, default=None)
    p.add_argument("--paired", action="store_true", help="per-radius lost/gained images vs dds")
    p.add_argument("--tsv_dir", type=str, default=None, help="also write compare.py-style tsv files")
    main(p.parse_args())
