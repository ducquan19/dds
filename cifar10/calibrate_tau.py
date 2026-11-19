"""Step 3: choose tau with a finite-sample guarantee (fixed-sequence testing, Learn-then-Test style).

Risk(tau) = P_{x, eps}( cheap conf >= tau  AND  cheap pred != expensive pred )
This bounds how much the cascade can lower p_A relative to the pure-DDS base classifier
(on average over the data distribution).

We scan tau from high (safe) to low and keep the lowest tau whose Clopper-Pearson upper
bound on Risk(tau) is <= delta. Fixed-sequence testing needs no multiplicity correction;
with prob >= 1 - alpha, the selected tau has Risk <= delta.

This guarantee is separate from the robustness certificate: the certificate is valid for
ANY tau fixed before certification. Calibrate on train images, certify on test images.

Example:  python cifar10/calibrate_tau.py probe_s0.25.npz --delta 0.01 --alpha 0.05
"""
import argparse
import numpy as np
from scipy.stats import beta


def cp_upper(k, n, alpha):
    return 1.0 if k == n else beta.ppf(1 - alpha, k + 1, n - k)


def main(a):
    d = np.load(a.probe)
    f = d["first"]                                 # one i.i.d. sample per image
    conf, cheap, exp = d["conf"][f], d["cheap"][f], d["exp"][f]
    n = len(conf)
    grid = np.round(np.arange(0.999, a.tau_min - 1e-9, -0.001), 3)

    chosen = None
    print(f"n={n}, delta={a.delta}, alpha={a.alpha}")
    for tau in grid:
        accept = conf >= tau
        k = int(np.sum(accept & (cheap != exp)))
        ucb = cp_upper(k, n, a.alpha)
        if ucb > a.delta:
            break
        chosen = (tau, accept.mean(), k / n, ucb)
    if chosen is None:
        print("No tau passes; use mode=dds or relax delta.")
        if a.out:
            open(a.out, "w").write("1.01\n")   # tau > 1 => cascade == dds (always expensive)
        return
    tau, acc, risk, ucb = chosen
    print(f"chosen tau={tau:.3f} | accept rate={acc:.3f} (expensive calls saved) "
          f"| empirical risk={risk:.4f} | UCB={ucb:.4f}")
    print(f"expected speed-up on the denoiser ~ {1 / max(1 - acc, 1e-6):.2f}x")
    if a.out:
        open(a.out, "w").write(f"{tau:.3f}\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("probe", type=str)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--tau_min", type=float, default=0.5)
    p.add_argument("--out", type=str, default=None, help="write chosen tau to this file")
    main(p.parse_args())
