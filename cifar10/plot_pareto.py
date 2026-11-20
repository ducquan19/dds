"""Pareto plot: certified robustness vs speed-up, from analyze_record.py --summary_csv files.

One curve per summary CSV (e.g. one per cheap model); points = dds (1x), cascade at each tau,
and the cheap model alone. The calibrated tau (from --tau_file) is drawn as a star.

  python cifar10/plot_pareto.py cohen=out/summary_0.25_cohen.csv [smoothadv=out/summary_0.25_smoothadv.csv] \
      --out out/pareto_0.25 --radius 0.5
Writes <out>.pdf and <out>.png.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical order (fixed, never cycled); markers give a non-colour channel
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k != "mode":
                r[k] = float(v) if v not in ("", "None") else None
    return rows


def main(a):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    panels = [("acr", "ACR")] + [(f"acc_r{r}", f"Certified acc. (%) at r = {r}") for r in a.radius]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 2.8), constrained_layout=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    curves = [(s.split("=", 1)[0], load(s.split("=", 1)[1])) for s in a.files]
    assert len(curves) <= len(SERIES), "more than 4 curves: use small multiples"

    for ax, (key, label) in zip(axes, panels):
        dds = next(r for r in curves[0][1] if r["mode"] == "dds")
        ax.axhline(dds[key], color=INK2, lw=1, ls="--", zorder=1)
        ax.annotate("DDS (always diffusion)", (1, dds[key]), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=INK2)
        for i, (name, rows) in enumerate(curves):
            col, mk = SERIES[i], MARKERS[i]
            casc = sorted([r for r in rows if r["mode"] in ("dds",) or r["mode"].startswith("tau=")],
                          key=lambda r: r["speedup"])
            xs, ys = [r["speedup"] for r in casc], [r[key] for r in casc]
            ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=5, mec="white", mew=1, zorder=3,
                    label=f"cascade ({name})")
            cal = [r for r in rows if r["calibrated"] == 1]
            for r in cal:
                ax.plot(r["speedup"], r[key], marker="*", ms=13, color=col, mec="white", mew=1, zorder=4)
                ax.annotate(f"calibrated\n{r['mode'].replace('tau=', 'τ=')}", (r["speedup"], r[key]),
                            xytext=(36, -6), textcoords="offset points", ha="left", va="top",
                            fontsize=7.5, color=INK,
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=6))
            ch = next((r for r in rows if r["mode"] == "cheap"), None)
            if ch:
                ax.plot(ch["speedup"], ch[key], marker=mk, ms=7, mfc="white", mec=col, mew=1.5, zorder=3, ls="none")
                if i == 0:
                    ax.annotate("cheap only", (ch["speedup"], ch[key]), xytext=(-4, 7),
                                textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
            if len(curves) > 1:   # direct label at the fast end of each curve
                ax.annotate(name, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20, 40])
        ax.set_xticklabels(["1×", "2×", "3×", "5×", "10×", "20×", "40×"])
        ax.minorticks_off()
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel("Speed-up over DDS (GPU time per image)")
        ax.set_ylabel(label)
    if len(curves) > 1:
        axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200)
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="name=summary.csv (one curve each, <= 4)")
    p.add_argument("--radius", type=float, nargs="*", default=[0.5])
    p.add_argument("--out", type=str, default="out/pareto")
    main(p.parse_args())
