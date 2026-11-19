"""Step 5: certified accuracy at fixed radii + cost, for any number of result .tsv files.

Works with original certify.py output, certify_cascade.py output and the Carlini
reference files in Dual-Randomized-Smoothing/reproduce/data/carlini/.

Example:
  python cifar10/compare.py ref=../Dual-Randomized-Smoothing/reproduce/data/carlini/0.25.tsv \
      dds=out/dds_0.25.tsv cascade=out/cascade_0.25.tsv --skip 20
(--skip 20 keeps only idx % 20 == 0 from files that cover all 10k images, so all rows match.)
"""
import argparse
import csv
import glob


def secs(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def main(a):
    radii = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("name".ljust(16) + "n".rjust(6) + "".join(f"{'r=' + str(r):>9}" for r in radii)
          + "  sec/img  exp_frac")
    a.common = None
    if a.intersect:   # restrict every file to the idx present in all of them (partial runs)
        sets = []
        for spec in a.files:
            ps = sorted(glob.glob(spec.split("=", 1)[1])) or [spec.split("=", 1)[1]]
            sets.append({int(r["idx"]) for pth in ps for r in csv.DictReader(open(pth), delimiter="\t")})
        a.common = set.intersection(*sets)
    for spec in a.files:
        name, path = spec.split("=", 1)
        paths = sorted(glob.glob(path)) or [path]      # glob merges shards: dds=out/dds_*.tsv
        rows = [r for pth in paths for r in csv.DictReader(open(pth), delimiter="\t")
                if int(r["idx"]) % a.skip == 0]
        if a.common is not None:
            rows = [r for r in rows if int(r["idx"]) in a.common]
        n = len(rows)
        if n == 0:
            print(name.ljust(16) + "     0  (no rows)")
            continue
        n = len(rows)
        accs = [100 * sum(r["label"] == r["predict"] and float(r["radius"]) >= rad for r in rows) / n
                for rad in radii]
        t = sum(secs(r["time"]) for r in rows) / n
        ef = (sum(float(r["exp_frac"]) for r in rows) / n) if "exp_frac" in rows[0] else 1.0
        print(name.ljust(16) + str(n).rjust(6) + "".join(f"{x:9.1f}" for x in accs)
              + f"  {t:7.2f}  {ef:8.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="name=path.tsv")
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--intersect", action="store_true", help="only compare images present in every file")
    main(p.parse_args())
