"""
Compare the pruned route re-run against the committed numbers, and show what it
does to the equal-byte comparison.

The reproducibility audit found this route unstable: drift was never zero on it
and reached +0.248 / -0.268 on single seeds. It also happens to be the GRU's
winning route on HAPT and WISDM, so the equal-byte margins rest on it. This
re-runs all five seeds with the thread count pinned and asks whether the margins
survive.

  python analyze_pruned_v2.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

EXP = Path("experiments")
DATASETS = ["hapt", "wisdm", "pamap2"]
CELLS = ["gru", "lstm"]


def seeds(tag: str, ds: str, cell: str) -> dict[int, float]:
    out = {}
    for p in sorted(glob.glob(str(EXP / f"tier2pruned_{ds}{tag}_{cell}_h16_s*_e200.json"))):
        s = int(p.split("_s")[-1].split("_")[0])
        with open(p, encoding="utf-8") as f:
            out[s] = json.load(f)["q15_macro_f1"]
    return out


def shrink_mean(ds: str, cell: str) -> float:
    v = [json.load(open(p, encoding="utf-8"))[cell]["q15_f1"]
         for p in sorted(glob.glob(str(EXP / f"deploy_rnn200_{ds}_s*.json")))]
    return float(np.mean(v))


def fastgrnn_mean(ds: str) -> float:
    v = [json.load(open(p, encoding="utf-8"))["fastgrnn"]["q15_f1"]
         for p in sorted(glob.glob(str(EXP / f"deploy_{ds}_s*.json")))]
    return float(np.mean(v))


print("=" * 76)
print("PRUNED ROUTE RE-RUN, threads pinned  (committed  vs  V2)")
print("=" * 76)

for ds in DATASETS:
    print(f"\n--- {ds.upper()} ---")
    fg = fastgrnn_mean(ds)
    print(f"  FastGRNN (unchanged, reproduces bit-exactly): {fg:.4f}")

    for cell in CELLS:
        old, new = seeds("", ds, cell), seeds("V2", ds, cell)
        common = sorted(set(old) & set(new))
        if not common:
            print(f"  {cell:4s} pruned: (V2 pending)")
            continue

        o = np.array([old[s] for s in common])
        n = np.array([new[s] for s in common])
        sh = shrink_mean(ds, cell)

        print(f"  {cell:4s} pruned  committed {o.mean():.4f}±{o.std():.4f}   "
              f"V2 {n.mean():.4f}±{n.std():.4f}   delta {n.mean()-o.mean():+.4f}"
              f"   (n={len(common)}/5)")
        if len(common) < 5:
            print(f"       seeds compared: {common}")

        # best-of-two-routes is what the equal-byte table actually uses
        best_old, route_old = (o.mean(), "pruned") if o.mean() >= sh else (sh, "shrink")
        best_new, route_new = (n.mean(), "pruned") if n.mean() >= sh else (sh, "shrink")
        if cell == "gru":
            m_old, m_new = fg - best_old, fg - best_new
            flip = " <-- route changes" if route_old != route_new else ""
            print(f"       best route  {route_old} {best_old:.4f}  ->  "
                  f"{route_new} {best_new:.4f}{flip}")
            print(f"       FastGRNN margin over GRU  {m_old:+.4f}  ->  {m_new:+.4f}")
            if np.sign(m_old) != np.sign(m_new):
                print("       WINNER CHANGES on this dataset")

print()
print("=" * 76)
print("Reading it: the FastGRNN column does not move -- that route reproduces")
print("bit-exactly. Any change in a margin comes from the baseline side, so a")
print("margin that grows means the committed comparison was too generous to the")
print("baseline, and one that shrinks means the opposite.")
print("Nothing here changes a hardware number.")
