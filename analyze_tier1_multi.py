"""
Cross-dataset Tier 1 (matched H=16) aggregation: HAPT + WISDM + PAMAP2.

Rebuilds per-cell mean/std from the per-seed JSONs for each dataset and prints
the combined table that answers the single-dataset reviewer objection.
"""
import glob
import json
import numpy as np
from pathlib import Path

EXP = Path("experiments")
CELLS = ["gru", "lstm", "fastgrnn"]
# dataset -> filename glob for per-seed Tier1 JSONs (HAPT is untagged)
DATASETS = {
    "HAPT":   "baseline_{cell}_h16_s*_e120.json",
    "WISDM":  "baseline_wisdm_{cell}_h16_s*_e120.json",
    "PAMAP2": "baseline_pamap2_{cell}_h16_s*_e120.json",
}


def collect(cell, pattern):
    f1s, npar = [], None
    for f in glob.glob(str(EXP / pattern.format(cell=cell))):
        r = json.load(open(f))
        f1s.append(r["test_macro_f1"]); npar = r["n_params"]
    return np.array(sorted(f1s)), npar


def main():
    table = {}
    for ds, pat in DATASETS.items():
        table[ds] = {}
        for cell in CELLS:
            f1s, npar = collect(cell, pat)
            if len(f1s):
                table[ds][cell] = {"mean": float(f1s.mean()), "std": float(f1s.std()),
                                   "n": len(f1s), "n_params": npar,
                                   "per_seed": f1s.tolist()}

    print("\n=== TIER 1 ACROSS DATASETS (matched H=16, FP32, macro-F1) ===")
    print(f"{'dataset':8s} {'GRU':>16s} {'LSTM':>16s} {'FastGRNN':>16s}   winner")
    for ds in DATASETS:
        cells = table[ds]
        cellstr = {}
        best, bestf1 = None, -1
        for cell in CELLS:
            if cell in cells:
                m, s = cells[cell]["mean"], cells[cell]["std"]
                cellstr[cell] = f"{m:.3f}+/-{s:.3f}"
                if m > bestf1:
                    best, bestf1 = cell, m
            else:
                cellstr[cell] = "--"
        print(f"{ds:8s} {cellstr['gru']:>16s} {cellstr['lstm']:>16s} "
              f"{cellstr['fastgrnn']:>16s}   {best}")

    # rank + stability summary
    print("\n=== Per-dataset ranking (1=best mean F1) and stability (std) ===")
    for ds in DATASETS:
        cells = table[ds]
        order = sorted([c for c in CELLS if c in cells],
                       key=lambda c: -cells[c]["mean"])
        most_stable = min(cells, key=lambda c: cells[c]["std"])
        rank = " > ".join(order)
        print(f"  {ds:8s}: {rank}   | most stable: {most_stable} "
              f"(std {cells[most_stable]['std']:.3f})")

    with open(EXP / "tier1_multidataset_summary.json", "w") as f:
        json.dump(table, f, indent=2)
    print("\nSaved: experiments/tier1_multidataset_summary.json")


if __name__ == "__main__":
    main()
