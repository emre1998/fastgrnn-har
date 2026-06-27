"""
Per-class macro-F1 tables (seed-averaged) for each dataset, Tier 1 (H=16).

Reads the per_class_f1 already stored in each result JSON -- no retraining.
Supports the paper's credibility argument: low absolute scores (esp. PAMAP2)
are driven by the single-accelerometer / class-imbalance setup and are SHARED
across cells, not a cell-specific artifact.

Output: experiments/perclass_{dataset}.json  +  printed tables.
"""
import glob
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

EXP = Path("experiments")
CELLS = ["gru", "lstm", "fastgrnn"]
DATASETS = {
    "hapt":   "baseline_{cell}_h16_s*_e120.json",
    "wisdm":  "baseline_wisdm_{cell}_h16_s*_e120.json",
    "pamap2": "baseline_pamap2_{cell}_h16_s*_e120.json",
}


def perclass(cell, pattern):
    acc = defaultdict(list)
    for f in glob.glob(str(EXP / pattern.format(cell=cell))):
        r = json.load(open(f))
        for k, v in r["per_class_f1"].items():
            acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


for ds, pat in DATASETS.items():
    table = {c: perclass(c, pat) for c in CELLS}
    if not table["gru"]:
        continue
    classes = list(table["gru"].keys())
    print(f"\n=== {ds.upper()} per-class macro-F1 (seed-averaged, H=16) ===")
    print(f"{'class':18s} {'GRU':>6s} {'LSTM':>6s} {'FastG':>6s}")
    for k in sorted(classes, key=lambda k: -table["gru"][k]):
        print(f"{k:18s} {table['gru'][k]:6.3f} {table['lstm'][k]:6.3f} "
              f"{table['fastgrnn'][k]:6.3f}")
    with open(EXP / f"perclass_{ds}.json", "w") as f:
        json.dump(table, f, indent=2)
    print(f"Saved: experiments/perclass_{ds}.json")
