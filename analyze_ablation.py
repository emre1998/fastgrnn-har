"""
L-S-Q compression ablation for FastGRNN, per dataset (5 seeds, macro-F1).

Stages (each row adds one compression step to the previous):
  dense    : full-rank FastGRNN H=16, FP32           (Tier 1 baseline)
  low-rank : + low-rank r_w=2/r_u=8, FP32            (run_lowrank_stage.py)
  +sparse  : + IHT sparsity 0.5, FP32                (deploy sparse_fp32_f1)
  +Q15     : + calibrated Q15 (deployed)             (deploy q15_f1)

Shows where accuracy AND variance enter -- supports the Failure Analysis section
(the HAPT seed-collapse and whether it originates at the L or S step).
"""
import glob
import json
import numpy as np
from pathlib import Path

EXP = Path("experiments")
DATASETS = ["hapt", "wisdm", "pamap2"]


def load_f1(pattern, key=None):
    vals = []
    for f in sorted(glob.glob(str(EXP / pattern))):
        r = json.load(open(f))
        vals.append(r[key] if key else r["test_macro_f1"])
    return np.array(vals)


def dense(ds):
    # Tier1 full-rank FastGRNN: hapt is unprefixed, others tagged
    pat = ("baseline_fastgrnn_h16_s*_e120.json" if ds == "hapt"
           else f"baseline_{ds}_fastgrnn_h16_s*_e120.json")
    return load_f1(pat)


def sparse_fp32(ds):
    return np.array([json.load(open(f))["fastgrnn"]["sparse_fp32_f1"]
                     for f in sorted(glob.glob(str(EXP / f"deploy_{ds}_s*.json")))])


def q15(ds):
    return np.array([json.load(open(f))["fastgrnn"]["q15_f1"]
                     for f in sorted(glob.glob(str(EXP / f"deploy_{ds}_s*.json")))])


def main():
    out = {}
    print("=== FastGRNN L-S-Q ablation (macro-F1, mean+/-std, worst seed) ===")
    print(f"{'dataset':8s} {'stage':10s} {'meanF1':>8s} {'std':>7s} {'worst':>7s}")
    for ds in DATASETS:
        stages = {"dense": dense(ds), "low-rank": load_f1(f"lowrank_{ds}_s*.json"),
                  "+sparse": sparse_fp32(ds), "+Q15": q15(ds)}
        out[ds] = {}
        for name, v in stages.items():
            if len(v) == 0:
                print(f"{ds:8s} {name:10s}   (missing)")
                continue
            out[ds][name] = {"mean": float(v.mean()), "std": float(v.std()),
                             "worst": float(v.min()), "n": len(v)}
            print(f"{ds:8s} {name:10s} {v.mean():8.3f} {v.std():7.3f} {v.min():7.3f}")
        print()
    with open(EXP / "ablation_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: experiments/ablation_summary.json")


if __name__ == "__main__":
    main()
