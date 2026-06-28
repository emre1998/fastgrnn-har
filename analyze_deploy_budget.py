"""
Aggregate the per-seed deploy-budget results into the cross-dataset table.

Answers, at the actual on-device operating point (~283 nonzero on HAPT/WISDM,
larger head on PAMAP2): which cell wins, how reliable (std), and how lossless
is Q15 -- across all three datasets. Also flags whether FastGRNN's L-S-Q
instability (the seed-collapse seen on HAPT) reproduces elsewhere.
"""
import glob
import json
import numpy as np
from pathlib import Path

EXP = Path("experiments")
CELLS = ["gru", "lstm", "fastgrnn"]
DATASETS = ["hapt", "wisdm", "pamap2"]


def load(ds):
    runs = []
    for f in sorted(glob.glob(str(EXP / f"deploy_{ds}_s*.json"))):
        runs.append(json.load(open(f)))
    return runs


def fp32_of(cell_res):
    return cell_res.get("sparse_fp32_f1", cell_res.get("fp32_f1"))


def main():
    summary = {}
    print("=== DEPLOY-BUDGET across datasets (matched nonzero, FP32 + Q15) ===")
    for ds in DATASETS:
        runs = load(ds)
        if not runs:
            continue
        summary[ds] = {}
        nz = runs[0]["fastgrnn"]["total_nonzero"]
        print(f"\n--- {ds.upper()}  (~{nz} nonzero / {nz*2} bytes, n={len(runs)} seeds) ---")
        print(f"{'cell':9s} {'FP32 mean':>10s} {'std':>7s} {'Q15 mean':>9s} {'std':>7s} "
              f"{'worst':>7s} {'dF1(Q)':>7s}")
        for c in CELLS:
            fp = np.array([fp32_of(r[c]) for r in runs])
            q = np.array([r[c]["q15_f1"] for r in runs])
            summary[ds][c] = {"fp32_mean": float(fp.mean()), "fp32_std": float(fp.std()),
                              "q15_mean": float(q.mean()), "q15_std": float(q.std()),
                              "q15_worst": float(q.min()),
                              "q15_drop": float(fp.mean() - q.mean()),
                              "fp32_per_seed": fp.tolist(), "q15_per_seed": q.tolist(),
                              "total_nonzero": int(runs[0][c]["total_nonzero"])}
            s = summary[ds][c]
            print(f"{c:9s} {s['fp32_mean']:10.3f} {s['fp32_std']:7.3f} "
                  f"{s['q15_mean']:9.3f} {s['q15_std']:7.3f} {s['q15_worst']:7.3f} "
                  f"{s['q15_drop']:+7.3f}")
        # winner + stability note
        best = max(CELLS, key=lambda c: summary[ds][c]["q15_mean"])
        stab = min(CELLS, key=lambda c: summary[ds][c]["q15_std"])
        print(f"  winner(Q15 mean): {best} | most stable: {stab} "
              f"(std {summary[ds][stab]['q15_std']:.3f})")

    with open(EXP / "deploy_budget_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: experiments/deploy_budget_summary.json")


if __name__ == "__main__":
    main()
