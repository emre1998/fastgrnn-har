"""
Combine the two compression routes for GRU/LSTM at the deployment byte budget and
report each baseline's BEST route (shrink-H vs pruned-H16), per dataset, vs FastGRNN.

shrink-H : deploy_rnn200_{ds}_s*.json  (fair 200-epoch dense small model + weight-Q15)
pruned   : tier2pruned_{ds}_{cell}_h16_s*_e200.json (H16 magnitude-pruned + weight-Q15)
FastGRNN : deploy_{ds}_s*.json (low-rank + IHT + calibrated Q15)
"""
import glob
import json
import numpy as np
from pathlib import Path

EXP = Path("experiments")
DATASETS = ["hapt", "wisdm", "pamap2"]


def arr(vals):
    return np.array(vals)


def main():
    out = {}
    print("=== DEPLOYMENT BUDGET: best route per baseline (Q15, 5 seeds) ===")
    for ds in DATASETS:
        fg = arr([json.load(open(f))["fastgrnn"]["q15_f1"]
                  for f in sorted(glob.glob(str(EXP / f"deploy_{ds}_s*.json")))])
        row = {"fastgrnn": {"mean": float(fg.mean()), "std": float(fg.std())}}
        print(f"\n--- {ds.upper()} ---   FastGRNN {fg.mean():.3f}+/-{fg.std():.3f}")
        for c in ["gru", "lstm"]:
            sh = arr([json.load(open(f))[c]["q15_f1"]
                      for f in sorted(glob.glob(str(EXP / f"deploy_rnn200_{ds}_s*.json")))])
            pr = arr([json.load(open(f))["q15_macro_f1"]
                      for f in sorted(glob.glob(str(EXP / f"tier2pruned_{ds}_{c}_h16_s*_e200.json")))])
            best = "shrink" if sh.mean() >= pr.mean() else "pruned"
            bestv = max(sh.mean(), pr.mean())
            row[c] = {"shrink_mean": float(sh.mean()), "shrink_std": float(sh.std()),
                      "pruned_mean": float(pr.mean()), "pruned_std": float(pr.std()),
                      "best_route": best, "best_mean": float(bestv)}
            print(f"  {c:4s}  shrink {sh.mean():.3f}+/-{sh.std():.3f}   "
                  f"pruned {pr.mean():.3f}+/-{pr.std():.3f}   -> best={best} ({bestv:.3f})")
        # verdict at budget
        gru_best = row["gru"]["best_mean"]; lstm_best = row["lstm"]["best_mean"]
        winner = max([("fastgrnn", fg.mean()), ("gru", gru_best), ("lstm", lstm_best)],
                     key=lambda kv: kv[1])
        print(f"  => budget winner: {winner[0]} ({winner[1]:.3f})")
        row["winner"] = winner[0]
        out[ds] = row

    with open(EXP / "best_route_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: experiments/best_route_summary.json")


if __name__ == "__main__":
    main()
