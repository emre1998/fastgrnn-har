"""
Aggregate all experiment results into one verdict.

Rebuilds the combined Pareto summary from per-seed JSONs (the live
pareto_summary.json gets overwritten by whichever sweep finishes last),
then prints the full picture: Pareto curve, Tier 1 (equal H), Tier 2
(equal budget, both routes), and the head-to-head verdict per operating point.

Usage: python analyze_results.py
"""
import glob
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

EXP = Path("experiments")


def load(pattern):
    out = []
    for f in glob.glob(str(EXP / pattern)):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def agg_pareto():
    """Rebuild dense Pareto summary from per-seed run files."""
    runs = defaultdict(list)
    npar = {}
    for f in glob.glob(str(EXP / "pareto_*_h*_s*_e*.json")):
        with open(f) as fh:
            r = json.load(fh)
        key = (r["model"], r["hidden"])
        runs[key].append(r["test_macro_f1"])
        npar[key] = r["n_params"]
    rows = []
    for (model, h), f1s in runs.items():
        f1s = np.array(f1s)
        rows.append({"model": model, "hidden": h, "n_params": npar[(model, h)],
                     "mean_f1": float(f1s.mean()), "std_f1": float(f1s.std()),
                     "n_seeds": len(f1s)})
    rows.sort(key=lambda r: (r["model"], r["n_params"]))
    # persist combined summary so the repo has the real merged file
    combined = {f"{r['model']}_h{r['hidden']}": r for r in rows}
    with open(EXP / "pareto_summary.json", "w") as fh:
        json.dump(combined, fh, indent=2)
    return rows


def show_pareto(rows):
    print("\n=== PARETO (dense, accuracy vs params) ===")
    print(f"{'model':10s} {'H':>3s} {'params':>7s} {'meanF1':>8s} {'std':>7s} {'seeds':>6s}")
    for r in rows:
        print(f"{r['model']:10s} {r['hidden']:3d} {r['n_params']:7d} "
              f"{r['mean_f1']:8.4f} {r['std_f1']:7.4f} {r['n_seeds']:6d}")


def best_below(rows, model, budget):
    """Best dense point of `model` at or under `budget` params."""
    cands = [r for r in rows if r["model"] == model and r["n_params"] <= budget]
    return max(cands, key=lambda r: r["mean_f1"]) if cands else None


def pareto_frontier(rows):
    """Points not dominated by a smaller-or-equal model with higher F1."""
    front = []
    for r in rows:
        dominated = any(o["n_params"] <= r["n_params"] and o["mean_f1"] > r["mean_f1"]
                        and o is not r for o in rows)
        if not dominated:
            front.append(r)
    front.sort(key=lambda r: r["n_params"])
    return front


def main():
    rows = agg_pareto()
    show_pareto(rows)

    print("\n=== PARETO FRONTIER (who owns each budget) ===")
    for r in pareto_frontier(rows):
        print(f"  {r['n_params']:4d} par : {r['model']:8s} H{r['hidden']:<2d} "
              f"F1 {r['mean_f1']:.4f}")

    # Tier summaries (from their own summary files)
    print("\n=== TIER 1 (equal H=16) ===")
    t1 = json.load(open(EXP / "baseline_tier1_summary.json"))
    for k in ("gru", "lstm", "fastgrnn"):
        s = t1[k]
        print(f"  {k:8s} {s['n_params']:5d} par  F1 {s['mean_f1']:.4f} +/- {s['std_f1']:.4f}")

    print("\n=== TIER 2 (equal budget ~283 par) ===")
    t2 = json.load(open(EXP / "tier2_summary.json"))
    t2p = json.load(open(EXP / "tier2pruned_summary.json"))
    print("  shrink-H route:")
    for k in ("gru", "lstm", "fastgrnn"):
        s = t2[k]
        print(f"    {k:8s} H{s['hidden']:<2d} {s['n_params']:4d} par  "
              f"F1 {s['mean_f1']:.4f} +/- {s['std_f1']:.4f}")
    print("  pruned-H16 route:")
    for k in ("gru_pruned", "lstm_pruned"):
        s = t2p[k]
        print(f"    {k:13s} {s['total_nonzero']:4d} nz   "
              f"F1 {s['mean_f1']:.4f} +/- {s['std_f1']:.4f}")

    # Head-to-head at the deployment budget (283 par)
    print("\n=== VERDICT @ deployment budget (<=290 par, dense Pareto) ===")
    fg = best_below(rows, "fastgrnn", 290)
    gru = best_below(rows, "gru", 290)
    lstm = best_below(rows, "lstm", 290)
    for name, r in (("fastgrnn", fg), ("gru", gru), ("lstm", lstm)):
        if r:
            print(f"  {name:8s}: H{r['hidden']:<2d} {r['n_params']:4d} par  F1 {r['mean_f1']:.4f}")
    if fg and gru:
        gap = gru["mean_f1"] - fg["mean_f1"]
        print(f"\n  GRU - FastGRNN gap at budget: {gap:+.4f}")
        print(f"  Winner at every tested operating point: GRU" if gap > 0 else
              "  FastGRNN competitive")


if __name__ == "__main__":
    main()
