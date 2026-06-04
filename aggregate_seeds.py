"""
Aggregate the multi-seed sweep results and report mean +/- std per config.
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

EXP = Path("experiments")
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

# r_u -> list of run dicts
by_ru = defaultdict(list)

for p in sorted(EXP.glob("fastgrnn_h16_rw2_ru*_s*_e100.json")):
    data = json.loads(p.read_text())
    by_ru[data["r_u"]].append(data)

if not by_ru:
    print("No multi-seed results found. Run run_multiseed_sweep.py first.")
    raise SystemExit

print("=" * 110)
print(" MULTI-SEED SUMMARY (mean +/- std, N seeds)")
print("=" * 110)
print(f"{'r_u':>4} {'N':>3} {'Params':>8}  {'Acc mean':>10} {'Acc std':>9}  {'F1 mean':>10} {'F1 std':>9}")
print("-" * 110)
rows = []
for r_u in sorted(by_ru):
    runs = by_ru[r_u]
    n = len(runs)
    accs = [r["test_accuracy"] for r in runs]
    f1s  = [r["test_macro_f1"] for r in runs]
    params = runs[0]["n_params"]
    acc_mean, acc_std = statistics.mean(accs), statistics.stdev(accs) if n > 1 else 0.0
    f1_mean,  f1_std  = statistics.mean(f1s),  statistics.stdev(f1s)  if n > 1 else 0.0
    rows.append({"r_u": r_u, "n": n, "params": params,
                 "acc_mean": acc_mean, "acc_std": acc_std,
                 "f1_mean": f1_mean, "f1_std": f1_std,
                 "accs": accs, "f1s": f1s})
    print(f"{r_u:>4} {n:>3} {params:>8}  {acc_mean:>10.4f} {acc_std:>9.4f}  {f1_mean:>10.4f} {f1_std:>9.4f}")

# Per-class mean+std
print("\n" + "=" * 110)
print(" PER-CLASS F1 (mean +/- std)")
print("=" * 110)
print(f"{'Class':<12}", end="")
for r_u in sorted(by_ru):
    print(f" {f'r_u={r_u}':>16}", end="")
print()
print("-" * (12 + 17 * len(by_ru)))
for cls in CLASS_NAMES:
    print(f"{cls:<12}", end="")
    for r_u in sorted(by_ru):
        vals = [r["per_class_f1"][cls] for r in by_ru[r_u]]
        m = statistics.mean(vals); s = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f" {m:.3f}+-{s:.3f}".rjust(17), end="")
    print()

# All raw numbers
print("\n" + "=" * 110)
print(" RAW NUMBERS (per-seed F1)")
print("=" * 110)
for row in rows:
    print(f"r_u={row['r_u']:>2}: F1 = {[f'{f:.4f}' for f in row['f1s']]}")

# Official selection: highest mean F1
best = max(rows, key=lambda r: r["f1_mean"])
print(f"\n=== WINNER: r_u={best['r_u']} ===")
print(f"  Test F1 mean +/- std:  {best['f1_mean']:.4f} +/- {best['f1_std']:.4f}")
print(f"  Test Acc mean +/- std: {best['acc_mean']:.4f} +/- {best['acc_std']:.4f}")
print(f"  Parameters: {best['params']}")

# Simple significance check: how many sigma between the winner and the rest?
print("\n=== ROUGH SIGNIFICANCE CHECK ===")
for row in rows:
    if row["r_u"] == best["r_u"]:
        continue
    delta = best["f1_mean"] - row["f1_mean"]
    # Rough effect-size proxy: difference divided by pooled std (not a real t-test)
    pooled_std = ((best["f1_std"]**2 + row["f1_std"]**2) / 2) ** 0.5
    n_sigma = delta / pooled_std if pooled_std > 1e-9 else float("inf")
    print(f"  r_u={best['r_u']} vs r_u={row['r_u']}: delta={delta:+.4f}, ~{n_sigma:.1f} sigma")

# Save
out = {
    "configs": [
        {**{k: v for k, v in r.items() if k not in ("accs", "f1s")},
         "accs": r["accs"], "f1s": r["f1s"]}
        for r in rows
    ],
    "winner": {"r_u": best["r_u"], "f1_mean": best["f1_mean"], "f1_std": best["f1_std"]},
}
with open("experiments/multiseed_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: experiments/multiseed_summary.json")
