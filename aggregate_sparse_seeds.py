"""
Summarize sparse sp50 multi-seed results.
"""

import json
import statistics
from pathlib import Path

EXP = Path("experiments")
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

runs = []
for p in sorted(EXP.glob("sparse_h16_rw2_ru8_sp50_s*_e100.json")):
    runs.append(json.loads(p.read_text()))

if not runs:
    print("No sp50 results found.")
    raise SystemExit

# Multi-seed summary
accs = [r["test_accuracy"] for r in runs]
f1s  = [r["test_macro_f1"] for r in runs]
eff_params = [r["effective_params"] for r in runs]
best_epochs = [r["best_epoch"] for r in runs]
actual_sparsity = [sum(r["actual_sparsity"].values())/4 for r in runs]

print("=" * 90)
print(" SPARSE 50% MULTI-SEED SUMMARY")
print("=" * 90)
print(f"N seeds: {len(runs)}")
print(f"\n{'metric':<25} {'mean':>10} {'std':>10}  raw")
print("-" * 90)
def show(name, vals):
    m = statistics.mean(vals); s = statistics.stdev(vals) if len(vals)>1 else 0
    raw = [f"{v:.4f}" if v < 1.5 else f"{v}" for v in vals]
    print(f"{name:<25} {m:>10.4f} {s:>10.4f}  {raw}")

show("Test Accuracy",     accs)
show("Test Macro-F1",     f1s)
show("Effective params",  eff_params)
show("Actual sparsity",   actual_sparsity)
show("Best epoch",        best_epochs)

# Per-class mean+std
print("\nPer-class F1 (mean +/- std):")
for cls in CLASS_NAMES:
    vals = [r["per_class_f1"][cls] for r in runs]
    m = statistics.mean(vals); s = statistics.stdev(vals) if len(vals)>1 else 0
    raw_str = " [" + ", ".join(f"{v:.3f}" for v in vals) + "]"
    print(f"  {cls:<12} {m:.4f} +/- {s:.4f}{raw_str}")

# Comparison vs dense r_u=8 multi-seed
print("\n" + "=" * 90)
print(" COMPARISON")
print("=" * 90)
try:
    ru8 = json.loads((EXP / "multiseed_summary.json").read_text())
    ru8_winner = ru8["winner"]
    print(f"Dense r_u=8 multi-seed:  F1 = {ru8_winner['f1_mean']:.4f} +/- {ru8_winner['f1_std']:.4f}")
    sp50_mean, sp50_std = statistics.mean(f1s), statistics.stdev(f1s) if len(f1s)>1 else 0
    print(f"Sparse sp50 multi-seed:  F1 = {sp50_mean:.4f} +/- {sp50_std:.4f}")
    delta = sp50_mean - ru8_winner['f1_mean']
    pooled_std = ((sp50_std**2 + ru8_winner['f1_std']**2) / 2) ** 0.5
    sigma = delta / pooled_std if pooled_std > 1e-9 else float('inf')
    print(f"Delta: {delta:+.4f}, ~{sigma:.2f} sigma")
except FileNotFoundError:
    pass
