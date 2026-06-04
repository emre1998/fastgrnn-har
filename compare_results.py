"""
Lay all experiment results side by side and cross-compare.

Automatically picks up every fastgrnn_*.json plus mlp_baseline.json under
experiments/.
"""

import json
import math
from pathlib import Path

EXP = Path("experiments")
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

# Load every result
runs = {}
mlp_path = EXP / "mlp_baseline.json"
if mlp_path.exists():
    runs["MLP (baseline)"] = json.loads(mlp_path.read_text())

# Saturation gets special handling
sat = EXP / "saturation_h16.json"
if sat.exists():
    s = json.loads(sat.read_text())
    # Saturation stores per-epoch test_f1 in the history; pick the val-selected epoch
    h = s["history"]
    best_test = max(h, key=lambda r: r["test_f1"])
    val_sel_epoch = s["best_val_epoch"]
    val_sel = next(r for r in h if r["epoch"] == val_sel_epoch)
    runs["FastGRNN H=16 e=120 (val-sel)"] = {
        "n_params": s["config"]["n_params"],
        "test_accuracy": val_sel["test_acc"],
        "test_macro_f1": val_sel["test_f1"],
        "test_loss": 0.0,
        "per_class_f1": None,  # Saturation does not record per-class F1
    }

# Other FastGRNN experiments
for p in sorted(EXP.glob("fastgrnn_*.json")):
    if "saturation" in p.name:
        continue
    data = json.loads(p.read_text())
    label = data.get("model", "FastGRNN") + f" H={data['hidden_size']}"
    if data.get("r_w") is not None:
        label += f" r_w={data['r_w']} r_u={data['r_u']}"
    epochs = data.get("epochs_run", data.get("epochs", "?"))
    label += f" e={epochs}"
    runs[label] = data

# --- 1) Headline metrics ---
print("=" * 110)
print(" HEADLINE METRICS")
print("=" * 110)
print(f"{'Model':<45} {'Params':>8} {'Acc':>8} {'Macro-F1':>10} {'Loss':>8}")
print("-" * 110)
for name, r in runs.items():
    loss = r.get("test_loss", 0)
    print(f"{name:<45} {r['n_params']:>8} {r['test_accuracy']:>8.4f} {r['test_macro_f1']:>10.4f} {loss:>8.4f}")

# --- 2) Per-class F1 ---
print("\n" + "=" * 110)
print(" PER-CLASS F1")
print("=" * 110)
with_pc = {n: r for n, r in runs.items() if r.get("per_class_f1") is not None}
print(f"{'Class':<13}", end="")
for name in with_pc:
    print(f" {name[:20]:>20}", end="")
print()
print("-" * (13 + 21 * len(with_pc)))
for cls in CLASS_NAMES:
    print(f"{cls:<13}", end="")
    for name in with_pc:
        print(f" {with_pc[name]['per_class_f1'].get(cls, 0):>20.4f}", end="")
    print()

# --- 3) Best per metric ---
print("\n" + "=" * 110)
print(" WINNERS")
print("=" * 110)
best_acc = max(runs.items(), key=lambda kv: kv[1]["test_accuracy"])
best_f1  = max(runs.items(), key=lambda kv: kv[1]["test_macro_f1"])
print(f"Highest accuracy : {best_acc[0]}  ({best_acc[1]['test_accuracy']:.4f})")
print(f"Highest macro-F1 : {best_f1[0]}  ({best_f1[1]['test_macro_f1']:.4f})")

# --- 4) Parameter efficiency ---
print(f"\nParameter efficiency (Macro-F1 / log10(params)):")
ranked = sorted(runs.items(), key=lambda kv: kv[1]["test_macro_f1"] / math.log10(kv[1]["n_params"]),
                reverse=True)
for name, r in ranked:
    eff = r["test_macro_f1"] / math.log10(r["n_params"])
    print(f"  {name:<45}  {eff:.4f}  (F1={r['test_macro_f1']:.4f}, params={r['n_params']})")

# --- 5) Per-class winners ---
print(f"\nPer-class winners (only experiments with per-class F1):")
for cls in CLASS_NAMES:
    best = max(with_pc.items(), key=lambda kv: kv[1]["per_class_f1"].get(cls, 0))
    print(f"  {cls:<13} -> {best[0]:<40} {best[1]['per_class_f1'][cls]:.4f}")
