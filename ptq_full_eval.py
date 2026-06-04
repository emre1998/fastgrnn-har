"""
Full PTQ evaluation: multi-seed + activation Q15.

For every sp50_s{0..4} model we run three scenarios:
  1) float32 baseline
  2) Q15 weights only
  3) Q15 weights + Q15 activations (the real deployment simulation)
"""

import json
import copy
import statistics
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from pathlib import Path
from fastgrnn_model import FastGRNNClassifier
from quantize import (quantize_weights, wrap_cell_for_activation_quantization,
                      calibrate_activations, wrap_cell_with_calibrated_quantization,
                      model_size_bytes)

NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
SEEDS = [0, 1, 2, 3, 4]

# --- Data ---
data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_te, y_te = data["X_test"], data["y_test"]
X_tr = data["X_train"]
y_te = y_te - 1
mean = X_tr.mean(axis=(0, 1))
std  = X_tr.std(axis=(0, 1)) + 1e-8
X_te_n = ((X_te - mean) / std).astype(np.float32)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_te_n), torch.from_numpy(y_te).long()),
    batch_size=256
)
# A few training batches for calibration
X_tr_n = ((X_tr - mean) / std).astype(np.float32)
y_tr_dummy = np.zeros(len(X_tr_n), dtype=np.int64)
calib_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr_dummy).long()),
    batch_size=256, shuffle=True,
)
criterion = nn.CrossEntropyLoss()

def evaluate(model):
    model.eval()
    all_pred, all_true = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
            all_pred.append(pred.numpy()); all_true.append(y.numpy())
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    return correct/total, f1_score(y_true, y_pred, average="macro"), f1_score(y_true, y_pred, average=None).tolist()

def build_model_from(ckpt):
    m = FastGRNNClassifier(input_size=3, hidden_size=16, num_classes=NUM_CLASSES,
                           r_w=2, r_u=8, sparse=True)
    m.load_state_dict(torch.load(ckpt))
    return m

# --- Multi-seed loop ---
results = {"float32": [], "q15_weights": [], "q15_weights_acts": []}
per_class = {"float32": [], "q15_weights": [], "q15_weights_acts": []}

print(f"Multi-seed PTQ - {len(SEEDS)} seeds\n")
print(f"{'seed':>4} {'mode':<20} {'acc':>8} {'f1':>8}  delta_f1")
print("-" * 60)

for seed in SEEDS:
    ckpt = f"sparse_h16_rw2_ru8_sp50_s{seed}_e100_best.pt"
    if not Path(ckpt).exists():
        print(f"  s{seed}: {ckpt} not found, skipping")
        continue

    # 1) float32 baseline
    m = build_model_from(ckpt)
    acc, f1, pc = evaluate(m)
    results["float32"].append((acc, f1))
    per_class["float32"].append(pc)
    f1_base = f1
    print(f"{seed:>4} {'float32':<20} {acc:>8.4f} {f1:>8.4f}  baseline")

    # 2) Q15 weights only
    m_q = copy.deepcopy(m)
    quantize_weights(m_q, verbose=False)
    acc, f1, pc = evaluate(m_q)
    results["q15_weights"].append((acc, f1))
    per_class["q15_weights"].append(pc)
    print(f"{seed:>4} {'q15_weights':<20} {acc:>8.4f} {f1:>8.4f}  {f1-f1_base:+.4f}")

    # 3) Q15 weights + activations (WITH CALIBRATION)
    m_qa = copy.deepcopy(m)
    quantize_weights(m_qa, verbose=False)
    stats = calibrate_activations(m_qa, calib_loader, n_batches=5)
    wrap_cell_with_calibrated_quantization(m_qa.cell, stats, headroom=1.1)
    acc, f1, pc = evaluate(m_qa)
    results["q15_weights_acts"].append((acc, f1))
    per_class["q15_weights_acts"].append(pc)
    print(f"{seed:>4} {'q15+acts (calib)':<20} {acc:>8.4f} {f1:>8.4f}  {f1-f1_base:+.4f}  "
          f"[z={stats['z']:.2f}, h_t={stats['h_t']:.2f}]")
    print()

# --- Summary ---
print("=" * 70)
print(" MULTI-SEED SUMMARY (mean +/- std)")
print("=" * 70)
print(f"{'mode':<22} {'acc_mean':>10} {'acc_std':>9} {'f1_mean':>10} {'f1_std':>9}")
print("-" * 70)
for mode, vals in results.items():
    if not vals:
        continue
    accs = [v[0] for v in vals]
    f1s  = [v[1] for v in vals]
    am, asd = statistics.mean(accs), statistics.stdev(accs) if len(accs)>1 else 0
    fm, fsd = statistics.mean(f1s),  statistics.stdev(f1s)  if len(f1s)>1 else 0
    print(f"{mode:<22} {am:>10.4f} {asd:>9.4f} {fm:>10.4f} {fsd:>9.4f}")

# Delta F1 summary
print("\nDelta F1 (vs float32):")
base = [v[1] for v in results["float32"]]
for mode in ("q15_weights", "q15_weights_acts"):
    vals = [v[1] for v in results[mode]]
    deltas = [v - b for v, b in zip(vals, base)]
    if deltas:
        dm = statistics.mean(deltas); ds = statistics.stdev(deltas) if len(deltas)>1 else 0
        print(f"  {mode:<22}  delta mean = {dm:+.4f}, std = {ds:.4f}")

# Per-class summary (final mode only)
print("\n" + "=" * 70)
print(" PER-CLASS F1 (q15_weights+acts, mean +/- std)")
print("=" * 70)
pcs = per_class["q15_weights_acts"]
if pcs:
    for i, cls in enumerate(CLASS_NAMES):
        vals = [pc[i] for pc in pcs]
        m = statistics.mean(vals); s = statistics.stdev(vals) if len(vals)>1 else 0
        print(f"  {cls:<12} {m:.4f} +/- {s:.4f}")

# Size table (fixed, independent of seed)
print("\n" + "=" * 70)
print(" DEPLOYMENT BUDGET (Q15)")
print("=" * 70)
m = build_model_from(f"sparse_h16_rw2_ru8_sp50_s0_e100_best.pt")
sz = model_size_bytes(m, dtype_bits=16)
print(f"  Sparse model (Q15): {sz['sparse_size_bytes']} bytes = {sz['sparse_size_kb']:.2f} KB")
print(f"  Arduino usage:      {100*sz['sparse_size_bytes']/(32*1024):.2f}% of Flash")
print(f"  MSP430 usage:       {100*sz['sparse_size_bytes']/(16*1024):.2f}% of Flash")

# Save
out = {
    "seeds": SEEDS,
    "modes": {mode: {"accs": [v[0] for v in vals], "f1s": [v[1] for v in vals]}
              for mode, vals in results.items() if vals},
    "per_class": {mode: pcs for mode, pcs in per_class.items() if pcs},
    "size_bytes": sz["sparse_size_bytes"],
}
with open("experiments/ptq_full_multiseed.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: experiments/ptq_full_multiseed.json")
