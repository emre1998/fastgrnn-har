"""
Post-Training Quantization (PTQ) evaluation.

Take the sparse sp50_s0 model -> Q15 weights -> compare accuracy and footprint.

Usage:  python ptq_eval.py
"""

import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from pathlib import Path
from fastgrnn_model import FastGRNNClassifier
from quantize import quantize_weights, model_size_bytes

CHECKPOINT = "sparse_h16_rw2_ru8_sp50_s0_e100_best.pt"
NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

torch.manual_seed(0)
np.random.seed(0)

# --- Data (same pipeline) ---
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

def evaluate(model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total += len(y)
            all_pred.append(pred.numpy())
            all_true.append(y.numpy())
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    return {
        "loss": total_loss/total,
        "acc": total_correct/total,
        "f1": f1_score(y_true, y_pred, average="macro"),
        "per_class_f1": dict(zip(CLASS_NAMES, f1_score(y_true, y_pred, average=None).tolist())),
        "y_true": y_true, "y_pred": y_pred,
    }

# --- Model: sparse low-rank ---
print(f"Loading: {CHECKPOINT}\n")
model = FastGRNNClassifier(
    input_size=3, hidden_size=16, num_classes=NUM_CLASSES,
    r_w=2, r_u=8, sparse=True,
)
model.load_state_dict(torch.load(CHECKPOINT))

# --- 1) Float32 baseline ---
print("=" * 60)
print(" 1) Float32 BASELINE")
print("=" * 60)
size_f32 = model_size_bytes(model, dtype_bits=32)
print(f"  Nonzero params: {size_f32['nonzero_params']}")
print(f"  Size (float32): {size_f32['sparse_size_bytes']} bytes = {size_f32['sparse_size_bytes']/1024:.2f} KB")
result_f32 = evaluate(model)
print(f"  Test acc: {result_f32['acc']:.4f}")
print(f"  Test F1 : {result_f32['f1']:.4f}")

# --- 2) Q15 PTQ ---
print("\n" + "=" * 60)
print(" 2) Q15 POST-TRAINING QUANTIZATION")
print("=" * 60)
model_q = copy.deepcopy(model)
print("Quantized tensors and their scales:")
scales = quantize_weights(model_q, verbose=True)

print()
size_q15 = model_size_bytes(model_q, dtype_bits=16)
print(f"  Size (Q15): {size_q15['sparse_size_bytes']} bytes = {size_q15['sparse_size_bytes']/1024:.2f} KB")
print(f"  Saving: {size_f32['sparse_size_bytes']} -> {size_q15['sparse_size_bytes']} bytes "
      f"({100*(1-size_q15['sparse_size_bytes']/size_f32['sparse_size_bytes']):.0f}% reduction)")
result_q15 = evaluate(model_q)
print(f"  Test acc: {result_q15['acc']:.4f}   (delta: {result_q15['acc']-result_f32['acc']:+.4f})")
print(f"  Test F1 : {result_q15['f1']:.4f}   (delta: {result_q15['f1']-result_f32['f1']:+.4f})")

# --- 3) Per-class comparison ---
print("\n" + "=" * 60)
print(" 3) PER-CLASS F1 (float vs Q15)")
print("=" * 60)
print(f"{'Class':<14} {'float32':>10} {'Q15':>10} {'delta':>10}")
for cls in CLASS_NAMES:
    f32 = result_f32["per_class_f1"][cls]
    q15 = result_q15["per_class_f1"][cls]
    print(f"{cls:<14} {f32:>10.4f} {q15:>10.4f} {q15-f32:>+10.4f}")

# --- 4) Confusion matrix (Q15) ---
print("\n=== Q15 confusion matrix ===")
cm = confusion_matrix(result_q15["y_true"], result_q15["y_pred"])
header = "       " + " ".join(f"{n[:5]:>6}" for n in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    row = " ".join(f"{cm[i, j]:>6d}" for j in range(NUM_CLASSES))
    print(f"{name[:6]:6s} {row}")

# --- 5) Hardware budget comparison ---
print("\n" + "=" * 60)
print(" 4) HARDWARE BUDGET")
print("=" * 60)
print(f"  Arduino Uno (ATmega328P):   32 KB Flash, 2 KB SRAM")
print(f"  MSP430G2553:                16 KB Flash, 512 B SRAM")
print(f"  Our model (Q15, sparse):    {size_q15['sparse_size_bytes']} bytes = {size_q15['sparse_size_kb']:.2f} KB")
print(f"  Arduino usage: {100*size_q15['sparse_size_bytes']/(32*1024):.2f}% of Flash")
print(f"  MSP430 usage:  {100*size_q15['sparse_size_bytes']/(16*1024):.2f}% of Flash")

# --- 6) Save ---
out = {
    "checkpoint": CHECKPOINT,
    "float32": {"acc": result_f32["acc"], "f1": result_f32["f1"],
                "per_class_f1": result_f32["per_class_f1"],
                "size_bytes": size_f32["sparse_size_bytes"]},
    "q15_ptq": {"acc": result_q15["acc"], "f1": result_q15["f1"],
                "per_class_f1": result_q15["per_class_f1"],
                "size_bytes": size_q15["sparse_size_bytes"]},
    "delta_acc": result_q15["acc"] - result_f32["acc"],
    "delta_f1":  result_q15["f1"]  - result_f32["f1"],
    "scales": scales,
}
Path("experiments").mkdir(exist_ok=True)
with open("experiments/ptq_q15.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: experiments/ptq_q15.json")
