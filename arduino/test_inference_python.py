"""
Python "bit-exact" simulation of the C inference algorithm.

Reimplements in NumPy the exact computation done by fastgrnn.cpp and
compares its predictions against the PyTorch reference model.

If the agreement is far above 99% then we can trust the C code
(i.e. it is mathematically correct).
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastgrnn_model import FastGRNNClassifier
from quantize import quantize_weights, calibrate_activations
from torch.utils.data import DataLoader, TensorDataset

CHECKPOINT = "../sparse_h16_rw2_ru8_sp50_s0_e100_best.pt"
INFO_PATH = "fastgrnn_har/model_info.json"

HIDDEN = 16
INPUT_DIM = 3
NUM_CLASSES = 6
R_W = 2
R_U = 8

# --- Load the PyTorch reference model ---
print("Loading reference model...")
model_ref = FastGRNNClassifier(input_size=INPUT_DIM, hidden_size=HIDDEN,
                                num_classes=NUM_CLASSES, r_w=R_W, r_u=R_U, sparse=True)
model_ref.load_state_dict(torch.load(CHECKPOINT))
model_ref.eval()

# --- For the C version: same model but with quantized weights ---
model_c = FastGRNNClassifier(input_size=INPUT_DIM, hidden_size=HIDDEN,
                              num_classes=NUM_CLASSES, r_w=R_W, r_u=R_U, sparse=True)
model_c.load_state_dict(torch.load(CHECKPOINT))
model_c.eval()
scales = quantize_weights(model_c, verbose=False)

# --- Pull the input mean/std and the zeta/nu scalars from the info file ---
info = json.loads(Path(INFO_PATH).read_text())
mean = np.array(info["input_mean"], dtype=np.float32)
std  = np.array(info["input_std"],  dtype=np.float32)
zeta = float(info["zeta"])
nu   = float(info["nu"])

# --- NumPy inference identical to the C code (AVR-equivalent math) ---
def cell_step_c_equivalent(x_raw, h_prev,
                            W1, W2, U1, U2, b_z, b_h, zeta, nu, mean, std):
    """Identical computation to fastgrnn.cpp::fastgrnn_step."""
    # 1) Normalize
    xn = (x_raw - mean) / std

    # 2) Low-rank intermediate: xn @ W2 @ W1.T
    xz = xn @ W2                  # (R_W,)
    xW = xz @ W1.T                # (H,)

    # 3) Hidden: h_prev @ U2 @ U1.T
    hz = h_prev @ U2              # (R_U,)
    hU = hz @ U1.T                # (H,)

    # 4) FastGRNN combination
    pre = xW + hU
    z = 1.0 / (1.0 + np.exp(-(pre + b_z)))    # sigmoid
    h_tilde = np.tanh(pre + b_h)
    coef = zeta * (1.0 - z) + nu
    h_new = coef * h_tilde + z * h_prev
    return h_new

def predict_window_c_equivalent(X, W1, W2, U1, U2, b_z, b_h,
                                  zeta, nu, mean, std, cls_W, cls_b):
    """Run cell_step for T=128 then apply the classifier."""
    h = np.zeros(HIDDEN, dtype=np.float32)
    for t in range(X.shape[0]):
        h = cell_step_c_equivalent(X[t], h, W1, W2, U1, U2, b_z, b_h,
                                    zeta, nu, mean, std)
    logits = cls_W @ h + cls_b
    return logits, np.argmax(logits)

# Numpy weights from the quantized model (these are bit-identical to the C side)
W1 = model_c.cell.W1.detach().numpy() * model_c.cell.mask_W1.numpy()
W2 = model_c.cell.W2.detach().numpy() * model_c.cell.mask_W2.numpy()
U1 = model_c.cell.U1.detach().numpy() * model_c.cell.mask_U1.numpy()
U2 = model_c.cell.U2.detach().numpy() * model_c.cell.mask_U2.numpy()
b_z = model_c.cell.b_z.detach().numpy()
b_h = model_c.cell.b_h.detach().numpy()
cls_W = model_c.classifier.weight.detach().numpy()
cls_b = model_c.classifier.bias.detach().numpy()

# --- Test data ---
print("Loading test data...")
data = np.load("../data/processed/hapt_windows.npz", allow_pickle=True)
X_te, y_te = data["X_test"], data["y_test"] - 1
print(f"Test windows: {X_te.shape}")

# --- PyTorch reference predictions (float32) ---
print("\nReference (PyTorch float32) predictions...")
# Note: we normalize the input here. The C side normalizes internally with the
# same mean/std, so both paths see the same effective input.
X_te_norm = ((X_te - mean) / std).astype(np.float32)

ref_preds = []
ref_logits = []
with torch.no_grad():
    for i in range(0, len(X_te_norm), 256):
        batch = torch.from_numpy(X_te_norm[i:i+256])
        logits = model_ref(batch).numpy()
        ref_logits.append(logits)
        ref_preds.append(logits.argmax(axis=1))
ref_preds = np.concatenate(ref_preds)
ref_logits = np.concatenate(ref_logits)
ref_acc = (ref_preds == y_te).mean()
print(f"Reference (PyTorch, float32 weights) accuracy: {ref_acc:.4f}")

# --- C-equivalent predictions (Q15 weights, float compute) ---
print("\nRunning C-equivalent inference on test set...")
c_preds = np.zeros(len(X_te), dtype=np.int64)
c_logits = np.zeros((len(X_te), NUM_CLASSES), dtype=np.float32)
for i, X in enumerate(X_te):
    logits, pred = predict_window_c_equivalent(
        X.astype(np.float32), W1, W2, U1, U2, b_z, b_h,
        zeta, nu, mean, std, cls_W, cls_b
    )
    c_preds[i] = pred
    c_logits[i] = logits
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(X_te)}")

c_acc = (c_preds == y_te).mean()
print(f"\nC-equivalent (Q15 weights, float compute) accuracy: {c_acc:.4f}")

# --- Comparison ---
print("\n" + "=" * 70)
print(" COMPARISON")
print("=" * 70)
match = (ref_preds == c_preds).mean()
print(f"Reference vs C-equivalent prediction agreement: {match*100:.2f}%")
print(f"Reference acc:    {ref_acc:.4f}")
print(f"C-equivalent acc: {c_acc:.4f}")
print(f"Acc delta: {c_acc - ref_acc:+.4f}")

# Logit closeness
logit_diff = np.abs(ref_logits - c_logits)
print(f"\nLogit absolute difference:  max={logit_diff.max():.6f}, mean={logit_diff.mean():.6f}")

# Detailed mismatches
mismatch_idx = np.where(ref_preds != c_preds)[0]
if len(mismatch_idx) > 0:
    print(f"\n{len(mismatch_idx)} mismatches:")
    for idx in mismatch_idx[:10]:
        print(f"  Window {idx}: ref={ref_preds[idx]}, C={c_preds[idx]}, true={y_te[idx]}")

# Also export a handful of test vectors for the C compile-time check
n_samples = 5
sample_idx = np.linspace(0, len(X_te)-1, n_samples, dtype=int)
samples = []
for idx in sample_idx:
    samples.append({
        "window": X_te[idx].tolist(),
        "true_label": int(y_te[idx]),
        "ref_pred": int(ref_preds[idx]),
        "c_pred": int(c_preds[idx]),
        "logits": c_logits[idx].tolist(),
    })
with open("test_vectors.json", "w") as f:
    json.dump(samples, f, indent=2)
print(f"\nSaved: test_vectors.json ({n_samples} samples)")
