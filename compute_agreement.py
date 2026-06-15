"""
Compute PyTorch-FP32 vs NumPy-Q15 prediction agreement for 5 seeds.
Uses the same 3,399 test windows used throughout the paper.
"""
import json
import numpy as np
import torch
from pathlib import Path
from fastgrnn_model import FastGRNNClassifier
from fastgrnn_numpy import run_sequence

SEEDS = [0, 1, 2, 3, 4]
NUM_CLASSES = 6
Q15_MAX = 32767.0

data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_te, y_te = data["X_test"], data["y_test"] - 1
X_tr = data["X_train"]
mean = X_tr.mean(axis=(0, 1))
std  = X_tr.std(axis=(0, 1)) + 1e-8
X_te_n = ((X_te - mean) / std).astype(np.float32)   # (N, T, D)

def q15_round(arr):
    scale = np.abs(arr).max() / Q15_MAX
    if scale == 0:
        return arr, scale
    return np.round(arr / scale) * scale, scale

def numpy_q15_predict(model, X):
    """Run NumPy C-equivalent inference with Q15 weights."""
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}

    # Low-rank + sparse mask → effective W and U
    # W1:(H,rw), W2:(D,rw), mask applied during IHT training
    W1 = sd["cell.W1"] * sd["cell.mask_W1"]
    W2 = sd["cell.W2"] * sd["cell.mask_W2"]
    U1 = sd["cell.U1"] * sd["cell.mask_U1"]
    U2 = sd["cell.U2"] * sd["cell.mask_U2"]
    W = W1 @ W2.T                                # (H, D)
    U = U1 @ U2.T                                # (H, H)
    b_z = sd["cell.b_z"]
    b_h = sd["cell.b_h"]
    zeta = float(torch.sigmoid(torch.tensor(sd["cell.zeta_raw"])))
    nu   = float(torch.sigmoid(torch.tensor(sd["cell.nu_raw"])))
    W_cls = sd["classifier.weight"]              # (C, H)
    b_cls = sd["classifier.bias"]               # (C,)

    # Quantize to Q15 grid
    W,   _ = q15_round(W)
    U,   _ = q15_round(U)
    b_z, _ = q15_round(b_z)
    b_h, _ = q15_round(b_h)
    W_cls, _ = q15_round(W_cls)
    b_cls, _ = q15_round(b_cls)

    preds = []
    for window in X:                              # (T, D)
        h = np.zeros(W.shape[0], dtype=np.float32)
        h = run_sequence(window, h, W, U, b_z, b_h, zeta, nu)
        logits = W_cls @ h + b_cls
        preds.append(int(np.argmax(logits)))
    return np.array(preds)

def pytorch_predict(model, X):
    model.eval()
    x_t = torch.from_numpy(X)
    with torch.no_grad():
        logits = model(x_t)
    return logits.argmax(dim=1).numpy()

results = {}
print(f"{'Seed':>4}  {'PT-NP agree':>12}  {'PT-NP %':>8}  {'NP F1':>8}")
print("-" * 45)

for seed in SEEDS:
    ckpt = f"sparse_h16_rw2_ru8_sp50_s{seed}_e100_best.pt"
    m = FastGRNNClassifier(input_size=3, hidden_size=16, num_classes=NUM_CLASSES,
                           r_w=2, r_u=8, sparse=True)
    m.load_state_dict(torch.load(ckpt))

    pt_preds  = pytorch_predict(m, X_te_n)
    np_preds  = numpy_q15_predict(m, X_te_n)

    agree_n   = int((pt_preds == np_preds).sum())
    agree_pct = agree_n / len(y_te) * 100

    from sklearn.metrics import f1_score
    np_f1 = f1_score(y_te, np_preds, average="macro")

    print(f"{seed:>4}  {agree_n:>5}/{len(y_te):<6}  {agree_pct:>7.2f}%  {np_f1:>8.4f}")
    results[seed] = {"agree_n": agree_n, "total": len(y_te),
                     "agree_pct": agree_pct, "np_q15_f1": np_f1}

Path("experiments/agreement_5seed.json").write_text(json.dumps(results, indent=2))
print("\nSaved: experiments/agreement_5seed.json")
