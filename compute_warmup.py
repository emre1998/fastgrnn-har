"""
Warm-up distribution: for 100 random test windows, find the first step t*
at which the per-step prediction matches the final-window prediction and
stays matching for all remaining steps.

Uses the deployed seed-0 model with Q15-rounded weights (same as compute_agreement.py).

Outputs:
  experiments/warmup_distribution.json  — per-window t* values + stats
"""
import json
import numpy as np
import torch
from pathlib import Path
from fastgrnn_model import FastGRNNClassifier
from fastgrnn_numpy import run_sequence

SEED = 0
NUM_CLASSES = 6
Q15_MAX = 32767.0
N_WINDOWS = 100
RNG_SEED = 42

data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_te, y_te = data["X_test"], data["y_test"] - 1
X_tr = data["X_train"]
mean = X_tr.mean(axis=(0, 1))
std  = X_tr.std(axis=(0, 1)) + 1e-8
X_te_n = ((X_te - mean) / std).astype(np.float32)

rng = np.random.default_rng(RNG_SEED)
idx = rng.choice(len(X_te_n), size=N_WINDOWS, replace=False)

def q15_round(arr):
    scale = np.abs(arr).max() / Q15_MAX
    if scale == 0:
        return arr
    return np.round(arr / scale) * scale

ckpt = f"sparse_h16_rw2_ru8_sp50_s{SEED}_e100_best.pt"
m = FastGRNNClassifier(input_size=3, hidden_size=16, num_classes=NUM_CLASSES,
                       r_w=2, r_u=8, sparse=True)
m.load_state_dict(torch.load(ckpt))
m.eval()

sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
W1 = sd["cell.W1"] * sd["cell.mask_W1"]
W2 = sd["cell.W2"] * sd["cell.mask_W2"]
U1 = sd["cell.U1"] * sd["cell.mask_U1"]
U2 = sd["cell.U2"] * sd["cell.mask_U2"]
W = q15_round(W1 @ W2.T)
U = q15_round(U1 @ U2.T)
b_z = q15_round(sd["cell.b_z"])
b_h = q15_round(sd["cell.b_h"])
zeta = float(torch.sigmoid(torch.tensor(sd["cell.zeta_raw"])))
nu   = float(torch.sigmoid(torch.tensor(sd["cell.nu_raw"])))
W_cls = q15_round(sd["classifier.weight"])
b_cls = q15_round(sd["classifier.bias"])

T = X_te_n.shape[1]
H = W.shape[0]

stabilization_steps = []

for i in idx:
    window = X_te_n[i]          # (T, D)
    h0 = np.zeros(H, dtype=np.float32)

    # Run full sequence, collect all hidden states
    _, all_h = run_sequence(window, h0, W, U, b_z, b_h, zeta, nu, return_all=True)

    # Per-step predictions
    step_preds = np.array([int(np.argmax(W_cls @ all_h[t] + b_cls))
                           for t in range(T)])

    final_pred = step_preds[-1]

    # Find first t* where step_preds[t*:] are all == final_pred
    t_star = T  # default: never stabilized
    for t in range(T - 1, -1, -1):
        if step_preds[t] != final_pred:
            t_star = t + 1
            break
    else:
        t_star = 0  # all steps already match final pred

    stabilization_steps.append(int(t_star))

arr = np.array(stabilization_steps)
median_t = float(np.median(arr))
q1 = float(np.percentile(arr, 25))
q3 = float(np.percentile(arr, 75))
iqr = q3 - q1
median_s  = median_t / 50.0
iqr_s     = iqr / 50.0

print(f"N windows : {N_WINDOWS}")
print(f"Stabilization step (samples @ 50 Hz):")
print(f"  median : {median_t:.1f}  ({median_s:.2f} s)")
print(f"  IQR    : {q1:.1f} - {q3:.1f}  (IQR={iqr:.1f}, {iqr_s:.2f} s)")
print(f"  min/max: {arr.min()} / {arr.max()}")

result = {
    "n_windows": N_WINDOWS,
    "rng_seed": RNG_SEED,
    "model_seed": SEED,
    "stabilization_steps": stabilization_steps,
    "median_samples": median_t,
    "q1_samples": q1,
    "q3_samples": q3,
    "iqr_samples": iqr,
    "median_seconds": median_s,
    "iqr_seconds": iqr_s,
    "min_samples": int(arr.min()),
    "max_samples": int(arr.max()),
}
Path("experiments/warmup_distribution.json").write_text(json.dumps(result, indent=2))
print("\nSaved: experiments/warmup_distribution.json")
