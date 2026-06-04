"""
Export a sparse low-rank PyTorch checkpoint as an Arduino-ready C header.

Output: arduino/fastgrnn_har/model_weights.h
  - All weights as int16_t PROGMEM arrays + per-tensor float scales
  - Activation scales from calibration (h_t ≈ 75/32767)
  - Input normalization parameters (mean, std)
  - Sparse masks (zeros are stored directly as 0)
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastgrnn_model import FastGRNNClassifier
from quantize import quantize_weights, calibrate_activations, model_size_bytes
from torch.utils.data import DataLoader, TensorDataset

CHECKPOINT = "../sparse_h16_rw2_ru8_sp50_s0_e100_best.pt"
# Write the header into both the Arduino and the MSP folders (portable header)
OUT_HEADERS = [
    "fastgrnn_har/model_weights.h",
    "../msp/fastgrnn_har_msp/model_weights.h",
]
OUT_HEADER = OUT_HEADERS[0]  # info JSON only on the Arduino side

HIDDEN = 16
INPUT_DIM = 3
NUM_CLASSES = 6
WINDOW_T = 128
R_W = 2
R_U = 8

# --- Load model ---
print(f"Loading checkpoint: {CHECKPOINT}")
model = FastGRNNClassifier(
    input_size=INPUT_DIM, hidden_size=HIDDEN, num_classes=NUM_CLASSES,
    r_w=R_W, r_u=R_U, sparse=True,
)
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

# --- Data (for normalization and calibration) ---
data = np.load("../data/processed/hapt_windows.npz", allow_pickle=True)
X_tr = data["X_train"]
mean = X_tr.mean(axis=(0, 1))
std  = X_tr.std(axis=(0, 1)) + 1e-8
print(f"Input mean: {mean}")
print(f"Input std:  {std}")

# --- Quantize weights to Q15 ---
print("\nQuantizing weights to Q15...")
scales = quantize_weights(model, verbose=True)

# --- Activation calibration ---
print("\nCalibrating activations on train data...")
X_tr_n = ((X_tr - mean) / std).astype(np.float32)
calib_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr_n[:1024]), torch.zeros(1024).long()),
    batch_size=256
)
act_stats = calibrate_activations(model, calib_loader, n_batches=5)
print(f"Activation max (calibration): {act_stats}")

# 10% headroom
H_MAX  = act_stats["h_t"]     * 1.1
Z_MAX  = act_stats["z"]       * 1.1
HT_MAX = act_stats["h_tilde"] * 1.1

# --- Convert tensors to C format ---
def tensor_to_q15_array(W: torch.Tensor, scale: float):
    """Float tensor -> int16 numpy array (Q15 quantized)."""
    q = (W.detach().numpy() / scale).round().clip(-32768, 32767).astype(np.int16)
    return q

def format_array_1d(arr, indent=2):
    lines = []
    indent_str = " " * indent
    line = indent_str
    for i, v in enumerate(arr):
        if len(line) > 90:
            lines.append(line)
            line = indent_str
        line += f"{v:6d}, "
    if line.strip():
        lines.append(line.rstrip(", "))
    return "\n".join(lines)

def format_array_2d(arr, indent=2):
    lines = []
    indent_str = " " * indent
    for row in arr:
        line = indent_str + "{ " + ", ".join(f"{v:6d}" for v in row) + " },"
        lines.append(line)
    return "\n".join(lines)

cell = model.cell

# Quantize tensors (already done in-place, scales stored in `scales` dict)
W1_q = tensor_to_q15_array(cell.W1, scales["cell.W1"])
W2_q = tensor_to_q15_array(cell.W2, scales["cell.W2"])
U1_q = tensor_to_q15_array(cell.U1, scales["cell.U1"])
U2_q = tensor_to_q15_array(cell.U2, scales["cell.U2"])
b_z_q = tensor_to_q15_array(cell.b_z, scales["cell.b_z"])
b_h_q = tensor_to_q15_array(cell.b_h, scales["cell.b_h"])

# zeta and nu: raw parameters, in (0,1) after sigmoid — kept as float scalars
zeta = torch.sigmoid(cell.zeta_raw).item()
nu   = torch.sigmoid(cell.nu_raw).item()

cls_W_q = tensor_to_q15_array(model.classifier.weight, scales["classifier.weight"])
cls_b_q = tensor_to_q15_array(model.classifier.bias, scales["classifier.bias"])

# Count: how many nonzero entries
nonzero = sum((arr != 0).sum() for arr in [W1_q, W2_q, U1_q, U2_q, b_z_q, b_h_q, cls_W_q, cls_b_q])
total = sum(arr.size for arr in [W1_q, W2_q, U1_q, U2_q, b_z_q, b_h_q, cls_W_q, cls_b_q]) + 2  # +2 for zeta/nu
print(f"\nTotal params: {total}, nonzero: {nonzero}, sparsity: {100*(1-nonzero/total):.1f}%")

# --- Build the header file ---
header = f"""/*
 * model_weights.h - FastGRNN HAR model, auto-generated
 *
 * Pipeline: low-rank (r_w={R_W}, r_u={R_U}) + sparsity 50% + Q15
 * Source checkpoint: {Path(CHECKPOINT).name}
 * Total params (dense count): {total}
 * Nonzero params: {nonzero}
 * Flash footprint (Q15): {nonzero * 2} bytes
 *
 * PORTABLE: targets both AVR (Arduino) and MSP430 (Energia).
 *   - AVR: uses the PROGMEM macro
 *   - MSP430 and others: `const` already lands in Flash
 */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

#ifdef __AVR__
  #include <avr/pgmspace.h>
#else
  #ifndef PROGMEM
    #define PROGMEM
  #endif
#endif

// --- Architecture constants ---
#define HIDDEN_SIZE  {HIDDEN}
#define INPUT_DIM    {INPUT_DIM}
#define NUM_CLASSES  {NUM_CLASSES}
#define WINDOW_T     {WINDOW_T}
#define R_W          {R_W}
#define R_U          {R_U}

// --- Fixed-point scales ---
const float W1_SCALE       = {scales['cell.W1']:.8e}f;
const float W2_SCALE       = {scales['cell.W2']:.8e}f;
const float U1_SCALE       = {scales['cell.U1']:.8e}f;
const float U2_SCALE       = {scales['cell.U2']:.8e}f;
const float BZ_SCALE       = {scales['cell.b_z']:.8e}f;
const float BH_SCALE       = {scales['cell.b_h']:.8e}f;
const float CLS_W_SCALE    = {scales['classifier.weight']:.8e}f;
const float CLS_B_SCALE    = {scales['classifier.bias']:.8e}f;

// --- Activation scales (from calibration, 10% headroom) ---
const float Z_MAX_ABS       = {Z_MAX:.4f}f;   // sigmoid output range
const float H_TILDE_MAX_ABS = {HT_MAX:.4f}f;  // tanh output range
const float H_T_MAX_ABS     = {H_MAX:.4f}f;   // h_t actual range (wide!)

// --- Scalar parameters (post-sigmoid) ---
const float ZETA = {zeta:.6f}f;
const float NU   = {nu:.6f}f;

// --- Input normalization ---
const float INPUT_MEAN[INPUT_DIM] = {{ {mean[0]:.6f}f, {mean[1]:.6f}f, {mean[2]:.6f}f }};
const float INPUT_STD[INPUT_DIM]  = {{ {std[0]:.6f}f,  {std[1]:.6f}f,  {std[2]:.6f}f }};

// --- Class names (for UART output) ---
const char* const CLASS_NAMES[NUM_CLASSES] = {{
  "WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"
}};

// ============================================================================
// WEIGHTS - int16_t, PROGMEM (Flash)
// ============================================================================

// W1: (HIDDEN_SIZE, R_W) = ({HIDDEN}, {R_W})
const int16_t W1[HIDDEN_SIZE][R_W] PROGMEM = {{
{format_array_2d(W1_q)}
}};

// W2: (INPUT_DIM, R_W) = ({INPUT_DIM}, {R_W})
const int16_t W2[INPUT_DIM][R_W] PROGMEM = {{
{format_array_2d(W2_q)}
}};

// U1: (HIDDEN_SIZE, R_U) = ({HIDDEN}, {R_U})
const int16_t U1[HIDDEN_SIZE][R_U] PROGMEM = {{
{format_array_2d(U1_q)}
}};

// U2: (HIDDEN_SIZE, R_U) = ({HIDDEN}, {R_U})
const int16_t U2[HIDDEN_SIZE][R_U] PROGMEM = {{
{format_array_2d(U2_q)}
}};

// b_z, b_h: (HIDDEN_SIZE,)
const int16_t B_Z[HIDDEN_SIZE] PROGMEM = {{
{format_array_1d(b_z_q)}
}};

const int16_t B_H[HIDDEN_SIZE] PROGMEM = {{
{format_array_1d(b_h_q)}
}};

// Classifier W: (NUM_CLASSES, HIDDEN_SIZE)
const int16_t CLS_W[NUM_CLASSES][HIDDEN_SIZE] PROGMEM = {{
{format_array_2d(cls_W_q)}
}};

// Classifier bias: (NUM_CLASSES,)
const int16_t CLS_B[NUM_CLASSES] PROGMEM = {{
{format_array_1d(cls_b_q)}
}};

#endif // MODEL_WEIGHTS_H
"""

for out_path in OUT_HEADERS:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header)
    sz = Path(out_path).stat().st_size
    print(f"  Saved: {out_path} ({sz} bytes)")

# Summary file (helpful during development)
info = {
    "checkpoint": str(CHECKPOINT),
    "scales": {k: float(v) for k, v in scales.items()},
    "activation_stats": act_stats,
    "headroom": 1.1,
    "z_max_abs": Z_MAX,
    "h_tilde_max_abs": HT_MAX,
    "h_t_max_abs": H_MAX,
    "zeta": zeta, "nu": nu,
    "input_mean": mean.tolist(), "input_std": std.tolist(),
    "total_params": total, "nonzero_params": int(nonzero),
}
with open("fastgrnn_har/model_info.json", "w") as f:
    json.dump(info, f, indent=2)
print(f"Saved: fastgrnn_har/model_info.json")
