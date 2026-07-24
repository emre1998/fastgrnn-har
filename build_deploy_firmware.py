"""
B1: build + verify GRU/LSTM deployment firmware for the MCU latency/energy study.

Trains the shrink-H deployment model (HAPT, seed 0, 200 epochs), quantizes weights
to per-tensor Q15, emits arduino/{cell}_har/model_weights.h in the layout the C
firmware (gru.cpp / lstm.cpp) expects, then VERIFIES correctness two ways:
  1. NumPy "C-equivalent" (Q15 weights, float compute) vs PyTorch reference.
  2. Emits test_vectors so the compiled C can be checked on host (verify_firmware.py).

Usage:
  python build_deploy_firmware.py --cell gru   --hidden 6
  python build_deploy_firmware.py --cell lstm  --hidden 5
"""
import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

import repro
repro.pin_threads()      # thread count changes results -- see REPRODUCIBILITY.md

parser = argparse.ArgumentParser()
parser.add_argument("--cell", choices=["gru", "lstm"], required=True)
parser.add_argument("--hidden", type=int, required=True)
parser.add_argument("--data", default="data/processed/hapt_windows.npz")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--val_holdout", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

GATES = 3 if args.cell == "gru" else 4
torch.manual_seed(args.seed); np.random.seed(args.seed)

data = np.load(args.data, allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te = data["X_test"], data["y_test"]
y_tr, y_te = y_tr - 1, y_te - 1
NUM_CLASSES = int(max(y_tr.max(), y_te.max())) + 1
WINDOW_T, INPUT_DIM = X_tr.shape[1], X_tr.shape[2]
CLASS_NAMES = ([str(s).split(" ", 1)[-1] for s in data["activity_labels"]]
               if "activity_labels" in data else [f"class{i}" for i in range(NUM_CLASSES)])
CLASS_NAMES = CLASS_NAMES[:NUM_CLASSES]   # activity_labels may list more than the kept classes
uniq = sorted(set(s_tr.tolist())); val_subjects = set(uniq[-args.val_holdout:])
vmask = np.array([s in val_subjects for s in s_tr])
X_trn, y_trn = X_tr[~vmask], y_tr[~vmask]
X_val, y_val = X_tr[vmask], y_tr[vmask]
mean = X_trn.mean(axis=(0, 1)); std = X_trn.std(axis=(0, 1)) + 1e-8
norm = lambda X: ((X - mean) / std).astype(np.float32)


class SmallRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = (nn.GRU if args.cell == "gru" else nn.LSTM)(INPUT_DIM, args.hidden, batch_first=True)
        self.classifier = nn.Linear(args.hidden, NUM_CLASSES)
    def forward(self, X):
        out, _ = self.rnn(X)
        return self.classifier(out[:, -1, :])


def loader(X, y, bs=64, sh=False):
    return DataLoader(TensorDataset(torch.from_numpy(norm(X)), torch.from_numpy(y).long()),
                      batch_size=bs, shuffle=sh)
train_loader, val_loader = loader(X_trn, y_trn, 64, True), loader(X_val, y_val, 256)
crit = nn.CrossEntropyLoss()


@torch.no_grad()
def f1_of(model, X, y):
    model.eval()
    p = model(torch.from_numpy(norm(X))).argmax(1).numpy()
    return f1_score(y, p, average="macro")


# --- train (or load) the deployment checkpoint ---
ckpt = Path(f"deploy_{args.cell}_h{args.hidden}_hapt_s{args.seed}.pt")
model = SmallRNN()
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt)); print(f"Loaded {ckpt}")
else:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(); loss = crit(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        vf1 = f1_of(model, X_val, y_val)
        if vf1 > best: best, best_state = vf1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); torch.save(best_state, ckpt)
    print(f"Trained + saved {ckpt} (val F1 {best:.4f})")
model.eval()

# --- pull weights, PyTorch gate order matches C: GRU r,z,n / LSTM i,f,g,o ---
W_ih = model.rnn.weight_ih_l0.detach().numpy()   # (G*H, D)
W_hh = model.rnn.weight_hh_l0.detach().numpy()   # (G*H, H)
b_ih = model.rnn.bias_ih_l0.detach().numpy()     # (G*H,)
b_hh = model.rnn.bias_hh_l0.detach().numpy()
cls_W = model.classifier.weight.detach().numpy()
cls_b = model.classifier.bias.detach().numpy()


def q15(arr):
    sc = float(np.abs(arr).max()) / 32767.0 or 1.0
    q = np.round(arr / sc).clip(-32768, 32767).astype(np.int16)
    return q, sc


qW_ih, sW_ih = q15(W_ih); qW_hh, sW_hh = q15(W_hh)
qb_ih, sb_ih = q15(b_ih); qb_hh, sb_hh = q15(b_hh)
qC_W, sC_W = q15(cls_W); qC_b, sC_b = q15(cls_b)

# --- NumPy C-equivalent verification (Q15 weights dequantized, float compute) ---
Wih = qW_ih.astype(np.float32) * sW_ih
Whh = qW_hh.astype(np.float32) * sW_hh
Bih = qb_ih.astype(np.float32) * sb_ih
Bhh = qb_hh.astype(np.float32) * sb_hh
CW = qC_W.astype(np.float32) * sC_W
Cb = qC_b.astype(np.float32) * sC_b
H = args.hidden
sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -8, 8)))
tnh = lambda x: np.tanh(np.clip(x, -8, 8))


def c_equiv_window(X):
    h = np.zeros(H, np.float32); c = np.zeros(H, np.float32)
    for t in range(X.shape[0]):
        xn = (X[t] - mean) / std
        gi = Wih @ xn + Bih
        gh = Whh @ h + Bhh
        if args.cell == "gru":
            r = sig(gi[0:H] + gh[0:H]); z = sig(gi[H:2*H] + gh[H:2*H])
            n = tnh(gi[2*H:3*H] + r * gh[2*H:3*H])
            h = (1 - z) * n + z * h
        else:
            pre = gi + gh
            ii = sig(pre[0:H]); ff = sig(pre[H:2*H]); gg = tnh(pre[2*H:3*H]); oo = sig(pre[3*H:4*H])
            c = ff * c + ii * gg; h = oo * tnh(c)
    return CW @ h + Cb


# reference (PyTorch) vs C-equivalent
with torch.no_grad():
    ref_logits = model(torch.from_numpy(norm(X_te))).numpy()
ref_pred = ref_logits.argmax(1)
c_pred = np.zeros(len(X_te), np.int64); c_logits = np.zeros((len(X_te), NUM_CLASSES), np.float32)
for i, X in enumerate(X_te):
    c_logits[i] = c_equiv_window(X.astype(np.float32)); c_pred[i] = c_logits[i].argmax()
agree = (ref_pred == c_pred).mean()
ref_f1 = f1_score(y_te, ref_pred, average="macro")
c_f1 = f1_score(y_te, c_pred, average="macro")
print(f"\n=== {args.cell.upper()} H{H} bit-exact check ===")
print(f"PyTorch F1 {ref_f1:.4f} | C-equivalent(Q15) F1 {c_f1:.4f} | agreement {agree*100:.2f}%")
print(f"logit max abs diff {np.abs(ref_logits - c_logits).max():.5f}")

# --- emit model_weights.h ---
def arr2d(a):
    return "\n".join("  { " + ", ".join(f"{v:6d}" for v in row) + " }," for row in a)
def arr1d(a):
    return "  " + ", ".join(f"{v:6d}" for v in a)

nz = int(sum((x != 0).sum() for x in [qW_ih, qW_hh, qb_ih, qb_hh, qC_W, qC_b]))
outdir = Path(f"arduino/{args.cell}_har"); outdir.mkdir(parents=True, exist_ok=True)
hdr = f"""/* model_weights.h - {args.cell.upper()} HAR, auto-generated (build_deploy_firmware.py)
 * Dense shrink-H deployment model | H={H} | {GATES} gates | Q15 weights
 * Nonzero params: {nz} | Flash (Q15): {nz*2} bytes | F1 {c_f1:.4f}
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

#define HIDDEN_SIZE  {H}
#define INPUT_DIM    {INPUT_DIM}
#define NUM_CLASSES  {NUM_CLASSES}
#define WINDOW_T     {WINDOW_T}
#define N_GATES      {GATES}

const float W_IH_SCALE  = {sW_ih:.8e}f;
const float W_HH_SCALE  = {sW_hh:.8e}f;
const float B_IH_SCALE  = {sb_ih:.8e}f;
const float B_HH_SCALE  = {sb_hh:.8e}f;
const float CLS_W_SCALE = {sC_W:.8e}f;
const float CLS_B_SCALE = {sC_b:.8e}f;

const float INPUT_MEAN[INPUT_DIM] = {{ {", ".join(f"{v:.6f}f" for v in mean)} }};
const float INPUT_STD[INPUT_DIM]  = {{ {", ".join(f"{v:.6f}f" for v in std)} }};
const char* const CLASS_NAMES[NUM_CLASSES] = {{ {", ".join(f'"{n}"' for n in CLASS_NAMES)} }};

const int16_t W_IH[N_GATES*HIDDEN_SIZE][INPUT_DIM] PROGMEM = {{
{arr2d(qW_ih)}
}};
const int16_t W_HH[N_GATES*HIDDEN_SIZE][HIDDEN_SIZE] PROGMEM = {{
{arr2d(qW_hh)}
}};
const int16_t B_IH[N_GATES*HIDDEN_SIZE] PROGMEM = {{
{arr1d(qb_ih)}
}};
const int16_t B_HH[N_GATES*HIDDEN_SIZE] PROGMEM = {{
{arr1d(qb_hh)}
}};
const int16_t CLS_W[NUM_CLASSES][HIDDEN_SIZE] PROGMEM = {{
{arr2d(qC_W)}
}};
const int16_t CLS_B[NUM_CLASSES] PROGMEM = {{
{arr1d(qC_b)}
}};
#endif // MODEL_WEIGHTS_H
"""
(outdir / "model_weights.h").write_text(hdr)
print(f"Saved {outdir/'model_weights.h'} ({nz} nonzero, {nz*2} B Flash)")

# --- test vectors for the compiled-C host check ---
idx = np.linspace(0, len(X_te) - 1, 5, dtype=int)
tv = [{"window": X_te[i].tolist(), "c_pred": int(c_pred[i]),
       "logits": c_logits[i].tolist()} for i in idx]
(outdir / "test_vectors.json").write_text(json.dumps(tv, indent=2))
print(f"Saved {outdir/'test_vectors.json'} (5 vectors)")
