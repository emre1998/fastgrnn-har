"""
Deployment-budget comparison at the actual on-device operating point, per dataset.

Self-contained pipeline (no checkpoint passing between scripts). EACH cell
reaches the SAME byte budget by the compression NATURAL TO ITS ARCHITECTURE --
we do not force GRU/LSTM through FastGRNN's low-rank+IHT scheme (that would be an
apples-to-potato-chips comparison):
  FastGRNN : Low-rank (r_w=2, r_u=8) -> IHT sparsity 0.5 (cubic ramp) -> Q15
             (calibrated weights + activations -- FastGRNN's designed deploy recipe)
  GRU/LSTM : shrink hidden size to the largest H that fits the byte budget
             (dense, the natural compact form for a vanilla RNN) -> weight-Q15

Reports, per cell, FP32 and Q15 macro-F1 + the nonzero/byte budget. Run one
seed per process for parallelism; aggregate with analyze_deploy_budget.py.

Identical-treatment note: the calibrated activation-Q15 (LUT) recipe is
FastGRNN-specific; GRU/LSTM get weight-Q15 only (their activations stay FP32).
This is the CONSERVATIVE direction -- it slightly favors the baselines, which
already win/tie, so it cannot manufacture a pro-FastGRNN result.

Usage:
  python run_deploy_budget.py --data data/processed/wisdm_windows.npz --seed 0
"""
import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from pathlib import Path

from fastgrnn_model import FastGRNNClassifier
from quantize import (quantize_weights, calibrate_activations,
                      wrap_cell_with_calibrated_quantization, q15_round)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/processed/hapt_windows.npz")
parser.add_argument("--tag", default=None)
parser.add_argument("--val_holdout", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--r_w", type=int, default=2)
parser.add_argument("--r_u", type=int, default=8)
parser.add_argument("--l_epochs", type=int, default=100)
parser.add_argument("--s_epochs", type=int, default=100)
parser.add_argument("--ramp_epochs", type=int, default=50)
parser.add_argument("--target_sparsity", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

TAG = args.tag or Path(args.data).stem.replace("_windows", "")
DEVICE = torch.device("cpu")   # FastGRNN's T-loop is faster on CPU
torch.manual_seed(args.seed); np.random.seed(args.seed)
print(f"[{TAG} seed {args.seed}] deploy-budget L-S-Q pipeline")

# ---------------- data (same pipeline as tier1) ----------------
data = np.load(args.data, allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te = data["X_test"], data["y_test"]
y_tr, y_te = y_tr - 1, y_te - 1
NUM_CLASSES = int(max(y_tr.max(), y_te.max())) + 1
CLASS_NAMES = ([str(s).split(" ", 1)[-1] for s in data["activity_labels"]]
               if "activity_labels" in data else [f"class{i}" for i in range(NUM_CLASSES)])
uniq = sorted(set(s_tr.tolist())); val_subjects = set(uniq[-args.val_holdout:])
vmask = np.array([s in val_subjects for s in s_tr])
X_trn, y_trn = X_tr[~vmask], y_tr[~vmask]
X_val, y_val = X_tr[vmask], y_tr[vmask]
mean = X_trn.mean(axis=(0, 1)); std = X_trn.std(axis=(0, 1)) + 1e-8
norm = lambda X: ((X - mean) / std).astype(np.float32)
def loader(X, y, bs=64, sh=False):
    return DataLoader(TensorDataset(torch.from_numpy(norm(X)), torch.from_numpy(y).long()),
                      batch_size=bs, shuffle=sh)
train_loader = loader(X_trn, y_trn, 64, True)
val_loader = loader(X_val, y_val, 256)
test_loader = loader(X_te, y_te, 256)
calib_loader = loader(X_trn, y_trn, 256, False)
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def macro_f1(model, ld=test_loader):
    model.eval(); P, Tr = [], []
    for x, y in ld:
        P.append(model(x).argmax(1).numpy()); Tr.append(y.numpy())
    return f1_score(np.concatenate(Tr), np.concatenate(P), average="macro")


def head_params():
    return args.hidden * NUM_CLASSES + NUM_CLASSES   # Linear(H, C)


# ======================= FastGRNN L -> S -> Q =======================
def fastgrnn_lsq():
    m = FastGRNNClassifier(3, args.hidden, NUM_CLASSES, r_w=args.r_w, r_u=args.r_u, sparse=True)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)

    # --- L: low-rank dense (masks all ones), best-val ---
    best, best_state = -1, None
    for ep in range(args.l_epochs):
        m.train()
        for x, y in train_loader:
            opt.zero_grad(); loss = criterion(m(x), y); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        vf1 = macro_f1(m, val_loader)
        if vf1 > best:
            best, best_state = vf1, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)

    # --- S: IHT cubic ramp to target sparsity, best-val after ramp ---
    def sp_at(ep):
        if ep > args.ramp_epochs:
            return args.target_sparsity
        t = ep / args.ramp_epochs
        return args.target_sparsity * (1 - (1 - t) ** 3)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    best, best_state = -1, None
    for ep in range(1, args.s_epochs + 1):
        m.cell.apply_pruning(sp_at(ep))
        m.train()
        for x, y in train_loader:
            opt.zero_grad(); loss = criterion(m(x), y); loss.backward()
            with torch.no_grad():
                m.cell.W1.grad.mul_(m.cell.mask_W1); m.cell.W2.grad.mul_(m.cell.mask_W2)
                m.cell.U1.grad.mul_(m.cell.mask_U1); m.cell.U2.grad.mul_(m.cell.mask_U2)
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            with torch.no_grad():
                m.cell.W1.data.mul_(m.cell.mask_W1); m.cell.W2.data.mul_(m.cell.mask_W2)
                m.cell.U1.data.mul_(m.cell.mask_U1); m.cell.U2.data.mul_(m.cell.mask_U2)
        vf1 = macro_f1(m, val_loader)
        if ep > args.ramp_epochs and vf1 > best:
            best, best_state = vf1, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)

    sparse_f1 = macro_f1(m)
    cell_nz = m.cell.effective_params()
    total_nz = cell_nz + head_params()

    # --- Q: calibrated Q15 (weights + activations) ---
    stats = calibrate_activations(m, calib_loader, n_batches=8)
    quantize_weights(m, verbose=False)
    wrap_cell_with_calibrated_quantization(m.cell, stats)
    q15_f1 = macro_f1(m)
    return {"sparse_fp32_f1": float(sparse_f1), "q15_f1": float(q15_f1),
            "cell_nonzero": int(cell_nz), "total_nonzero": int(total_nz)}


# ======================= GRU/LSTM shrink-H (natural compact form) -> weight-Q15 =======================
class SmallRNN(nn.Module):
    def __init__(self, kind, h):
        super().__init__()
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(3, h, batch_first=True)
        self.classifier = nn.Linear(h, NUM_CLASSES)

    def forward(self, X):
        out, _ = self.rnn(X)
        return self.classifier(out[:, -1, :])


def rnn_total_params(kind, h):
    gates = 3 if kind == "gru" else 4
    cell = gates * (h * 3 + h * h + 2 * h)          # ih + hh + 2 biases
    head = h * NUM_CLASSES + NUM_CLASSES
    return cell + head


def fit_hidden(kind, budget):
    """Largest H whose dense param count fits the byte budget."""
    h = 1
    while rnn_total_params(kind, h + 1) <= budget:
        h += 1
    return h


def rnn_budget(kind, budget):
    h = fit_hidden(kind, budget)
    total = rnn_total_params(kind, h)
    m = SmallRNN(kind, h)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    best, best_state = -1, None
    for ep in range(1, args.s_epochs + 1):          # same epoch budget as FastGRNN S stage
        m.train()
        for x, y in train_loader:
            opt.zero_grad(); loss = criterion(m(x), y); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        vf1 = macro_f1(m, val_loader)
        if vf1 > best:
            best, best_state = vf1, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    fp32 = macro_f1(m)

    # weight-Q15: per-tensor int16 rounding (architecture-agnostic)
    with torch.no_grad():
        for p in list(m.rnn.parameters()) + list(m.classifier.parameters()):
            p.data.copy_(q15_round(p.data)[0])
    q15 = macro_f1(m)
    return {"hidden": int(h), "fp32_f1": float(fp32), "q15_f1": float(q15),
            "cell_nonzero": int(total - (h * NUM_CLASSES + NUM_CLASSES)),
            "total_nonzero": int(total)}


def main():
    Path("experiments").mkdir(exist_ok=True)
    out = f"experiments/deploy_{TAG}_s{args.seed}.json"
    if Path(out).exists():
        print(f"SKIP (exists): {out}"); return
    res = {"dataset": TAG, "seed": args.seed, "num_classes": NUM_CLASSES,
           "head_params": head_params()}
    print("  FastGRNN L-S-Q ...")
    res["fastgrnn"] = fastgrnn_lsq()
    budget = res["fastgrnn"]["total_nonzero"]   # byte budget GRU/LSTM must fit (shrink-H)
    print(f"  byte budget (total nonzero) = {budget}")
    for kind in ("gru", "lstm"):
        print(f"  {kind} shrink-H to budget + weight-Q15 ...")
        res[kind] = rnn_budget(kind, budget)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")
    for c in ("fastgrnn", "gru", "lstm"):
        r = res[c]
        fp = r.get("sparse_fp32_f1", r.get("fp32_f1"))
        print(f"  {c:9s} nz={r['total_nonzero']:4d}  FP32={fp:.3f}  Q15={r['q15_f1']:.3f}")


if __name__ == "__main__":
    main()
