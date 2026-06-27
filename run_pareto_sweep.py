"""
Pareto sweep — accuracy vs parameter count, dense models, all three cells.

Closes the low-budget regime (FastGRNN's design home) and builds the
accuracy-vs-params Pareto curve for a reviewer-proof presentation. All cells
DENSE (no compression) so only architecture + size vary; symmetric and fair.

Combined with the H=16 points (Tier 1) and the compressed budget points (Tier 2),
this answers: is there ANY size at which FastGRNN beats GRU/LSTM?

Runs on GPU if available. FastGRNNClassifier already creates its hidden state on
the input's device, so it moves to CUDA cleanly.

Usage:
    python run_pareto_sweep.py
    python run_pareto_sweep.py --hiddens 8 --models fastgrnn --seeds 0 --epochs 4  # smoke
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

parser = argparse.ArgumentParser()
parser.add_argument("--hiddens", nargs="+", type=int, default=[4, 6, 8, 10, 12])
parser.add_argument("--models", nargs="+", default=["fastgrnn", "gru", "lstm"])
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te = data["X_test"], data["y_test"]
y_tr, y_te = y_tr - 1, y_te - 1
uniq = sorted(set(s_tr.tolist())); val_subjects = set(uniq[-4:])
val_mask = np.array([s in val_subjects for s in s_tr])
X_trn, y_trn = X_tr[~val_mask], y_tr[~val_mask]
X_val, y_val = X_tr[val_mask], y_tr[val_mask]
mean = X_trn.mean(axis=(0, 1)); std = X_trn.std(axis=(0, 1)) + 1e-8
def normalize(X): return ((X - mean) / std).astype(np.float32)
X_trn_n, X_val_n, X_te_n = normalize(X_trn), normalize(X_val), normalize(X_te)
def make_loader(X, y, bs=64, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                      batch_size=bs, shuffle=shuffle)
train_loader = make_loader(X_trn_n, y_trn, 64, True)
val_loader = make_loader(X_val_n, y_val, 256)
test_loader = make_loader(X_te_n, y_te, 256)


class TorchRNNClassifier(nn.Module):
    def __init__(self, kind, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, X):
        out, _ = self.rnn(X)
        return self.classifier(out[:, -1, :])


def build(kind, h):
    if kind == "fastgrnn":
        return FastGRNNClassifier(input_size=3, hidden_size=h, num_classes=NUM_CLASSES)
    return TorchRNNClassifier(kind, 3, h, NUM_CLASSES)


criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, trues, c, t = [], [], 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(dim=1); c += (p == y).sum().item(); t += len(y)
        preds.append(p.cpu().numpy()); trues.append(y.cpu().numpy())
    yp, yt = np.concatenate(preds), np.concatenate(trues)
    return c / t, f1_score(yt, yp, average="macro")


def train_one(kind, h, seed):
    Path("experiments").mkdir(exist_ok=True)
    out = f"experiments/pareto_{kind}_h{h}_s{seed}_e{args.epochs}.json"
    if Path(out).exists():
        with open(out) as f:
            r = json.load(f)
        print(f"  [{kind:8s} H{h:<2d} s{seed}] SKIP F1={r['test_macro_f1']:.4f}")
        return r
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(kind, h).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1, best_ep, best_state = -1.0, 0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        _, vf1 = evaluate(model, val_loader)
        if vf1 > best_f1:
            best_f1, best_ep, best_state = vf1, ep, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    te_acc, te_f1 = evaluate(model, test_loader)
    r = {"model": kind, "hidden": h, "seed": seed, "n_params": int(n_params),
         "best_epoch": best_ep, "test_accuracy": float(te_acc), "test_macro_f1": float(te_f1)}
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"  [{kind:8s} H{h:<2d} s{seed}] F1={te_f1:.4f} params={n_params} -> {out}")
    return r


def main():
    print(f"Pareto sweep | H={args.hiddens} | models={args.models} | "
          f"seeds={args.seeds} | {args.epochs} ep\n")
    summary = {}
    for kind in args.models:
        for h in args.hiddens:
            f1s, npar = [], None
            for seed in args.seeds:
                r = train_one(kind, h, seed); f1s.append(r["test_macro_f1"]); npar = r["n_params"]
            f1s = np.array(f1s)
            key = f"{kind}_h{h}"
            summary[key] = {"model": kind, "hidden": h, "n_params": npar,
                            "mean_f1": float(f1s.mean()), "std_f1": float(f1s.std()),
                            "per_seed_f1": f1s.tolist()}
            print(f"  --> {key}: {npar} par, F1 {f1s.mean():.4f} +/- {f1s.std():.4f}\n")
    with open("experiments/pareto_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== PARETO SWEEP SUMMARY (dense, accuracy vs params) ===")
    print(f"{'model':10s} {'H':>3s} {'params':>7s} {'mean F1':>9s} {'std':>7s}")
    for k, s in sorted(summary.items(), key=lambda kv: (kv[1]['model'], kv[1]['n_params'])):
        print(f"{s['model']:10s} {s['hidden']:3d} {s['n_params']:7d} {s['mean_f1']:9.4f} {s['std_f1']:7.4f}")
    print("\nSaved: experiments/pareto_summary.json")


if __name__ == "__main__":
    main()
