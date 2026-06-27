"""
Tier 2 equal-budget comparison (pre-registered: Baseline_Experiment_Protocol.md, sec 4).

Each cell at the deployment budget (~283 nonzero params / 566 B), FP32, 5 seeds,
each compressed by the means natural to it (fair to each cell's nature):
    FastGRNN  -> low-rank + IHT sparsity (its deployed recipe; numbers from the paper)
    GRU       -> shrink hidden size: H=7  (~300 params, dense)
    LSTM      -> shrink hidden size: H=6  (~306 params, dense)

This script measures the GRU/LSTM budget points. FastGRNN's budget point is its
sparse deployed model (0.856 +/- 0.099 FP32), produced by train_sparse.py under
the identical data pipeline.

Usage:
    python run_baseline_tier2.py
    python run_baseline_tier2.py --epochs 2   # smoke test
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

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
args = parser.parse_args()

# Budget-matched hidden sizes (see protocol sec 4)
BUDGET = {"gru": 7, "lstm": 6}

NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te = data["X_test"], data["y_test"]
y_tr, y_te = y_tr - 1, y_te - 1

uniq = sorted(set(s_tr.tolist()))
val_subjects = set(uniq[-4:])
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
        self.kind = kind
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, X):
        out, _ = self.rnn(X)
        return self.classifier(out[:, -1, :])


criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues, correct, total = [], [], 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(dim=1)
        correct += (p == y).sum().item(); total += len(y)
        preds.append(p.cpu().numpy()); trues.append(y.cpu().numpy())
    yp, yt = np.concatenate(preds), np.concatenate(trues)
    return correct / total, f1_score(yt, yp, average="macro"), yt, yp


def train_one(kind, hidden, seed, epochs, lr):
    Path("experiments").mkdir(exist_ok=True)
    out = f"experiments/tier2_{kind}_h{hidden}_s{seed}_e{epochs}.json"
    if Path(out).exists():
        with open(out) as f:
            r = json.load(f)
        print(f"  [{kind:5s} H{hidden} seed {seed}] SKIP F1={r['test_macro_f1']:.4f}")
        return r
    torch.manual_seed(seed); np.random.seed(seed)
    model = TorchRNNClassifier(kind, 3, hidden, NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1, best_ep, best_state = -1.0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        _, vf1, _, _ = evaluate(model, val_loader)
        if vf1 > best_f1:
            best_f1, best_ep, best_state = vf1, ep, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    te_acc, te_f1, yt, yp = evaluate(model, test_loader)
    per_class = f1_score(yt, yp, average=None).tolist()
    r = {"model": kind, "hidden": hidden, "seed": seed, "epochs": epochs,
         "n_params": int(n_params), "best_epoch": best_ep, "best_val_f1": float(best_f1),
         "test_accuracy": float(te_acc), "test_macro_f1": float(te_f1),
         "per_class_f1": {n: float(s) for n, s in zip(CLASS_NAMES, per_class)}}
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"  [{kind:5s} H{hidden} seed {seed}] F1={te_f1:.4f} acc={te_acc:.4f} "
          f"params={n_params} best_ep={best_ep} -> {out}")
    return r


def main():
    print(f"Tier 2 equal-budget | {args.epochs} epochs | seeds={args.seeds}")
    print(f"Budget-matched H: {BUDGET}\n")
    summary = {}
    for kind, hidden in BUDGET.items():
        print(f"=== {kind} (H={hidden}) ===")
        f1s = []
        n_params = None
        for seed in args.seeds:
            r = train_one(kind, hidden, seed, args.epochs, args.lr)
            f1s.append(r["test_macro_f1"]); n_params = r["n_params"]
        f1s = np.array(f1s)
        summary[kind] = {"hidden": hidden, "n_params": n_params,
                         "mean_f1": float(f1s.mean()), "std_f1": float(f1s.std()),
                         "per_seed_f1": f1s.tolist()}
        print(f"  --> {kind} H{hidden}: F1 {f1s.mean():.4f} +/- {f1s.std():.4f} "
              f"({n_params} par)\n")

    # FastGRNN budget point (from paper, identical pipeline) for the comparison table
    summary["fastgrnn"] = {"hidden": 16, "n_params": 283,
                           "mean_f1": 0.856, "std_f1": 0.099,
                           "per_seed_f1": [0.921, 0.680, 0.893, 0.895, 0.890],
                           "note": "low-rank+IHT sparse stage, FP32 (train_sparse.py); Q15~=FP32"}

    with open("experiments/tier2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== TIER 2 SUMMARY (equal budget ~283 par / 566 B, FP32) ===")
    print(f"{'model':10s} {'H':>3s} {'params':>7s} {'mean F1':>9s} {'std':>7s}")
    for k in ("fastgrnn", "gru", "lstm"):
        s = summary[k]
        print(f"{k:10s} {s['hidden']:3d} {s['n_params']:7d} {s['mean_f1']:9.4f} {s['std_f1']:7.4f}")
    print("\nSaved: experiments/tier2_summary.json")


if __name__ == "__main__":
    main()
