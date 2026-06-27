"""
Tier 1 baseline control (pre-registered: Baseline_Experiment_Protocol.md).

Matched-hidden-size (H=16), FP32, no compression head-to-head:
    FastGRNN vs GRU vs LSTM
under one identical harness (same data pipeline, optimizer, schedule, seeds)
as train_fastgrnn.py.

Fills the borrowed "---" F1 cells of tab:baselines with measured numbers.

Usage:
    python run_baseline_tier1.py                 # all 3 models, seeds 0-4, 120 epochs
    python run_baseline_tier1.py --epochs 2 --models gru --seeds 0   # smoke test
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
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--models", nargs="+", default=["fastgrnn", "gru", "lstm"])
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
args = parser.parse_args()

NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

# ----------------------------------------------------------------------------
# Data — identical pipeline to train_fastgrnn.py
# ----------------------------------------------------------------------------
data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te = data["X_test"], data["y_test"]
y_tr = y_tr - 1
y_te = y_te - 1

uniq = sorted(set(s_tr.tolist()))
val_subjects = set(uniq[-4:])
val_mask = np.array([s in val_subjects for s in s_tr])
X_trn, y_trn = X_tr[~val_mask], y_tr[~val_mask]
X_val, y_val = X_tr[val_mask], y_tr[val_mask]

mean = X_trn.mean(axis=(0, 1))
std = X_trn.std(axis=(0, 1)) + 1e-8
def normalize(X): return ((X - mean) / std).astype(np.float32)
X_trn_n, X_val_n, X_te_n = normalize(X_trn), normalize(X_val), normalize(X_te)

def make_loader(X, y, batch_size=64, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                      batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_trn_n, y_trn, 64, shuffle=True)
val_loader = make_loader(X_val_n, y_val, 256)
test_loader = make_loader(X_te_n, y_te, 256)


class TorchRNNClassifier(nn.Module):
    """GRU/LSTM at matched H; final hidden state -> linear head (mirrors
    FastGRNNClassifier's final-h readout)."""
    def __init__(self, kind, input_size, hidden_size, num_classes):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif kind == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(kind)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        out, _ = self.rnn(X)            # (B, T, H)
        return self.classifier(out[:, -1, :])   # final timestep


def build_model(kind, hidden):
    if kind == "fastgrnn":
        return FastGRNNClassifier(input_size=3, hidden_size=hidden, num_classes=NUM_CLASSES)
    return TorchRNNClassifier(kind, input_size=3, hidden_size=hidden, num_classes=NUM_CLASSES)


criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    correct, total = 0, 0
    for x, y in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += len(y)
        all_pred.append(pred.numpy())
        all_true.append(y.numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    return correct / total, f1_score(y_true, y_pred, average="macro"), y_true, y_pred


def train_one(kind, seed, epochs, lr, hidden):
    Path("experiments").mkdir(exist_ok=True)
    out = f"experiments/baseline_{kind}_h{hidden}_s{seed}_e{epochs}.json"
    if Path(out).exists():
        with open(out) as f:
            result = json.load(f)
        print(f"  [{kind:8s} seed {seed}] SKIP (already done)  "
              f"F1={result['test_macro_f1']:.4f}  -> {out}")
        return result

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(kind, hidden)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        _, val_f1, _, _ = evaluate(model, val_loader)
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    te_acc, te_f1, y_true, y_pred = evaluate(model, test_loader)
    per_class = f1_score(y_true, y_pred, average=None).tolist()
    result = {
        "model": kind, "hidden": hidden, "seed": seed,
        "epochs": epochs, "lr": lr,
        "n_params": int(n_params),
        "best_epoch": best_epoch, "best_val_f1": float(best_val_f1),
        "test_accuracy": float(te_acc), "test_macro_f1": float(te_f1),
        "per_class_f1": {n: float(s) for n, s in zip(CLASS_NAMES, per_class)},
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{kind:8s} seed {seed}] F1={te_f1:.4f}  acc={te_acc:.4f}  "
          f"params={n_params}  best_epoch={best_epoch}  -> {out}")
    return result


def main():
    print(f"Tier 1 baseline control | H={args.hidden} | {args.epochs} epochs | "
          f"models={args.models} | seeds={args.seeds}\n")
    summary = {}
    for kind in args.models:
        f1s, accs, n_params = [], [], None
        print(f"=== {kind} ===")
        for seed in args.seeds:
            r = train_one(kind, seed, args.epochs, args.lr, args.hidden)
            f1s.append(r["test_macro_f1"])
            accs.append(r["test_accuracy"])
            n_params = r["n_params"]
        f1s = np.array(f1s)
        summary[kind] = {
            "n_params": n_params,
            "mean_f1": float(f1s.mean()), "std_f1": float(f1s.std()),
            "per_seed_f1": f1s.tolist(),
            "mean_acc": float(np.mean(accs)),
        }
        print(f"  --> {kind}: F1 {f1s.mean():.4f} +/- {f1s.std():.4f}  "
              f"(params {n_params})\n")

    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/baseline_tier1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TIER 1 SUMMARY (matched H, FP32) ===")
    print(f"{'model':10s} {'params':>7s} {'mean F1':>9s} {'std':>7s}   per-seed")
    for kind, s in summary.items():
        seeds_str = " ".join(f"{v:.3f}" for v in s["per_seed_f1"])
        print(f"{kind:10s} {s['n_params']:7d} {s['mean_f1']:9.4f} "
              f"{s['std_f1']:7.4f}   [{seeds_str}]")
    print("\nSaved: experiments/baseline_tier1_summary.json")


if __name__ == "__main__":
    main()
