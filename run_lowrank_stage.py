"""
L-stage of the L-S-Q ablation: low-rank FastGRNN (r_w=2, r_u=8), NO sparsity, FP32.

Mirrors run_deploy_budget's L phase exactly so the ablation row is comparable:
dense full-rank (Tier1) -> low-rank (this) -> +sparse (deploy sparse_fp32) -> +Q15 (deploy q15).

Isolates whether accuracy/variance enters at the low-rank (L) step or the sparsity (S) step.
One seed per process for parallelism.

Usage: python run_lowrank_stage.py --data data/processed/wisdm_windows.npz --seed 0 --val_holdout 4
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
parser.add_argument("--data", default="data/processed/hapt_windows.npz")
parser.add_argument("--tag", default=None)
parser.add_argument("--val_holdout", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--r_w", type=int, default=2)
parser.add_argument("--r_u", type=int, default=8)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

TAG = args.tag or Path(args.data).stem.replace("_windows", "")
DEVICE = torch.device("cpu")
torch.manual_seed(args.seed); np.random.seed(args.seed)

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
train_loader, val_loader, test_loader = loader(X_trn, y_trn, 64, True), loader(X_val, y_val, 256), loader(X_te, y_te, 256)
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def macro_f1(model, ld):
    model.eval(); P, T = [], []
    for x, y in ld:
        P.append(model(x).argmax(1).numpy()); T.append(y.numpy())
    yt, yp = np.concatenate(T), np.concatenate(P)
    return f1_score(yt, yp, average="macro"), yt, yp


def main():
    Path("experiments").mkdir(exist_ok=True)
    out = f"experiments/lowrank_{TAG}_s{args.seed}.json"
    if Path(out).exists():
        print(f"SKIP {out}"); return
    model = FastGRNNClassifier(3, args.hidden, NUM_CLASSES, r_w=args.r_w, r_u=args.r_u, sparse=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        vf1, _, _ = macro_f1(model, val_loader)
        if vf1 > best:
            best, best_state = vf1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    te_f1, yt, yp = macro_f1(model, test_loader)
    per_class = f1_score(yt, yp, average=None).tolist()
    n_params = sum(p.numel() for p in model.parameters())
    r = {"dataset": TAG, "seed": args.seed, "stage": "lowrank",
         "n_params": int(n_params), "test_macro_f1": float(te_f1),
         "per_class_f1": {n: float(s) for n, s in zip(CLASS_NAMES, per_class)}}
    json.dump(r, open(out, "w"), indent=2)
    print(f"[{TAG} s{args.seed}] low-rank F1={te_f1:.4f} params={n_params} -> {out}")


if __name__ == "__main__":
    main()
