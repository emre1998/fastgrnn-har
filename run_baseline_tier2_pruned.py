"""
Tier 2 — pruned route for GRU/LSTM (closes the "condemned to tiny H" objection).

Gives each baseline its SECOND route to the equal budget: keep H=16 but
magnitude-prune the cell to ~181 nonzero (head dense, 102) => 283 total / 566 B,
exactly analogous to FastGRNN's IHT sparsity stage. Cubic sparsity ramp + finetune,
mirroring train_sparse.py.

For each baseline the paper reports max(shrink-H dense, this pruned-H16) — its best
shot at the budget. If pruned-H16 < dense-small-H (the expected outcome), it proves
shrink-H was the fair/best route, not a handicap.

Usage:
    python run_baseline_tier2_pruned.py
    python run_baseline_tier2_pruned.py --epochs 4   # smoke test
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
parser.add_argument("--ramp_epochs", type=int, default=60)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--keep_cell", type=int, default=181,
                    help="Target nonzero cell params (FastGRNN deployed = 181)")
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--models", nargs="+", default=["gru", "lstm"])
args = parser.parse_args()

NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
CELL_PARAMS = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]
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


class PrunedRNN(nn.Module):
    def __init__(self, kind, hidden):
        super().__init__()
        self.kind = kind
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(3, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, NUM_CLASSES)
        self.masks = {n: torch.ones_like(getattr(self.rnn, n)) for n in CELL_PARAMS}

    def forward(self, X):
        out, _ = self.rnn(X)
        return self.classifier(out[:, -1, :])

    def cell_total(self):
        return sum(getattr(self.rnn, n).numel() for n in CELL_PARAMS)

    @torch.no_grad()
    def apply_pruning(self, keep):
        params = [getattr(self.rnn, n) for n in CELL_PARAMS]
        allw = torch.cat([p.detach().abs().flatten() for p in params])
        total = allw.numel()
        keep = max(1, min(keep, total))
        thresh = torch.topk(allw, keep, largest=True).values.min()
        for n, p in zip(CELL_PARAMS, params):
            m = (p.detach().abs() >= thresh).float()
            self.masks[n] = m
            p.data.mul_(m)

    @torch.no_grad()
    def reapply(self):
        for n in CELL_PARAMS:
            getattr(self.rnn, n).data.mul_(self.masks[n])

    def mask_grads(self):
        for n in CELL_PARAMS:
            g = getattr(self.rnn, n).grad
            if g is not None:
                g.mul_(self.masks[n])

    def nonzero_total(self):
        cell = int(sum(m.sum().item() for m in self.masks.values()))
        head = sum(p.numel() for p in self.classifier.parameters())
        return cell, head, cell + head


criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, trues, c, t = [], [], 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(dim=1); c += (p == y).sum().item(); t += len(y)
        preds.append(p.cpu().numpy()); trues.append(y.cpu().numpy())
    yp, yt = np.concatenate(preds), np.concatenate(trues)
    return c / t, f1_score(yt, yp, average="macro"), yt, yp


def keep_at(epoch, total, target_keep, ramp):
    if epoch >= ramp:
        return target_keep
    sp_target = 1 - target_keep / total
    t = epoch / ramp
    sp = sp_target * (1 - (1 - t) ** 3)
    return int(round((1 - sp) * total))


def train_one(kind, seed):
    Path("experiments").mkdir(exist_ok=True)
    out = f"experiments/tier2pruned_{kind}_h{args.hidden}_s{seed}_e{args.epochs}.json"
    if Path(out).exists():
        with open(out) as f:
            r = json.load(f)
        print(f"  [{kind:5s} pruned seed {seed}] SKIP F1={r['test_macro_f1']:.4f}")
        return r
    torch.manual_seed(seed); np.random.seed(seed)
    model = PrunedRNN(kind, args.hidden).to(DEVICE)
    for n in model.masks:
        model.masks[n] = model.masks[n].to(DEVICE)
    total = model.cell_total()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1, best_ep, best_state, best_nz = -1.0, 0, None, None
    for ep in range(1, args.epochs + 1):
        model.apply_pruning(keep_at(ep, total, args.keep_cell, args.ramp_epochs))
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward()
            model.mask_grads()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); model.reapply()
        _, vf1, _, _ = evaluate(model, val_loader)
        if ep > args.ramp_epochs and vf1 > best_f1:
            best_f1, best_ep = vf1, ep
            best_state = copy.deepcopy(model.state_dict())
            best_nz = model.nonzero_total()
    model.load_state_dict(best_state)
    te_acc, te_f1, yt, yp = evaluate(model, test_loader)
    per_class = f1_score(yt, yp, average=None).tolist()
    cell_nz, head_nz, tot_nz = best_nz
    r = {"model": kind + "_pruned", "hidden": args.hidden, "seed": seed,
         "cell_nonzero": cell_nz, "head": head_nz, "total_nonzero": tot_nz,
         "best_epoch": best_ep, "best_val_f1": float(best_f1),
         "test_accuracy": float(te_acc), "test_macro_f1": float(te_f1),
         "per_class_f1": {n: float(s) for n, s in zip(CLASS_NAMES, per_class)}}
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"  [{kind:5s} pruned seed {seed}] F1={te_f1:.4f} acc={te_acc:.4f} "
          f"nz={tot_nz} (cell {cell_nz}+head {head_nz}) best_ep={best_ep} -> {out}")
    return r


def main():
    print(f"Tier 2 PRUNED route | H={args.hidden} keep_cell={args.keep_cell} | "
          f"{args.epochs} ep (ramp {args.ramp_epochs}) | seeds={args.seeds}\n")
    summary = {}
    for kind in args.models:
        print(f"=== {kind} pruned (H={args.hidden} -> {args.keep_cell} cell nonzero) ===")
        f1s, nz = [], None
        for seed in args.seeds:
            r = train_one(kind, seed); f1s.append(r["test_macro_f1"]); nz = r["total_nonzero"]
        f1s = np.array(f1s)
        summary[kind + "_pruned"] = {"total_nonzero": nz,
            "mean_f1": float(f1s.mean()), "std_f1": float(f1s.std()),
            "per_seed_f1": f1s.tolist()}
        print(f"  --> {kind} pruned: F1 {f1s.mean():.4f} +/- {f1s.std():.4f} "
              f"({nz} nonzero)\n")
    with open("experiments/tier2pruned_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: experiments/tier2pruned_summary.json")


if __name__ == "__main__":
    main()
