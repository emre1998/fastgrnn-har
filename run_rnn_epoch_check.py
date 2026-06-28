"""
Fairness check: give GRU/LSTM the SAME total training budget as FastGRNN's
L-S-Q pipeline (l_epochs + s_epochs = 200) and re-measure their shrink-H
deployment-budget accuracy.

Rules out the confound "FastGRNN won only because it trained 2x longer."
If GRU/LSTM do not improve at 200 epochs, FastGRNN's deployment-budget
advantage is architectural (compressibility), not an optimization artifact.

Reads the byte budget (FastGRNN total_nonzero) from the existing
deploy_{tag}_s*.json. Saves deploy_rnn{epochs}_{tag}_s{seed}.json.

Usage:
  python run_rnn_epoch_check.py --data data/processed/wisdm_windows.npz --seed 0 --epochs 200
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
from quantize import q15_round

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/processed/hapt_windows.npz")
parser.add_argument("--tag", default=None)
parser.add_argument("--val_holdout", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

TAG = args.tag or Path(args.data).stem.replace("_windows", "")
torch.manual_seed(args.seed); np.random.seed(args.seed)

# byte budget = FastGRNN total nonzero from the matching deploy run
ref = json.load(open(f"experiments/deploy_{TAG}_s{args.seed}.json"))
BUDGET = ref["fastgrnn"]["total_nonzero"]
print(f"[{TAG} s{args.seed}] RNN epoch-check | budget={BUDGET} | epochs={args.epochs}")

data = np.load(args.data, allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te = data["X_test"], data["y_test"]
y_tr, y_te = y_tr - 1, y_te - 1
NUM_CLASSES = int(max(y_tr.max(), y_te.max())) + 1
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
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def macro_f1(model, ld):
    model.eval(); P, T = [], []
    for x, y in ld:
        P.append(model(x).argmax(1).numpy()); T.append(y.numpy())
    return f1_score(np.concatenate(T), np.concatenate(P), average="macro")


class SmallRNN(nn.Module):
    def __init__(self, kind, h):
        super().__init__()
        self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(3, h, batch_first=True)
        self.classifier = nn.Linear(h, NUM_CLASSES)

    def forward(self, X):
        out, _ = self.rnn(X)
        return self.classifier(out[:, -1, :])


def total_params(kind, h):
    g = 3 if kind == "gru" else 4
    return g * (h * 3 + h * h + 2 * h) + h * NUM_CLASSES + NUM_CLASSES


def fit_hidden(kind):
    h = 1
    while total_params(kind, h + 1) <= BUDGET:
        h += 1
    return h


def run(kind):
    h = fit_hidden(kind)
    m = SmallRNN(kind, h)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    best, best_state = -1, None
    for ep in range(args.epochs):
        m.train()
        for x, y in train_loader:
            opt.zero_grad(); loss = criterion(m(x), y); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        vf1 = macro_f1(m, val_loader)
        if vf1 > best:
            best, best_state = vf1, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    fp32 = macro_f1(m, test_loader)
    with torch.no_grad():
        for p in list(m.rnn.parameters()) + list(m.classifier.parameters()):
            p.data.copy_(q15_round(p.data)[0])
    q15 = macro_f1(m, test_loader)
    return {"hidden": int(h), "fp32_f1": float(fp32), "q15_f1": float(q15),
            "total_nonzero": int(total_params(kind, h))}


out = f"experiments/deploy_rnn{args.epochs}_{TAG}_s{args.seed}.json"
if Path(out).exists():
    print(f"SKIP {out}")
else:
    res = {"dataset": TAG, "seed": args.seed, "epochs": args.epochs, "budget": BUDGET}
    for kind in ("gru", "lstm"):
        res[kind] = run(kind)
        print(f"  {kind} H{res[kind]['hidden']} FP32={res[kind]['fp32_f1']:.3f} Q15={res[kind]['q15_f1']:.3f}")
    json.dump(res, open(out, "w"), indent=2)
    print(f"Saved {out}")
