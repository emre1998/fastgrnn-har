"""
FastGRNN training - same conditions as the MLP baseline.

Goal: beat the MLP baseline of 85.47% accuracy / 84.73% macro-F1,
especially the per-class F1 on UPSTAIRS.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from pathlib import Path
from fastgrnn_model import FastGRNNClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--r_w", type=int, default=None, help="Low-rank W rank (default: vanilla)")
parser.add_argument("--r_u", type=int, default=None, help="Low-rank U rank (default: vanilla)")
parser.add_argument("--patience", type=int, default=0, help="Early-stopping patience (0 = disabled)")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--tag_seed", action="store_true", help="Append _s{seed} to the output filename")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ----------------------------------------------------------------------------
# 1) Data (identical pipeline to the MLP baseline)
# ----------------------------------------------------------------------------
data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te, s_te = data["X_test"],  data["y_test"],  data["subjects_test"]

y_tr = y_tr - 1   # 1-6 -> 0-5
y_te = y_te - 1
NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

# Subject-aware val (last four training subjects held out)
uniq = sorted(set(s_tr.tolist()))
val_subjects = set(uniq[-4:])
val_mask = np.array([s in val_subjects for s in s_tr])
X_trn, y_trn = X_tr[~val_mask], y_tr[~val_mask]
X_val, y_val = X_tr[ val_mask], y_tr[ val_mask]

mean = X_trn.mean(axis=(0, 1))
std  = X_trn.std(axis=(0, 1)) + 1e-8
def normalize(X): return ((X - mean) / std).astype(np.float32)
X_trn_n = normalize(X_trn)
X_val_n = normalize(X_val)
X_te_n  = normalize(X_te)

print(f"Train: {X_trn_n.shape}  Val: {X_val_n.shape}  Test: {X_te_n.shape}")

def make_loader(X, y, batch_size=64, shuffle=False):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
        batch_size=batch_size, shuffle=shuffle,
    )

train_loader = make_loader(X_trn_n, y_trn, batch_size=64, shuffle=True)
val_loader   = make_loader(X_val_n, y_val, batch_size=256)
test_loader  = make_loader(X_te_n,  y_te,  batch_size=256)

# ----------------------------------------------------------------------------
# 2) Model
# ----------------------------------------------------------------------------
HIDDEN = args.hidden
EPOCHS = args.epochs
seed_suffix = f"_s{args.seed}" if args.tag_seed else ""
if args.r_w is not None and args.r_u is not None:
    TAG = f"h{HIDDEN}_rw{args.r_w}_ru{args.r_u}{seed_suffix}_e{EPOCHS}"
else:
    TAG = f"h{HIDDEN}{seed_suffix}_e{EPOCHS}"
model = FastGRNNClassifier(
    input_size=3, hidden_size=HIDDEN, num_classes=NUM_CLASSES,
    r_w=args.r_w, r_u=args.r_u,
)
n_params = sum(p.numel() for p in model.parameters())
variant = f"H={HIDDEN}"
if args.r_w is not None:
    variant += f", r_w={args.r_w}, r_u={args.r_u} (LOW-RANK)"
print(f"\nModel: FastGRNNClassifier ({variant})")
print(f"Parameter count: {n_params}  (MLP baseline: 12,518)")

# ----------------------------------------------------------------------------
# 3) Training
# ----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += len(y)
        all_pred.append(pred.numpy())
        all_true.append(y.numpy())
    avg_loss = total_loss / total
    acc = total_correct / total
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, macro_f1, y_true, y_pred

print(f"\n{'epoch':>5} {'train_loss':>11} {'val_loss':>9} {'val_acc':>8} {'val_f1':>8}  {'zeta':>6} {'nu':>6}")
best_val_f1 = -1
best_epoch = 0
epochs_no_improve = 0
last_epoch_run = 0
for epoch in range(1, EPOCHS + 1):
    last_epoch_run = epoch
    model.train()
    total_loss, n = 0.0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # Gradient clipping - standard for RNNs, avoids exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    train_loss = total_loss / n
    val_loss, val_acc, val_f1, _, _ = evaluate(val_loader)

    # Track where zeta and nu are heading during training
    with torch.no_grad():
        zeta = torch.sigmoid(model.cell.zeta_raw).item()
        nu   = torch.sigmoid(model.cell.nu_raw).item()

    flag = ""
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"fastgrnn_{TAG}_best.pt")
        flag = "  *best*"
    else:
        epochs_no_improve += 1
    print(f"{epoch:5d} {train_loss:11.4f} {val_loss:9.4f} {val_acc:8.4f} {val_f1:8.4f}  {zeta:6.3f} {nu:6.3f}{flag}")

    if args.patience > 0 and epochs_no_improve >= args.patience:
        print(f"\n[EARLY STOP] val_f1 has not improved for {args.patience} epochs. "
              f"Best epoch: {best_epoch} (val_f1={best_val_f1:.4f}).")
        break

# ----------------------------------------------------------------------------
# 4) Test
# ----------------------------------------------------------------------------
model.load_state_dict(torch.load(f"fastgrnn_{TAG}_best.pt"))
te_loss, te_acc, te_f1, y_true, y_pred = evaluate(test_loader)

print(f"\n=== TEST SET ===")
print(f"Accuracy : {te_acc:.4f}   (MLP: 0.8547)")
print(f"Macro-F1 : {te_f1:.4f}   (MLP: 0.8473)")
print(f"Loss     : {te_loss:.4f}")

# Learned zeta and nu
with torch.no_grad():
    final_zeta = torch.sigmoid(model.cell.zeta_raw).item()
    final_nu   = torch.sigmoid(model.cell.nu_raw).item()
print(f"\nLearned zeta = {final_zeta:.4f}   nu = {final_nu:.4f}  (init: 0.5/0.5)")

print("\n=== Per-class report ===")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

print("=== Confusion matrix ===")
cm = confusion_matrix(y_true, y_pred)
header = "       " + " ".join(f"{n[:5]:>6}" for n in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    row = " ".join(f"{cm[i, j]:>6d}" for j in range(NUM_CLASSES))
    print(f"{name[:6]:6s} {row}")

# Per-class F1
per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()

# Save
results = {
    "model": "FastGRNN" + (" (low-rank)" if args.r_w is not None else ""),
    "hidden_size": HIDDEN,
    "r_w": args.r_w,
    "r_u": args.r_u,
    "epochs_planned": EPOCHS,
    "epochs_run": last_epoch_run,
    "best_epoch": best_epoch,
    "patience": args.patience,
    "lr": args.lr,
    "n_params": int(n_params),
    "test_accuracy": float(te_acc),
    "test_macro_f1": float(te_f1),
    "test_loss": float(te_loss),
    "per_class_f1": {name: float(s) for name, s in zip(CLASS_NAMES, per_class_f1)},
    "learned_zeta": float(final_zeta),
    "learned_nu": float(final_nu),
    "best_val_f1": float(best_val_f1),
    "baseline_mlp": {"accuracy": 0.8547, "macro_f1": 0.8473, "n_params": 12518},
}
Path("experiments").mkdir(exist_ok=True)
out_path = f"experiments/fastgrnn_{TAG}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
