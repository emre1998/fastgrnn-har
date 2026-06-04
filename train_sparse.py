"""
Week 6 — sparsity (the S stage of the L-S-Q pipeline).
Iterative Hard Thresholding (IHT) with a gradually increasing sparsity target.

Flow:
  1. Warm-start from an r_u=8 low-rank checkpoint
  2. Cubic schedule: sparsity 0 -> target gradually (within RAMP_EPOCHS)
  3. Fine-tune at full sparsity (remaining epochs)
  4. Save best val_f1, evaluate on the test set

Example usage:
  python train_sparse.py --target_sparsity 0.7 --epochs 100 --ramp_epochs 50 \\
      --init_ckpt fastgrnn_h16_rw2_ru8_s0_e100_best.pt
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
parser.add_argument("--r_w", type=int, default=2)
parser.add_argument("--r_u", type=int, default=8)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ramp_epochs", type=int, default=50,
                    help="Number of epochs for the sparsity ramp; full-sparsity fine-tune follows")
parser.add_argument("--target_sparsity", type=float, default=0.7,
                    help="Target sparsity ratio in [0, 1]")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--init_ckpt", type=str, default=None,
                    help="Low-rank checkpoint to warm-start from")
parser.add_argument("--best_after_ramp", action="store_true",
                    help="Track best val_f1 only after the sparsity ramp ends")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# --- Data (unchanged) ---
data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te       = data["X_test"],  data["y_test"]
y_tr = y_tr - 1
y_te = y_te - 1
NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

uniq = sorted(set(s_tr.tolist()))
val_subjects = set(uniq[-4:])
val_mask = np.array([s in val_subjects for s in s_tr])
X_trn, y_trn = X_tr[~val_mask], y_tr[~val_mask]
X_val, y_val = X_tr[ val_mask], y_tr[ val_mask]

mean = X_trn.mean(axis=(0, 1))
std  = X_trn.std(axis=(0, 1)) + 1e-8
def normalize(X): return ((X - mean) / std).astype(np.float32)
X_trn_n, X_val_n, X_te_n = normalize(X_trn), normalize(X_val), normalize(X_te)

def make_loader(X, y, batch_size=64, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                      batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_trn_n, y_trn, 64, shuffle=True)
val_loader   = make_loader(X_val_n, y_val, 256)
test_loader  = make_loader(X_te_n,  y_te,  256)

# --- Model (sparse low-rank) ---
model = FastGRNNClassifier(
    input_size=3, hidden_size=args.hidden, num_classes=NUM_CLASSES,
    r_w=args.r_w, r_u=args.r_u, sparse=True,
)
n_params_total = sum(p.numel() for p in model.parameters())
print(f"Model: SparseLowRank H={args.hidden}, r_w={args.r_w}, r_u={args.r_u}")
print(f"Total parameters (dense count): {n_params_total}")
print(f"Target sparsity: {args.target_sparsity:.0%}, ramp: {args.ramp_epochs} epochs")

# --- Warm-start: copy weights from a low-rank checkpoint ---
if args.init_ckpt and Path(args.init_ckpt).exists():
    src_state = torch.load(args.init_ckpt)
    own_state = model.state_dict()
    loaded = 0
    for k, v in src_state.items():
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k].copy_(v)
            loaded += 1
    print(f"Warm-start: loaded {loaded} tensors from {args.init_ckpt}.")
else:
    print(f"Training from scratch (no init checkpoint).")

# --- IHT cubic schedule ---
def sparsity_at(epoch, target, ramp_epochs):
    """Cubic schedule (Zhu & Gupta, 2017). epoch is 1-indexed."""
    if epoch > ramp_epochs:
        return target
    t = epoch / ramp_epochs
    return target * (1.0 - (1.0 - t) ** 3)

# --- Training ---
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
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    return total_loss/total, total_correct/total, f1_score(y_true, y_pred, average="macro"), y_true, y_pred

print(f"\n{'epoch':>5} {'sparsity':>9} {'eff.par':>8} {'train':>9} {'val_loss':>9} {'val_f1':>8}  flag")
TAG = f"sparse_h{args.hidden}_rw{args.r_w}_ru{args.r_u}_sp{int(args.target_sparsity*100)}_s{args.seed}_e{args.epochs}"
best_val_f1 = -1
best_epoch = 0
no_improve = 0
last_epoch = 0
sparsity_history = []

for epoch in range(1, args.epochs + 1):
    last_epoch = epoch
    # Sparsity update — at the start of every epoch, push toward the target.
    target_now = sparsity_at(epoch, args.target_sparsity, args.ramp_epochs)
    model.cell.apply_pruning(target_now)
    eff_params = model.cell.effective_params() + sum(p.numel() for p in model.classifier.parameters())

    model.train()
    total_loss, n = 0.0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        # Zero out gradients for masked weights
        with torch.no_grad():
            model.cell.W1.grad.mul_(model.cell.mask_W1)
            model.cell.W2.grad.mul_(model.cell.mask_W2)
            model.cell.U1.grad.mul_(model.cell.mask_U1)
            model.cell.U2.grad.mul_(model.cell.mask_U2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        # Re-apply the mask after the optimizer step (avoids numeric drift)
        with torch.no_grad():
            model.cell.W1.data.mul_(model.cell.mask_W1)
            model.cell.W2.data.mul_(model.cell.mask_W2)
            model.cell.U1.data.mul_(model.cell.mask_U1)
            model.cell.U2.data.mul_(model.cell.mask_U2)
        total_loss += loss.item() * len(y); n += len(y)
    train_loss = total_loss / n
    val_loss, val_acc, val_f1, _, _ = evaluate(val_loader)
    sparsity_history.append({"epoch": epoch, "sparsity": target_now,
                             "eff_params": eff_params,
                             "train_loss": train_loss, "val_loss": val_loss,
                             "val_acc": val_acc, "val_f1": val_f1})
    flag = ""
    track_best = (not args.best_after_ramp) or (epoch > args.ramp_epochs)
    if track_best and val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), f"{TAG}_best.pt")
        flag = "  *best*"
    elif track_best:
        no_improve += 1
    print(f"{epoch:5d} {target_now:9.3f} {eff_params:8d} {train_loss:9.4f} {val_loss:9.4f} {val_f1:8.4f}{flag}")

    if args.patience > 0 and no_improve >= args.patience and epoch > args.ramp_epochs:
        print(f"\n[EARLY STOP] val_f1 has not improved for {args.patience} epochs. Best epoch: {best_epoch}.")
        break

# --- Test ---
model.load_state_dict(torch.load(f"{TAG}_best.pt"))
te_loss, te_acc, te_f1, y_true, y_pred = evaluate(test_loader)

final_eff_params = model.cell.effective_params() + sum(p.numel() for p in model.classifier.parameters())
sparsity_actual = model.cell.current_sparsity()

print(f"\n=== TEST SET ===")
print(f"Accuracy : {te_acc:.4f}  (r_u=8 dense baseline: mean ~0.879)")
print(f"Macro F1 : {te_f1:.4f}")
print(f"Effective parameters: {final_eff_params}  (dense was 430)")
print(f"Per-tensor sparsity ratio:")
for name, s in sparsity_actual.items():
    print(f"  {name}: {s:.3f}")

print("\n=== Per-class ===")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

print("=== Confusion matrix ===")
cm = confusion_matrix(y_true, y_pred)
header = "       " + " ".join(f"{n[:5]:>6}" for n in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    row = " ".join(f"{cm[i, j]:>6d}" for j in range(NUM_CLASSES))
    print(f"{name[:6]:6s} {row}")

# Save
per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
results = {
    "model": "SparseLowRankFastGRNN",
    "hidden": args.hidden, "r_w": args.r_w, "r_u": args.r_u,
    "target_sparsity": args.target_sparsity,
    "ramp_epochs": args.ramp_epochs,
    "epochs_planned": args.epochs, "epochs_run": last_epoch,
    "best_epoch": best_epoch, "patience": args.patience,
    "seed": args.seed,
    "n_params_total": int(n_params_total),
    "effective_params": int(final_eff_params),
    "actual_sparsity": sparsity_actual,
    "test_accuracy": float(te_acc),
    "test_macro_f1": float(te_f1),
    "test_loss": float(te_loss),
    "per_class_f1": {name: float(s) for name, s in zip(CLASS_NAMES, per_class_f1)},
    "init_ckpt": args.init_ckpt,
    "sparsity_history": sparsity_history,
}
Path("experiments").mkdir(exist_ok=True)
out_path = f"experiments/{TAG}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
