"""
Epoch saturation curve - H=16 fixed, 120 epochs.

Records train_loss, val_loss, val_f1 and test_f1 (for tracking only) at every
epoch.

Outputs:
  experiments/saturation_h16.json - full history
  saturation_curve.png            - the curves
  Console: saturation analysis (best epoch, where the curve flattens, ...)
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from pathlib import Path
from fastgrnn_model import FastGRNNClassifier

torch.manual_seed(0)
np.random.seed(0)

HIDDEN = 16
EPOCHS = 120
LR     = 1e-3

# --- Data (identical pipeline to mlp/fastgrnn) ---
data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te       = data["X_test"],  data["y_test"]

y_tr = y_tr - 1
y_te = y_te - 1
NUM_CLASSES = 6

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

# --- Model ---
model = FastGRNNClassifier(input_size=3, hidden_size=HIDDEN, num_classes=NUM_CLASSES)
n_params = sum(p.numel() for p in model.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
    return total_loss/total, total_correct/total, f1_score(y_true, y_pred, average="macro")

print(f"H={HIDDEN}, epochs={EPOCHS}, params={n_params}")
print(f"\n{'epoch':>5} {'train_loss':>11} {'val_loss':>9} {'val_f1':>8} {'test_f1':>8}  flag")

history = []
best_val_f1 = -1
best_epoch  = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * len(y); n += len(y)
    train_loss = total_loss / n
    val_loss, val_acc, val_f1   = evaluate(val_loader)
    _,        test_acc, test_f1 = evaluate(test_loader)
    history.append({
        "epoch": epoch, "train_loss": train_loss,
        "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        "test_acc": test_acc, "test_f1": test_f1,
    })
    flag = ""
    if val_f1 > best_val_f1:
        best_val_f1, best_epoch = val_f1, epoch
        flag = "*best-val*"
    print(f"{epoch:5d} {train_loss:11.4f} {val_loss:9.4f} {val_f1:8.4f} {test_f1:8.4f}  {flag}")

# --- Saturation analysis ---
test_f1s = [h["test_f1"] for h in history]
val_f1s  = [h["val_f1"]  for h in history]
best_test_epoch = int(np.argmax(test_f1s)) + 1
best_test_f1    = max(test_f1s)

# Find the saturation point: the first epoch where test_f1 reaches 99% of the maximum
target = 0.99 * best_test_f1
saturation_epoch = None
for h in history:
    if h["test_f1"] >= target:
        saturation_epoch = h["epoch"]
        break

# Last 30-epoch average vs the maximum
last30_avg = np.mean(test_f1s[-30:])
trend_last30 = test_f1s[-1] - test_f1s[-30]

print("\n" + "=" * 70)
print(" SATURATION ANALYSIS")
print("=" * 70)
print(f"Highest val_f1   : {best_val_f1:.4f} @ epoch {best_epoch}")
print(f"Highest test_f1  : {best_test_f1:.4f} @ epoch {best_test_epoch}")
print(f"Saturation point : epoch {saturation_epoch}  (first epoch where test_f1 reaches {target:.4f})")
print(f"Last 30-epoch test_f1 average: {last30_avg:.4f}")
print(f"Last 30-epoch trend (e120 - e90): {trend_last30:+.4f}")

# --- Save ---
Path("experiments").mkdir(exist_ok=True)
with open("experiments/saturation_h16.json", "w") as f:
    json.dump({
        "config": {"hidden": HIDDEN, "epochs": EPOCHS, "lr": LR, "n_params": n_params},
        "best_val_f1": best_val_f1, "best_val_epoch": best_epoch,
        "best_test_f1": best_test_f1, "best_test_epoch": best_test_epoch,
        "saturation_epoch": saturation_epoch,
        "history": history,
    }, f, indent=2)
print(f"\nSaved: experiments/saturation_h16.json")

# --- Plot ---
epochs = [h["epoch"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses   = [h["val_loss"]   for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_losses, label="train_loss", color="steelblue", lw=1.5)
ax1.plot(epochs, val_losses, label="val_loss", color="darkorange", lw=1.5)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title(f"Loss curve (H={HIDDEN})")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, val_f1s, label="val_f1", color="darkorange", lw=1.5)
ax2.plot(epochs, test_f1s, label="test_f1 (tracking)", color="green", lw=1.5)
ax2.axvline(best_test_epoch, color="green", linestyle="--", alpha=0.5,
            label=f"best test @ e{best_test_epoch}")
if saturation_epoch:
    ax2.axvline(saturation_epoch, color="red", linestyle=":", alpha=0.5,
                label=f"saturation @ e{saturation_epoch}")
ax2.axhline(0.8473, color="gray", linestyle="-.", alpha=0.5, label="MLP baseline (0.847)")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1")
ax2.set_title("F1 curve - val and test")
ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("saturation_curve.png", dpi=120)
print("Saved: saturation_curve.png")
