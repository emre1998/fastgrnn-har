"""
Week 1 closer: tiny MLP baseline.

Goal: produce the "number to beat". FastGRNN should outperform this.

Architecture: flatten(128*3=384) -> Linear(384, 32) -> ReLU -> Linear(32, 6)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------------
# 1) Load data
# ----------------------------------------------------------------------------
data = np.load("data/processed/hapt_windows.npz", allow_pickle=True)
X_tr, y_tr, s_tr = data["X_train"], data["y_train"], data["subjects_train"]
X_te, y_te, s_te = data["X_test"],  data["y_test"],  data["subjects_test"]
print(f"Train: {X_tr.shape}  Test: {X_te.shape}")

# Remap labels 1-6 -> 0-5 (CrossEntropy expects zero-indexed targets)
y_tr = y_tr - 1
y_te = y_te - 1
NUM_CLASSES = 6
CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

# ----------------------------------------------------------------------------
# 2) Subject-aware validation: hold out the last four training subjects
# ----------------------------------------------------------------------------
unique_subjects = sorted(set(s_tr.tolist()))
val_subjects = set(unique_subjects[-4:])
val_mask  = np.array([s in val_subjects for s in s_tr])
trn_mask  = ~val_mask
X_trn, y_trn = X_tr[trn_mask], y_tr[trn_mask]
X_val, y_val = X_tr[val_mask], y_tr[val_mask]
print(f"Train      : {X_trn.shape}  ({len(unique_subjects)-4} subjects)")
print(f"Validation : {X_val.shape}  ({len(val_subjects)} subjects: {sorted(val_subjects)})")

# ----------------------------------------------------------------------------
# 3) Normalize: per-channel z-score with TRAIN statistics
# ----------------------------------------------------------------------------
mean = X_trn.mean(axis=(0, 1))       # shape (3,)
std  = X_trn.std(axis=(0, 1)) + 1e-8
print(f"Train mean (ax, ay, az): {mean}")
print(f"Train std  (ax, ay, az): {std}")

def normalize(X):
    return (X - mean) / std

X_trn_n = normalize(X_trn).astype(np.float32)
X_val_n = normalize(X_val).astype(np.float32)
X_te_n  = normalize(X_te ).astype(np.float32)

# ----------------------------------------------------------------------------
# 4) PyTorch Dataset / DataLoader
# ----------------------------------------------------------------------------
def make_loader(X, y, batch_size=64, shuffle=False):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
        batch_size=batch_size, shuffle=shuffle,
    )

train_loader = make_loader(X_trn_n, y_trn, batch_size=64, shuffle=True)
val_loader   = make_loader(X_val_n, y_val, batch_size=256)
test_loader  = make_loader(X_te_n,  y_te,  batch_size=256)

# ----------------------------------------------------------------------------
# 5) Model: flatten -> 384 -> 32 -> 6
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=384, hidden=32, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                     # (B, 128, 3) -> (B, 384)
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),   # raw logits, no softmax
        )
    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=128*3, hidden=32, num_classes=NUM_CLASSES)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameter count: {n_params}  ({n_params*4/1024:.1f} KB as float32)")

# ----------------------------------------------------------------------------
# 6) Training loop
# ----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 30

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

print(f"\n{'epoch':>5} {'train_loss':>11} {'val_loss':>9} {'val_acc':>8} {'val_f1':>8}")
best_val_f1 = -1
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    train_loss = total_loss / n

    val_loss, val_acc, val_f1, _, _ = evaluate(val_loader)
    flag = ""
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "mlp_baseline_best.pt")
        flag = "  *best*"
    print(f"{epoch:5d} {train_loss:11.4f} {val_loss:9.4f} {val_acc:8.4f} {val_f1:8.4f}{flag}")

# ----------------------------------------------------------------------------
# 7) Test-set evaluation with the best model
# ----------------------------------------------------------------------------
model.load_state_dict(torch.load("mlp_baseline_best.pt"))
te_loss, te_acc, te_f1, y_true, y_pred = evaluate(test_loader)

print(f"\n=== TEST SET (HAPT subjects 22-30, untouched) ===")
print(f"Loss      : {te_loss:.4f}")
print(f"Accuracy  : {te_acc:.4f}")
print(f"Macro-F1  : {te_f1:.4f}")

print("\n=== Per-class report ===")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

print("=== Confusion matrix (row = ground truth, column = prediction) ===")
cm = confusion_matrix(y_true, y_pred)
header = "       " + " ".join(f"{n[:5]:>6}" for n in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    row = " ".join(f"{cm[i, j]:>6d}" for j in range(NUM_CLASSES))
    print(f"{name[:6]:6s} {row}")

# Save results
results = {
    "model": "MLP",
    "n_params": int(n_params),
    "test_accuracy": float(te_acc),
    "test_macro_f1": float(te_f1),
    "test_loss": float(te_loss),
}
import json
Path("experiments").mkdir(exist_ok=True)
with open("experiments/mlp_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: experiments/mlp_baseline.json")
