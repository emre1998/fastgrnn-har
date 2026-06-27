"""
Window the WISDM v1.1 raw accelerometer stream into the common npz format.

Output: data/processed/wisdm_windows.npz
  X_train/X_test : (N, WIN, 3)  raw tri-axial acceleration
  y_train/y_test : (N,)         activity ID 1..6
  subjects_*     : (N,)         user ID

Window = 2.5 s (50 samples @ 20 Hz), 50% overlap, mirroring HAPT's 2.56 s.
Subject-disjoint split (first ~70% of user IDs train, rest test).
Normalization is done downstream by the training scripts (per-channel, from train).
"""
import numpy as np
from pathlib import Path
from collections import Counter

RAW = Path("data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

WIN, STEP = 50, 25                      # 2.5 s window, 50% overlap @ 20 Hz
ACTS = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
ACT_ID = {a: i + 1 for i, a in enumerate(ACTS)}   # 1..6

# --- robust parse: split on ';' record terminator, then ',' fields ---
raw = RAW.read_text(errors="ignore").replace("\n", "")
records = []
bad = 0
for rec in raw.split(";"):
    rec = rec.strip()
    if not rec:
        continue
    parts = rec.split(",")
    if len(parts) != 6:
        bad += 1
        continue
    try:
        user = int(parts[0]); act = parts[1].strip()
        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
    except ValueError:
        bad += 1
        continue
    if act not in ACT_ID:
        bad += 1
        continue
    records.append((user, ACT_ID[act], x, y, z))

print(f"Parsed {len(records)} samples, dropped {bad} malformed records")

# --- group consecutive samples by (user, activity) into contiguous segments ---
X_list, y_list, s_list = [], [], []
seg = []
cur = None
def flush(seg, user, aid):
    if len(seg) < WIN:
        return
    arr = np.asarray(seg, dtype=np.float32)
    for ws in range(0, len(arr) - WIN + 1, STEP):
        X_list.append(arr[ws:ws + WIN])
        y_list.append(aid)
        s_list.append(user)

for user, aid, x, y, z in records:
    key = (user, aid)
    if key != cur:
        if cur is not None:
            flush(seg, cur[0], cur[1])
        seg, cur = [], key
    seg.append((x, y, z))
if cur is not None:
    flush(seg, cur[0], cur[1])

X = np.stack(X_list)
y = np.array(y_list, dtype=np.int64)
subjects = np.array(s_list, dtype=np.int64)
print(f"X {X.shape}  y {y.shape}  subjects {len(set(subjects.tolist()))} users")

print("\nClass distribution:")
cnt = Counter(y.tolist())
for aid in sorted(cnt):
    print(f"  {aid} {ACTS[aid-1]:11s} {cnt[aid]:6d} ({100*cnt[aid]/len(y):4.1f}%)")

# --- subject-disjoint split: first ~70% of sorted user IDs -> train ---
users = sorted(set(subjects.tolist()))
n_train = round(len(users) * 0.7)
train_users = set(users[:n_train])
train_mask = np.array([s in train_users for s in subjects])
test_mask = ~train_mask

X_tr, y_tr, s_tr = X[train_mask], y[train_mask], subjects[train_mask]
X_te, y_te, s_te = X[test_mask], y[test_mask], subjects[test_mask]
print(f"\nSplit: train {len(train_users)} users {X_tr.shape} | "
      f"test {len(users)-len(train_users)} users {X_te.shape}")

out = OUT / "wisdm_windows.npz"
np.savez_compressed(
    out,
    X_train=X_tr, y_train=y_tr, subjects_train=s_tr,
    X_test=X_te, y_test=y_te, subjects_test=s_te,
    activity_labels=np.array([f"{i+1} {a}" for i, a in enumerate(ACTS)]),
)
print(f"Saved: {out}  ({out.stat().st_size/1e6:.1f} MB)")
