"""
Cut 128-sample, 50%-overlapping windows out of the HAPT raw data.

Output: data/processed/hapt_windows.npz
  X        : (N, 128, 3)   raw acceleration in g
  y        : (N,)          activity ID 1-12
  subjects : (N,)          user ID 1-30
"""

import numpy as np
from pathlib import Path
from collections import Counter

DATA = Path("data/hapt")
RAW  = DATA / "RawData"
OUT  = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

WIN  = 128   # window length
STEP = 64    # stride (50% overlap)
FS   = 50    # sampling rate

# Activity ID -> name
with open(DATA / "activity_labels.txt") as f:
    ACT = {int(line.split()[0]): line.split()[1] for line in f if line.strip()}

# Load all labeled intervals: [exp, user, activity, start, end]
labels = np.loadtxt(RAW / "labels.txt", dtype=int)
print(f"Total labeled intervals: {len(labels)}")

# Cache acceleration files so we don't re-read them
acc_cache = {}
def load_acc(exp, user):
    key = (exp, user)
    if key not in acc_cache:
        acc_cache[key] = np.loadtxt(RAW / f"acc_exp{exp:02d}_user{user:02d}.txt", dtype=np.float32)
    return acc_cache[key]

X_list, y_list, s_list = [], [], []
dropped_short = 0
dropped_transition = 0

# Keep only the six basic activities (per contract.md). Transitions (7-12) are dropped.
KEEP_ACTIVITIES = {1, 2, 3, 4, 5, 6}

for row in labels:
    exp, user, aid, s, e = row
    if aid not in KEEP_ACTIVITIES:
        dropped_transition += 1
        continue
    seg = load_acc(exp, user)[s:e+1]    # +1 because UCI ranges are end-inclusive
    if len(seg) < WIN:
        dropped_short += 1
        continue
    # Sliding window
    for ws in range(0, len(seg) - WIN + 1, STEP):
        X_list.append(seg[ws:ws+WIN])
        y_list.append(aid)
        s_list.append(user)

X = np.stack(X_list)            # (N, 128, 3)
y = np.array(y_list, dtype=np.int64)
subjects = np.array(s_list, dtype=np.int64)

print(f"\n=== Output ===")
print(f"X.shape       : {X.shape}")
print(f"y.shape       : {y.shape}")
print(f"subjects.shape: {subjects.shape}")
print(f"Intervals dropped (shorter than {WIN}): {dropped_short}")
print(f"Intervals dropped (transition classes 7-12): {dropped_transition}")
print(f"X dtype       : {X.dtype}  (memory: {X.nbytes/1e6:.1f} MB)")

# Class distribution
print("\n=== Class distribution ===")
cnt = Counter(y.tolist())
total = len(y)
for aid in sorted(cnt):
    n = cnt[aid]
    bar = "#" * int(60 * n / max(cnt.values()))
    print(f"  {aid:2d} {ACT[aid]:22s} {n:6d}  ({100*n/total:5.1f}%) {bar}")

# Per-subject window count
print(f"\n=== Per-subject window counts ===")
ucnt = Counter(subjects.tolist())
print(f"  {len(ucnt)} subjects, {min(ucnt.values())}-{max(ucnt.values())} windows per subject")

# Subject-disjoint split following the HAPT convention
train_mask = subjects <= 21
test_mask  = subjects >= 22

X_tr, y_tr, s_tr = X[train_mask], y[train_mask], subjects[train_mask]
X_te, y_te, s_te = X[test_mask],  y[test_mask],  subjects[test_mask]

print(f"\n=== Subject-disjoint split (1-21 train, 22-30 test) ===")
print(f"Train: X {X_tr.shape}  ({len(set(s_tr.tolist()))} subjects)")
print(f"Test : X {X_te.shape}  ({len(set(s_te.tolist()))} subjects)")

# Save
out_path = OUT / "hapt_windows.npz"
np.savez_compressed(
    out_path,
    X_train=X_tr, y_train=y_tr, subjects_train=s_tr,
    X_test=X_te,  y_test=y_te,  subjects_test=s_te,
    activity_labels=np.array([f"{k} {v}" for k, v in sorted(ACT.items())]),
)
print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
