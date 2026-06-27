"""
Window the PAMAP2 hand-IMU accelerometer into the common npz format.

Output: data/processed/pamap2_windows.npz
  X_train/X_test : (N, WIN, 3)  raw tri-axial acceleration (hand IMU, +-16g)
  y_train/y_test : (N,)         activity ID 1..K (remapped, contiguous)
  subjects_*     : (N,)         subject ID

PAMAP2 is 100 Hz; we downsample by 2 -> 50 Hz to match HAPT, then take
2.56 s windows (128 samples @ 50 Hz), 50% overlap. Hand 3D-accel +-16g =
columns 4,5,6 (0-based). Activity 0 (transient) and NaN rows dropped.
Subject 109 (incomplete) dropped. Subject-disjoint split: 101-106 train,
107-108 test. Normalization is done downstream by the training scripts.
"""
import numpy as np
from pathlib import Path
from collections import Counter

PROTO = Path("data/pamap2/PAMAP2_Dataset/PAMAP2_Dataset/Protocol")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

DS = 2                 # downsample 100 Hz -> 50 Hz
WIN, STEP = 128, 64    # 2.56 s, 50% overlap @ 50 Hz
ACC_COLS = [4, 5, 6]   # hand IMU 3D-accel +-16g
ACT_COL = 1

# Standard 12-activity protocol subset, with names.
PROTO_ACTS = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
              6: "cycling", 7: "Nordic_walking", 12: "ascending_stairs",
              13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
              24: "rope_jumping"}

TRAIN_SUBJ = {101, 102, 103, 104, 105, 106}
TEST_SUBJ = {107, 108}


def load_subject(path):
    arr = np.loadtxt(path, dtype=np.float32)
    sid = int(path.stem.replace("subject", ""))
    act = arr[:, ACT_COL].astype(int)
    acc = arr[:, ACC_COLS]
    return sid, act, acc


def windows_from(sid, act, acc, keep_ids):
    """Segment by contiguous activity, downsample, window."""
    Xs, ys, ss = [], [], []
    n = len(act)
    i = 0
    while i < n:
        a = act[i]
        j = i
        while j < n and act[j] == a:
            j += 1
        if a in keep_ids:
            seg = acc[i:j:DS]                      # downsample
            # drop windows containing NaN
            for ws in range(0, len(seg) - WIN + 1, STEP):
                w = seg[ws:ws + WIN]
                if not np.isnan(w).any():
                    Xs.append(w); ys.append(a); ss.append(sid)
        i = j
    return Xs, ys, ss


files = sorted(PROTO.glob("subject10[1-8].dat"))   # drop 109
print(f"Subjects: {[f.stem for f in files]}")

# First pass: which protocol activities actually appear in BOTH splits
present_train, present_test = set(), set()
cache = {}
for f in files:
    sid, act, acc = load_subject(f)
    cache[sid] = (act, acc)
    ids = set(np.unique(act).tolist()) & set(PROTO_ACTS)
    if sid in TRAIN_SUBJ:
        present_train |= ids
    elif sid in TEST_SUBJ:
        present_test |= ids
keep_ids = sorted(present_train & present_test)
print(f"Activities kept (present in both splits): "
      f"{[PROTO_ACTS[k] for k in keep_ids]}")

remap = {old: i + 1 for i, old in enumerate(keep_ids)}   # contiguous 1..K
names = [PROTO_ACTS[k] for k in keep_ids]

Xtr, ytr, str_, Xte, yte, ste = [], [], [], [], [], []
for sid, (act, acc) in cache.items():
    Xs, ys, ss = windows_from(sid, act, acc, set(keep_ids))
    ys = [remap[a] for a in ys]
    if sid in TRAIN_SUBJ:
        Xtr += Xs; ytr += ys; str_ += ss
    elif sid in TEST_SUBJ:
        Xte += Xs; yte += ys; ste += ss

X_tr = np.stack(Xtr); y_tr = np.array(ytr, dtype=np.int64); s_tr = np.array(str_, dtype=np.int64)
X_te = np.stack(Xte); y_te = np.array(yte, dtype=np.int64); s_te = np.array(ste, dtype=np.int64)

print(f"\nTrain X {X_tr.shape}  ({len(set(s_tr.tolist()))} subj)")
print(f"Test  X {X_te.shape}  ({len(set(s_te.tolist()))} subj)")
print("\nClass distribution (train):")
cnt = Counter(y_tr.tolist())
for k in sorted(cnt):
    print(f"  {k} {names[k-1]:18s} {cnt[k]:5d} ({100*cnt[k]/len(y_tr):4.1f}%)")

out = OUT / "pamap2_windows.npz"
np.savez_compressed(
    out,
    X_train=X_tr, y_train=y_tr, subjects_train=s_tr,
    X_test=X_te, y_test=y_te, subjects_test=s_te,
    activity_labels=np.array([f"{i+1} {n}" for i, n in enumerate(names)]),
)
print(f"\nSaved: {out}  ({out.stat().st_size/1e6:.1f} MB)  NUM_CLASSES={len(keep_ids)}")
