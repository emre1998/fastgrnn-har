"""
Get a feel for the HAPT data by eye.

- Load a single session (exp 01, user 01)
- Pull the activity intervals for that session out of labels.txt
- Plot the full acceleration signal, color-coded by activity
- Zoom in on one WALKING interval and overlay the 128-sample windows
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/hapt")
RAW  = DATA / "RawData"

# Activity ID -> name
with open(DATA / "activity_labels.txt") as f:
    ACT = {int(line.split()[0]): line.split()[1] for line in f if line.strip()}
print("Activities:", ACT)

# labels.txt: exp_id user_id activity_id start end
labels = np.loadtxt(RAW / "labels.txt", dtype=int)
print(f"\nTotal labeled intervals: {len(labels)}")
print(f"Columns: exp_id, user_id, activity_id, start_idx, end_idx")
print(f"First 3 rows:\n{labels[:3]}")

# Pick exp 1, user 1
EXP, USER = 1, 1
acc = np.loadtxt(RAW / f"acc_exp{EXP:02d}_user{USER:02d}.txt")
print(f"\nacc_exp{EXP:02d}_user{USER:02d}.txt shape: {acc.shape}")
print(f"Duration: {len(acc)/50:.1f} seconds (at 50 Hz)")

# Labeled intervals for this session
mask = (labels[:, 0] == EXP) & (labels[:, 1] == USER)
session_labels = labels[mask]
print(f"\nThis session has {len(session_labels)} labeled intervals.")
print("Sample intervals (first 5):")
for row in session_labels[:5]:
    aid = row[2]
    print(f"  {ACT[aid]:25s}  samples {row[3]:6d}-{row[4]:6d}  ({(row[4]-row[3])/50:.1f} s)")

# --- Plot 1: whole session, color-coded labels ---
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fs = 50
t = np.arange(len(acc)) / fs

axes[0].plot(t, acc[:, 0], label="ax", lw=0.6)
axes[0].plot(t, acc[:, 1], label="ay", lw=0.6)
axes[0].plot(t, acc[:, 2], label="az", lw=0.6)
axes[0].set_ylabel("Acceleration (g)")
axes[0].legend(loc="upper right")
axes[0].set_title(f"Session: exp{EXP:02d} user{USER:02d} - raw acceleration")

# Draw the labeled intervals on the bottom panel as bars
cmap = plt.colormaps.get_cmap("tab20")
for row in session_labels:
    aid, s, e = row[2], row[3], row[4]
    axes[1].barh(0, (e - s) / fs, left=s / fs, height=0.8,
                 color=cmap(aid % 20), edgecolor="k", linewidth=0.3)
    axes[1].text((s + e) / 2 / fs, 0, ACT[aid], ha="center", va="center", fontsize=6, rotation=90)
axes[1].set_xlabel("Time (s)")
axes[1].set_yticks([])
axes[1].set_title("Labeled activity intervals")

plt.tight_layout()
plt.savefig("explore_session.png", dpi=110)
print("\nSaved: explore_session.png")

# --- Plot 2: zoom into a WALKING interval, show the 128-sample windows ---
walking = session_labels[session_labels[:, 2] == 1]
if len(walking):
    s, e = walking[0, 3], walking[0, 4]
    seg = acc[s:e]
    t_seg = np.arange(len(seg)) / fs
    print(f"\nFirst WALKING interval: samples {s}-{e} = {len(seg)} samples = {len(seg)/fs:.1f} s")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_seg, seg[:, 0], label="ax", lw=0.8)
    ax.plot(t_seg, seg[:, 1], label="ay", lw=0.8)
    ax.plot(t_seg, seg[:, 2], label="az", lw=0.8)

    # 128-sample, 50%-overlap windows
    WIN, STEP = 128, 64
    n_windows = 0
    for ws in range(0, len(seg) - WIN + 1, STEP):
        we = ws + WIN
        ax.axvspan(ws / fs, we / fs, alpha=0.08, color="orange")
        n_windows += 1
    ax.set_title(f"WALKING interval + 128-sample (50% overlap) windows - {n_windows} total")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("explore_walking.png", dpi=110)
    print(f"Saved: explore_walking.png  ({n_windows} windows)")
else:
    print("\nNo WALKING interval in this session.")

plt.close("all")
print("\nDone. Look at the PNGs to get a feel for the signal.")
