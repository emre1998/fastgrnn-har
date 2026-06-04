"""
Part 4 test: run_sequence - the entire (T, D) sequence is processed in one call.

Bonus: do two different signal patterns ("still" and "walk") produce
different h_final? The model is not trained yet; the weights are random.
We are only showing that run_sequence works and that different inputs
do produce different summary vectors.
"""

import numpy as np
from fastgrnn_numpy import run_sequence

np.random.seed(0)

D, H, T = 3, 8, 128   # our target: 128 samples, 3 axes, 8 hidden units

# (Fixed, random) model parameters
W   = np.random.randn(H, D) * 0.1
U   = np.random.randn(H, H) * 0.1
b_z = np.zeros(H)
b_h = np.zeros(H)
zeta, nu = 0.5, 0.5
h0  = np.zeros(H)

# --- Signal 1: "still" - barely moving, just gravity at ~9.8 g on z ---
X_still = np.zeros((T, D))
X_still[:, 2] = 9.8                          # z axis = gravity
X_still += np.random.randn(T, D) * 0.05      # a tiny bit of noise

# --- Signal 2: "walk" - ~2 Hz oscillation on z axis (step rhythm) ---
X_walk = np.zeros((T, D))
fs = 50                                       # 50 Hz sampling
t_axis = np.arange(T) / fs
X_walk[:, 2] = 9.8 + 2.0 * np.sin(2 * np.pi * 2.0 * t_axis)   # step rhythm
X_walk[:, 0] = 0.5 * np.sin(2 * np.pi * 2.0 * t_axis + 1.0)   # lateral sway
X_walk += np.random.randn(T, D) * 0.1

# Run
h_still = run_sequence(X_still, h0, W, U, b_z, b_h, zeta, nu)
h_walk  = run_sequence(X_walk,  h0, W, U, b_z, b_h, zeta, nu)

print("=== Shapes ===")
print("X_still.shape :", X_still.shape, " (expected: (128, 3))")
print("h_still.shape :", h_still.shape, " (expected: (8,))")

print("\n=== h_final values ===")
print("h_still :", np.round(h_still, 3))
print("h_walk  :", np.round(h_walk,  3))

# How different are the two summaries? (cosine similarity)
cos_sim = np.dot(h_still, h_walk) / (np.linalg.norm(h_still) * np.linalg.norm(h_walk) + 1e-9)
diff = np.linalg.norm(h_still - h_walk)

print(f"\n|h_still - h_walk|  = {diff:.3f}   (we expect it to be far from zero)")
print(f"cosine similarity   = {cos_sim:.3f}   (1.0 = same direction, 0 = orthogonal, -1 = opposite)")

print("\nDifferent inputs producing different summaries -> run_sequence works.")
print("Since we have not trained yet, these summaries are not yet 'meaningful'.")
print("After training, each activity will land in a distinct region, and a Linear")
print("layer on top will map those regions to class labels.")
