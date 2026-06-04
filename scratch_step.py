"""
Full fastgrnn_step sanity check (after Part 3c).
1) Single step: is h_t produced and is its shape correct?
2) Many steps: over T=500 steps, does |h| explode or vanish? -> "stable" claim.
"""

import numpy as np
from fastgrnn_numpy import fastgrnn_step

np.random.seed(0)

D, H = 3, 8

W   = np.random.randn(H, D) * 0.1
U   = np.random.randn(H, H) * 0.1
b_z = np.zeros(H)
b_h = np.zeros(H)
zeta, nu = 0.5, 0.5

# --- SINGLE STEP ---
x_t    = np.array([0.2, -0.5, 9.8])
h_prev = np.zeros(H)
h_t = fastgrnn_step(x_t, h_prev, W, U, b_z, b_h, zeta, nu)

print("=== Single step ===")
print("h_t.shape :", h_t.shape, " (expected: (8,))")
print("h_t       :", h_t)
print("h_t in (-1, 1)?", np.all((h_t > -1) & (h_t < 1)))

# --- MANY STEPS: stability check ---
print("\n=== Stability check (T=500 steps) ===")
T = 500
h = np.zeros(H)
norms = []
for t in range(T):
    x = np.random.randn(D) * 0.5    # random input
    h = fastgrnn_step(x, h, W, U, b_z, b_h, zeta, nu)
    norms.append(np.linalg.norm(h))

print(f"First 5 norms : {[f'{n:.3f}' for n in norms[:5]]}")
print(f"Last 5 norms  : {[f'{n:.3f}' for n in norms[-5:]]}")
print(f"Min norm      : {min(norms):.3f}")
print(f"Max norm      : {max(norms):.3f}")
print(f"Mean          : {np.mean(norms):.3f}")
print(f"|h| ever exceed 10?   -> {any(n > 10 for n in norms)}  (no explosion = False)")
print(f"|h| ever drop below 0.01? -> {any(n < 0.01 for n in norms)}  (no vanish = False)")
