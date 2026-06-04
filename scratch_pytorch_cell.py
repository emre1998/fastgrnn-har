"""
FastGRNNCell sanity test.
1) Are the parameters registered correctly (does the optimizer see them)?
2) Are zeta and nu initialized to 0.5? (zeta_raw = 0 -> sigmoid = 0.5)
3) Do the batched forward shapes line up?
4) Over T=128 steps, does |h| explode or vanish?
"""

import torch
from fastgrnn_model import FastGRNNCell

torch.manual_seed(0)

B, T, D, H = 4, 128, 3, 8       # batch = 4, 128 steps, 3 axes, 8 hidden units

cell = FastGRNNCell(input_size=D, hidden_size=H)

# --- 1) Parameter check ---
print("=== 1) Parameters ===")
total = 0
for name, p in cell.named_parameters():
    n = p.numel()
    total += n
    print(f"  {name:10s}  shape {tuple(p.shape)!s:15s}  {n:>4} params")
print(f"  TOTAL: {total} parameters")

# --- 2) Initial zeta, nu values ---
print("\n=== 2) zeta, nu at init ===")
with torch.no_grad():
    zeta = torch.sigmoid(cell.zeta_raw).item()
    nu   = torch.sigmoid(cell.nu_raw).item()
print(f"  zeta = sigmoid(zeta_raw=0) = {zeta:.4f}  (expected: 0.5)")
print(f"  nu   = sigmoid(nu_raw=0)   = {nu:.4f}    (expected: 0.5)")

# --- 3) Batched forward ---
print("\n=== 3) Batched forward ===")
x_t    = torch.randn(B, D)
h_prev = torch.zeros(B, H)
h_t = cell(x_t, h_prev)
print(f"  x_t.shape    = {tuple(x_t.shape)}")
print(f"  h_prev.shape = {tuple(h_prev.shape)}")
print(f"  h_t.shape    = {tuple(h_t.shape)}  (expected: (4, 8))")
print(f"  h_t entirely in (-1, 1)? {((h_t > -1) & (h_t < 1)).all().item()}")

# --- 4) Stability (T=128) ---
print("\n=== 4) Stability check (T=128, raw random input) ===")
h = torch.zeros(B, H)
norms = []
with torch.no_grad():
    for t in range(T):
        x = torch.randn(B, D) * 0.5         # small-scale input
        h = cell(x, h)
        norms.append(h.norm(dim=1).mean().item())

print(f"  First 5 mean |h|: {[f'{n:.3f}' for n in norms[:5]]}")
print(f"  Last 5 mean |h| : {[f'{n:.3f}' for n in norms[-5:]]}")
print(f"  min: {min(norms):.3f}  max: {max(norms):.3f}  mean: {sum(norms)/len(norms):.3f}")
print(f"  |h| ever exceed 10? {any(n > 10 for n in norms)} (expected: False)")
print(f"  |h| ever drop below 0.01? {any(n < 0.01 for n in norms)} (expected: False)")

print("\nNote: the NumPy test's h[0]=27 issue does not appear here because")
print("the input has no time-constant gravity bias - we are feeding a small random")
print("signal. During real training on HAPT data we normalize (subtract gravity),")
print("so the problem will not appear there either.")
