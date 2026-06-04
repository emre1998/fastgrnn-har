"""
LowRankFastGRNNCell sanity test.
1) Parameter count (compare against vanilla at H=16)
2) Forward (B, T, D) -> (B, num_classes) still works
3) Backward graph - do W1, W2, U1, U2 + zeta_raw + nu_raw all get gradients?
4) Stability test - does |h| explode over 128 steps?
"""

import torch
from fastgrnn_model import FastGRNNClassifier

torch.manual_seed(0)

B, T, D, H, C = 4, 128, 3, 16, 6
r_w, r_u = 2, 4

print(f"Config: H={H}, D={D}, r_w={r_w}, r_u={r_u}")

# --- 1) Vanilla vs low-rank parameter comparison ---
vanilla = FastGRNNClassifier(input_size=D, hidden_size=H, num_classes=C)
lowrank = FastGRNNClassifier(input_size=D, hidden_size=H, num_classes=C, r_w=r_w, r_u=r_u)

n_van = sum(p.numel() for p in vanilla.parameters())
n_lor = sum(p.numel() for p in lowrank.parameters())

print(f"\n=== 1) Parameter count ===")
print(f"  Vanilla H=16:                  {n_van}")
print(f"  Low-rank H=16 r_w=2 r_u=4:     {n_lor}")
print(f"  Savings:                       {n_van - n_lor} ({100*(n_van-n_lor)/n_van:.1f}%)")
print(f"  MLP reference:                 12,518")

print(f"\nLow-rank parameter breakdown:")
total = 0
for name, p in lowrank.named_parameters():
    n = p.numel()
    total += n
    print(f"  {name:25s}  shape {tuple(p.shape)!s:18s}  {n:>4} params")

# --- 2) Forward ---
print(f"\n=== 2) Forward ===")
X = torch.randn(B, T, D)
logits = lowrank(X)
print(f"  X.shape      = {tuple(X.shape)}")
print(f"  logits.shape = {tuple(logits.shape)}  (expected: (4, 6))")
print(f"  Any NaN? {torch.isnan(logits).any().item()}, Any Inf? {torch.isinf(logits).any().item()}")
print(f"  logits min/max: {logits.min().item():.3f} / {logits.max().item():.3f}")

# --- 3) Backward ---
print(f"\n=== 3) Backward graph ===")
labels = torch.randint(0, C, (B,))
loss = torch.nn.functional.cross_entropy(logits, labels)
print(f"  Loss: {loss.item():.4f}  (expected: ~1.79 = log(6))")
loss.backward()

for name, p in lowrank.named_parameters():
    g = p.grad
    if g is None:
        print(f"    {name:25s}  NO GRAD (!)")
    else:
        print(f"    {name:25s}  grad norm = {g.norm().item():.6f}")

# --- 4) Stability test ---
print(f"\n=== 4) Stability test (T={T} steps) ===")
h = torch.zeros(B, H)
norms = []
with torch.no_grad():
    for t in range(T):
        x = torch.randn(B, D) * 0.5
        h = lowrank.cell(x, h)
        norms.append(h.norm(dim=1).mean().item())
print(f"  First 5 mean |h|: {[f'{n:.3f}' for n in norms[:5]]}")
print(f"  Last 5 mean |h| : {[f'{n:.3f}' for n in norms[-5:]]}")
print(f"  min: {min(norms):.3f}  max: {max(norms):.3f}")
print(f"  |h| ever exceed 10? {any(n > 10 for n in norms)} (expected: False)")
