"""
FastGRNNClassifier sanity test.
1) Parameter count (cell + linear head)
2) Forward (B, T, D) -> (B, num_classes)
3) Does backward propagate? (autograd graph through the recurrent loop)
"""

import torch
from fastgrnn_model import FastGRNNClassifier

torch.manual_seed(0)

B, T, D, H, C = 64, 128, 3, 8, 6

model = FastGRNNClassifier(input_size=D, hidden_size=H, num_classes=C)

# --- 1) Parameter check ---
print("=== 1) Parameters ===")
total = 0
for name, p in model.named_parameters():
    n = p.numel()
    total += n
    print(f"  {name:25s}  shape {tuple(p.shape)!s:18s}  {n:>5} params")
print(f"  TOTAL: {total} parameters  (MLP baseline was 12,518)")

# --- 2) Forward ---
print("\n=== 2) Forward ===")
X = torch.randn(B, T, D)
logits = model(X)
print(f"  X.shape      = {tuple(X.shape)}")
print(f"  logits.shape = {tuple(logits.shape)}  (expected: (64, 6))")
print(f"  Any NaN?  {torch.isnan(logits).any().item()}")
print(f"  Any Inf?  {torch.isinf(logits).any().item()}")
print(f"  logits min/max: {logits.min().item():.3f} / {logits.max().item():.3f}")

# --- 3) Backward graph ---
print("\n=== 3) Backward graph ===")
labels = torch.randint(0, C, (B,))
loss = torch.nn.functional.cross_entropy(logits, labels)
print(f"  Random loss: {loss.item():.4f}  (expected: ~log(6) = {torch.log(torch.tensor(6.0)).item():.4f})")
loss.backward()

# Did every parameter receive a gradient?
print("  Parameter gradients:")
for name, p in model.named_parameters():
    g = p.grad
    if g is None:
        print(f"    {name:25s}  NO GRAD (problem!)")
    else:
        print(f"    {name:25s}  grad norm = {g.norm().item():.6f}")
