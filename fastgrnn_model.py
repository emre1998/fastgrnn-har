"""
FastGRNN — PyTorch implementation.

Batch + autograd adaptation of the NumPy reference (fastgrnn_numpy.py).

Classes:
  FastGRNNCell                   — vanilla single-step cell
  LowRankFastGRNNCell            — W = W1 @ W2.T, U = U1 @ U2.T (Week 5)
  SparseLowRankFastGRNNCell      — low-rank + magnitude pruning masks (Week 6)
  FastGRNNClassifier             — T-step rollout + linear classifier head;
                                    supports all r_w/r_u + sparse combinations
"""

import torch
import torch.nn as nn


class FastGRNNCell(nn.Module):
    """
    FastGRNN cell — shared W, U + gate z_t + candidate h_tilde + zeta/nu mix.

    Forward inputs (batched):
        x_t:    (B, D)   current input
        h_prev: (B, H)   previous hidden state
    Returns:
        h_t:    (B, H)   new hidden state
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Learnable parameters — all nn.Parameter, automatically picked up by the optimizer.
        # W: input -> hidden. Small random init; std=1/sqrt(D) is a typical choice.
        self.W = nn.Parameter(torch.randn(hidden_size, input_size) / (input_size ** 0.5))
        # U: hidden -> hidden. Same scheme.
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size) / (hidden_size ** 0.5))
        # Biases — start at zero.
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # zeta and nu — CONSTRAINED parameters.
        # The raw value lives in (-inf, +inf); sigmoid pushes it into (0, 1).
        # zeta_raw = 0 -> sigmoid = 0.5, so initially zeta = nu = 0.5,
        # matching the NumPy reference exactly.
        self.zeta_raw = nn.Parameter(torch.tensor(0.0))
        self.nu_raw   = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Shapes:
        #   x_t:    (B, D)
        #   h_prev: (B, H)
        #   W:      (H, D)     -> W.T: (D, H)
        #   U:      (H, H)     -> U.T: (H, H)

        # Shared pre-activation. x_t @ W.T -> (B, H), h_prev @ U.T -> (B, H)
        pre = x_t @ self.W.T + h_prev @ self.U.T              # (B, H)

        # Gate: per-cell (0, 1) "keep the old value" ratio
        z_t = torch.sigmoid(pre + self.b_z)                   # (B, H)

        # Candidate new memory: signed value in (-1, 1)
        h_tilde = torch.tanh(pre + self.b_h)                  # (B, H)

        # zeta, nu in (0, 1) — the stability constraint from the paper
        zeta = torch.sigmoid(self.zeta_raw)
        nu   = torch.sigmoid(self.nu_raw)

        # Combine — bit-identical to the NumPy reference equation
        h_t = (zeta * (1.0 - z_t) + nu) * h_tilde + z_t * h_prev    # (B, H)

        return h_t


class LowRankFastGRNNCell(nn.Module):
    """
    Low-rank variant of the FastGRNN cell.
    W = W1 @ W2.T  (W1: H x r_w, W2: D x r_w)
    U = U1 @ U2.T  (U1: H x r_u, U2: H x r_u)

    In forward we DISTRIBUTE the multiplication — small first, then large:
      x_t @ W.T  =  (x_t @ W2) @ W1.T
      h    @ U.T  =  (h    @ U2) @ U1.T
    This saves both Flash and CPU.
    """

    def __init__(self, input_size: int, hidden_size: int, r_w: int, r_u: int):
        super().__init__()
        assert 1 <= r_w <= min(hidden_size, input_size), \
            f"r_w={r_w} invalid; must have 1 <= r_w <= min(H, D)={min(hidden_size, input_size)}"
        assert 1 <= r_u <= hidden_size, \
            f"r_u={r_u} invalid; must have 1 <= r_u <= H={hidden_size}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r_w = r_w
        self.r_u = r_u

        # W factors. Init: divide by sqrt(D) so the magnitude stays close to the vanilla case.
        self.W1 = nn.Parameter(torch.randn(hidden_size, r_w) / (input_size ** 0.5))
        self.W2 = nn.Parameter(torch.randn(input_size,  r_w) / (input_size ** 0.5))

        # U factors.
        self.U1 = nn.Parameter(torch.randn(hidden_size, r_u) / (hidden_size ** 0.5))
        self.U2 = nn.Parameter(torch.randn(hidden_size, r_u) / (hidden_size ** 0.5))

        # Same as vanilla: biases and constrained zeta/nu.
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.zeta_raw = nn.Parameter(torch.tensor(0.0))
        self.nu_raw   = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Distributed multiplication — pass through the bottleneck.
        # x_t @ W2 : (B, D) @ (D, r_w) = (B, r_w)
        # ... @ W1.T : (B, r_w) @ (r_w, H) = (B, H)
        xW = (x_t   @ self.W2) @ self.W1.T          # (B, H)
        hU = (h_prev @ self.U2) @ self.U1.T          # (B, H)

        pre = xW + hU                                 # (B, H)
        z_t = torch.sigmoid(pre + self.b_z)           # (B, H)
        h_tilde = torch.tanh(pre + self.b_h)          # (B, H)
        zeta = torch.sigmoid(self.zeta_raw)
        nu   = torch.sigmoid(self.nu_raw)
        h_t = (zeta * (1.0 - z_t) + nu) * h_tilde + z_t * h_prev    # (B, H)
        return h_t


class SparseLowRankFastGRNNCell(nn.Module):
    """
    Low-rank FastGRNN + magnitude pruning masks.
    Masks are buffers (not parameters) — they appear in state_dict but do
    not enter the optimizer. In forward: W1_effective = W1 * mask_W1 (element-wise).
    The apply_pruning(target_sparsity) method updates the masks.
    """

    def __init__(self, input_size: int, hidden_size: int, r_w: int, r_u: int):
        super().__init__()
        assert 1 <= r_w <= min(hidden_size, input_size)
        assert 1 <= r_u <= hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r_w = r_w
        self.r_u = r_u

        # Parameters (identical to the LowRank cell)
        self.W1 = nn.Parameter(torch.randn(hidden_size, r_w) / (input_size ** 0.5))
        self.W2 = nn.Parameter(torch.randn(input_size,  r_w) / (input_size ** 0.5))
        self.U1 = nn.Parameter(torch.randn(hidden_size, r_u) / (hidden_size ** 0.5))
        self.U2 = nn.Parameter(torch.randn(hidden_size, r_u) / (hidden_size ** 0.5))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.zeta_raw = nn.Parameter(torch.tensor(0.0))
        self.nu_raw   = nn.Parameter(torch.tensor(0.0))

        # Masks — registered as buffers. Initially all ones (dense).
        self.register_buffer("mask_W1", torch.ones_like(self.W1))
        self.register_buffer("mask_W2", torch.ones_like(self.W2))
        self.register_buffer("mask_U1", torch.ones_like(self.U1))
        self.register_buffer("mask_U2", torch.ones_like(self.U2))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Masked weights
        W1 = self.W1 * self.mask_W1
        W2 = self.W2 * self.mask_W2
        U1 = self.U1 * self.mask_U1
        U2 = self.U2 * self.mask_U2

        xW = (x_t   @ W2) @ W1.T                    # (B, H)
        hU = (h_prev @ U2) @ U1.T                    # (B, H)
        pre = xW + hU
        z_t = torch.sigmoid(pre + self.b_z)
        h_tilde = torch.tanh(pre + self.b_h)
        zeta = torch.sigmoid(self.zeta_raw)
        nu   = torch.sigmoid(self.nu_raw)
        h_t = (zeta * (1.0 - z_t) + nu) * h_tilde + z_t * h_prev
        return h_t

    @torch.no_grad()
    def apply_pruning(self, target_sparsity: float):
        """
        Per-tensor magnitude pruning.
        target_sparsity in [0, 1]: the fraction of entries to zero out.
        The smallest-magnitude weights are forced to zero.
        """
        for name in ("W1", "W2", "U1", "U2"):
            W = getattr(self, name)
            mask = getattr(self, f"mask_{name}")
            n_total = W.numel()
            n_zero  = int(n_total * target_sparsity)
            if n_zero == 0:
                mask.fill_(1.0)
                continue
            # Find the smallest |W| values
            flat = W.abs().flatten()
            threshold = torch.kthvalue(flat, n_zero).values
            new_mask = (W.abs() > threshold).float()
            mask.copy_(new_mask)
        # Multiply the parameters by the mask once so the next forward pass is consistent.
        self.W1.data.mul_(self.mask_W1)
        self.W2.data.mul_(self.mask_W2)
        self.U1.data.mul_(self.mask_U1)
        self.U2.data.mul_(self.mask_U2)

    def current_sparsity(self) -> dict:
        """Per-tensor actual sparsity ratio."""
        result = {}
        for name in ("W1", "W2", "U1", "U2"):
            mask = getattr(self, f"mask_{name}")
            result[name] = 1.0 - mask.mean().item()
        return result

    def effective_params(self) -> int:
        """Nonzero parameter count (cell, including biases and zeta/nu)."""
        n = 0
        for name in ("W1", "W2", "U1", "U2"):
            n += int(getattr(self, f"mask_{name}").sum().item())
        n += self.b_z.numel() + self.b_h.numel() + 2  # zeta + nu
        return n


class FastGRNNClassifier(nn.Module):
    """
    Takes a full window (B, T, D), unrolls a FastGRNNCell (or its low-rank
    variant) for T steps, and produces class logits from the final hidden
    state.

    Pass r_w and r_u to use LowRankFastGRNNCell; omit both to use the vanilla
    FastGRNNCell.

    Forward:
        X: (B, T, D)
    Returns:
        logits: (B, num_classes)  — raw logits, no softmax
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 r_w: int = None, r_u: int = None, sparse: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.sparse = sparse
        if r_w is None and r_u is None:
            if sparse:
                raise ValueError("sparse=True requires a low-rank cell (pass r_w and r_u).")
            self.cell = FastGRNNCell(input_size, hidden_size)
        elif r_w is not None and r_u is not None:
            if sparse:
                self.cell = SparseLowRankFastGRNNCell(input_size, hidden_size, r_w, r_u)
            else:
                self.cell = LowRankFastGRNNCell(input_size, hidden_size, r_w, r_u)
        else:
            raise ValueError("Pass r_w and r_u together, or neither.")
        # Classifier head: projection from the final hidden state to class scores.
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, D = X.shape

        # Initial hidden state: (B, H) zeros.
        # We borrow X's dtype/device for compatibility.
        h = torch.zeros(B, self.hidden_size, dtype=X.dtype, device=X.device)

        # T-step rollout — Python loop is slow but explicit; the Arduino
        # streaming inference path mirrors this loop exactly.
        for t in range(T):
            h = self.cell(X[:, t, :], h)        # (B, D) -> (B, H)

        # We only use the final h — a summary of the entire sequence.
        logits = self.classifier(h)             # (B, H) -> (B, num_classes)
        return logits
