"""
Q15 fixed-point quantization simulator.

Performs Q15 rounding in PyTorch float32 so we can measure the accuracy
impact before any real deployment.
"""

import torch
import torch.nn as nn


def q15_round(x: torch.Tensor, scale: float = None):
    """
    Round a tensor onto a Q15 grid (simulation).

    Q15: signed int16 in [-32768, 32767]. Real value = int_value * scale.
    If scale is not provided we use the per-tensor max_abs / 32767.

    Returns:
      x_q:  float32 tensor, values snapped to the Q15 grid
      scale: the scale that was used (this is what gets shipped to the C side)
    """
    if scale is None:
        max_abs = x.abs().max().clamp(min=1e-12).item()
        scale = max_abs / 32767.0
    if scale == 0:
        return x.clone(), 0.0
    q_int = torch.round(x / scale).clamp(-32768, 32767)
    x_q = q_int * scale
    return x_q, scale


@torch.no_grad()
def quantize_weights(model, verbose: bool = True):
    """
    Round every weight in the model onto the Q15 grid (in-place).
    If the cell is sparse the mask is applied first.
    zeta_raw / nu_raw scalars are left alone (the sigmoid already
    bounds them inside (0, 1)).

    Returns: scale dict — the per-tensor scale used (needed for deployment).
    """
    scales = {}
    cell = model.cell

    # Cell weights: low-rank or sparse
    for name in ("W1", "W2", "U1", "U2"):
        if not hasattr(cell, name):
            continue
        W = getattr(cell, name)
        # If sparse, apply the mask first
        mask_name = f"mask_{name}"
        if hasattr(cell, mask_name):
            W.mul_(getattr(cell, mask_name))
        x_q, sc = q15_round(W.data)
        W.data.copy_(x_q)
        scales[f"cell.{name}"] = sc
        if verbose:
            print(f"  cell.{name:5s}  scale={sc:.6f}  range=[{x_q.min().item():+.4f}, {x_q.max().item():+.4f}]")

    # Vanilla cell support (W and U directly, no low-rank factorization)
    for name in ("W", "U"):
        if hasattr(cell, name) and isinstance(getattr(cell, name), nn.Parameter):
            W = getattr(cell, name)
            x_q, sc = q15_round(W.data)
            W.data.copy_(x_q)
            scales[f"cell.{name}"] = sc
            if verbose:
                print(f"  cell.{name:5s}  scale={sc:.6f}")

    # Biases
    for name in ("b_z", "b_h"):
        if hasattr(cell, name):
            b = getattr(cell, name)
            x_q, sc = q15_round(b.data)
            b.data.copy_(x_q)
            scales[f"cell.{name}"] = sc
            if verbose:
                print(f"  cell.{name:5s}  scale={sc:.6f}")

    # Classifier head
    if hasattr(model, "classifier"):
        for pname in ("weight", "bias"):
            p = getattr(model.classifier, pname)
            x_q, sc = q15_round(p.data)
            p.data.copy_(x_q)
            scales[f"classifier.{pname}"] = sc
            if verbose:
                print(f"  classifier.{pname:7s}  scale={sc:.6f}")

    return scales


def q15_activation(x: torch.Tensor, max_abs: float = 1.0) -> torch.Tensor:
    """
    Fixed-point rounding within the range [-max_abs, +max_abs).
    max_abs = 1.0 -> classic Q15 (suitable for tanh/sigmoid outputs).
    max_abs = 2.0 -> Q14 (FastGRNN h_t can exceed |1| by construction).
    Scale = max_abs / 32767.
    """
    scale = max_abs / 32767.0
    q_int = torch.round(x / scale).clamp(-32767, 32767)
    return q_int * scale


@torch.no_grad()
def calibrate_activations(model, calibration_loader, n_batches: int = 5):
    """
    Run a floating-point pass and record the max abs of every activation.
    Returns: {'z': max, 'h_tilde': max, 'h_t': max}
    These maxima drive the Q15 scale for each activation tensor.
    """
    cell = model.cell
    stats = {"z": 0.0, "h_tilde": 0.0, "h_t": 0.0}

    orig_forward = cell.forward

    def calib_forward(x_t, h_prev):
        # Masked weights (if sparse)
        if hasattr(cell, "mask_W1"):
            W1 = cell.W1 * cell.mask_W1; W2 = cell.W2 * cell.mask_W2
            U1 = cell.U1 * cell.mask_U1; U2 = cell.U2 * cell.mask_U2
            xW = (x_t @ W2) @ W1.T; hU = (h_prev @ U2) @ U1.T
        elif hasattr(cell, "W1"):
            xW = (x_t @ cell.W2) @ cell.W1.T; hU = (h_prev @ cell.U2) @ cell.U1.T
        else:
            xW = x_t @ cell.W.T; hU = h_prev @ cell.U.T
        pre = xW + hU
        z = torch.sigmoid(pre + cell.b_z)
        h_tilde = torch.tanh(pre + cell.b_h)
        zeta = torch.sigmoid(cell.zeta_raw)
        nu   = torch.sigmoid(cell.nu_raw)
        h_t = (zeta * (1.0 - z) + nu) * h_tilde + z * h_prev

        stats["z"]       = max(stats["z"],       z.abs().max().item())
        stats["h_tilde"] = max(stats["h_tilde"], h_tilde.abs().max().item())
        stats["h_t"]     = max(stats["h_t"],     h_t.abs().max().item())
        return h_t

    cell.forward = calib_forward
    model.eval()
    for i, (x, _) in enumerate(calibration_loader):
        if i >= n_batches:
            break
        _ = model(x)
    cell.forward = orig_forward
    return stats


@torch.no_grad()
def wrap_cell_with_calibrated_quantization(cell, stats: dict, headroom: float = 1.1):
    """
    Wrap the cell forward with calibrated Q15 scaling.
    headroom: inflate the scale slightly (unseen test data may exceed the
    calibration maximum).
    """
    z_max  = stats["z"]       * headroom
    ht_max = stats["h_tilde"] * headroom
    h_max  = stats["h_t"]     * headroom

    def quantized_forward(x_t, h_prev):
        h_prev = q15_activation(h_prev, max_abs=h_max)
        if hasattr(cell, "mask_W1"):
            W1 = cell.W1 * cell.mask_W1; W2 = cell.W2 * cell.mask_W2
            U1 = cell.U1 * cell.mask_U1; U2 = cell.U2 * cell.mask_U2
            xW = (x_t @ W2) @ W1.T; hU = (h_prev @ U2) @ U1.T
        elif hasattr(cell, "W1"):
            xW = (x_t @ cell.W2) @ cell.W1.T; hU = (h_prev @ cell.U2) @ cell.U1.T
        else:
            xW = x_t @ cell.W.T; hU = h_prev @ cell.U.T
        pre = xW + hU

        z_t = q15_activation(torch.sigmoid(pre + cell.b_z), max_abs=z_max)
        h_tilde = q15_activation(torch.tanh(pre + cell.b_h), max_abs=ht_max)
        zeta = torch.sigmoid(cell.zeta_raw)
        nu   = torch.sigmoid(cell.nu_raw)
        h_t = (zeta * (1.0 - z_t) + nu) * h_tilde + z_t * h_prev
        return q15_activation(h_t, max_abs=h_max)

    cell.forward = quantized_forward


@torch.no_grad()
def wrap_cell_for_activation_quantization(cell):
    """
    Monkey-patch the cell forward so that activations (z, h_tilde, h_t)
    are rounded onto a Q15 grid. Written for the sparse low-rank cell,
    but works with the vanilla cell too (W and U are used directly).
    """
    orig_forward = cell.forward

    # The h_t formula allows |h| > 1; we use Q14 (max_abs = 2).
    # z and h_tilde are tanh/sigmoid outputs, so max_abs = 1 (full Q15) is fine.
    H_MAX = 2.0
    ACT_MAX = 1.0

    def quantized_forward(x_t, h_prev):
        # Streaming inference: h_prev (the h_t output) is in the Q14 range
        h_prev = q15_activation(h_prev, max_abs=H_MAX)

        # Masked weights (if the cell is sparse)
        if hasattr(cell, "mask_W1"):
            W1 = cell.W1 * cell.mask_W1
            W2 = cell.W2 * cell.mask_W2
            U1 = cell.U1 * cell.mask_U1
            U2 = cell.U2 * cell.mask_U2
        elif hasattr(cell, "W1"):
            W1, W2, U1, U2 = cell.W1, cell.W2, cell.U1, cell.U2
        else:
            # Vanilla cell
            xW = x_t @ cell.W.T
            hU = h_prev @ cell.U.T
            pre = xW + hU
            z_t = q15_activation(torch.sigmoid(pre + cell.b_z), max_abs=ACT_MAX)
            h_tilde = q15_activation(torch.tanh(pre + cell.b_h), max_abs=ACT_MAX)
            zeta = torch.sigmoid(cell.zeta_raw)
            nu   = torch.sigmoid(cell.nu_raw)
            h_t = (zeta * (1.0 - z_t) + nu) * h_tilde + z_t * h_prev
            return q15_activation(h_t, max_abs=H_MAX)

        # Low-rank (sparse or not) — distributed multiplication
        xW = (x_t @ W2) @ W1.T
        hU = (h_prev @ U2) @ U1.T
        pre = xW + hU

        z_t = q15_activation(torch.sigmoid(pre + cell.b_z), max_abs=ACT_MAX)
        h_tilde = q15_activation(torch.tanh(pre + cell.b_h), max_abs=ACT_MAX)
        zeta = torch.sigmoid(cell.zeta_raw)
        nu   = torch.sigmoid(cell.nu_raw)
        h_t = (zeta * (1.0 - z_t) + nu) * h_tilde + z_t * h_prev
        return q15_activation(h_t, max_abs=H_MAX)

    cell.forward = quantized_forward
    cell._original_forward = orig_forward


def model_size_bytes(model, dtype_bits: int = 16) -> dict:
    """
    Compute the deployment size of the model (Flash footprint).
    For a sparse model only the nonzero weights are counted.
    dtype_bits: number of bits per weight (Q15 = 16).
    """
    total_params = 0
    nonzero_params = 0
    cell = model.cell

    for name in ("W1", "W2", "U1", "U2", "W", "U"):
        if not hasattr(cell, name):
            continue
        W = getattr(cell, name)
        if not isinstance(W, nn.Parameter):
            continue
        n = W.numel()
        total_params += n
        mask_name = f"mask_{name}"
        if hasattr(cell, mask_name):
            nonzero_params += int(getattr(cell, mask_name).sum().item())
        else:
            nonzero_params += n

    for name in ("b_z", "b_h"):
        if hasattr(cell, name):
            n = getattr(cell, name).numel()
            total_params += n
            nonzero_params += n

    # zeta + nu
    total_params += 2
    nonzero_params += 2

    if hasattr(model, "classifier"):
        for pname in ("weight", "bias"):
            n = getattr(model.classifier, pname).numel()
            total_params += n
            nonzero_params += n

    bytes_per_param = dtype_bits / 8
    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "dense_size_bytes": int(total_params * bytes_per_param),
        "sparse_size_bytes": int(nonzero_params * bytes_per_param),
        "sparse_size_kb": nonzero_params * bytes_per_param / 1024,
    }
