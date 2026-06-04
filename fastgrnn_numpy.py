"""
FastGRNN - pure-NumPy forward pass (no training, just the inference math).

The implementation was built up step by step:
  - Part 2:  sigmoid
  - Part 3:  fastgrnn_step (single time step)
  - Part 4:  run_sequence (unroll a length-T sequence)
"""

import numpy as np


def sigmoid(x):
    """
    Element-wise logistic squashing into (0, 1).
    x: scalar or NumPy array (any shape)
    Returns: array with the same shape as x; every element in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


def fastgrnn_step(x_t, h_prev, W, U, b_z, b_h, zeta, nu):
    """
    A single complete time step of the FastGRNN cell.

    Equations:
      pre     = W @ x_t + U @ h_prev
      z_t     = sigmoid(pre + b_z)
      h_tilde = tanh(pre + b_h)
      h_t     = (zeta*(1 - z_t) + nu) * h_tilde + z_t * h_prev

    Shapes:
      x_t:    (D,)
      h_prev: (H,)
      W:      (H, D)
      U:      (H, H)
      b_z:    (H,)
      b_h:    (H,)
      zeta, nu: scalars in [0, 1]
    Returns:
      h_t:    (H,)  new hidden state
    """
    # Shared pre-activation - compute once, use twice
    pre = W @ x_t + U @ h_prev                              # (H,)

    # Gate: per-cell (0, 1) "keep the old value" ratio
    z_t = sigmoid(pre + b_z)                                # (H,)

    # Candidate new memory: signed value in (-1, 1)
    h_tilde = np.tanh(pre + b_h)                            # (H,)

    # Combine: zeta and nu control "how much new" we mix in.
    # (1 - z_t) is the classic GRU behavior; zeta scales it,
    # nu adds a constant baseline.
    h_t = (zeta * (1.0 - z_t) + nu) * h_tilde + z_t * h_prev  # (H,)

    return h_t


def run_sequence(X, h0, W, U, b_z, b_h, zeta, nu, return_all=False):
    """
    Run a length-T sequence through the FastGRNN cell.

    X:  (T, D)  - T time steps, each a D-dimensional input
    h0: (H,)    - initial hidden state (usually zero)
    return_all:
      False -> return only h_final (sufficient for classification)
      True  -> return every per-step h (useful for analysis)

    Returns:
      h_final:   (H,)
      all_h (optional): (T, H)
    """
    T = X.shape[0]
    H = h0.shape[0]
    h = h0
    if return_all:
        all_h = np.zeros((T, H))
    for t in range(T):
        h = fastgrnn_step(X[t], h, W, U, b_z, b_h, zeta, nu)
        if return_all:
            all_h[t] = h
    if return_all:
        return h, all_h
    return h
