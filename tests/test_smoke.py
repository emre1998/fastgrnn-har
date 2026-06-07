"""
Smoke tests - lightweight checks that run in CI without needing trained
checkpoints. Verifies:

  1. Module imports succeed (catches syntax errors, missing deps).
  2. The FastGRNN cell variants build and forward-pass with random inputs.
  3. Sparsification reduces the effective parameter count as expected.
  4. Q15 quantization round-trips a tensor through the simulator.
  5. The paper figure generator parses every committed JSON and produces
     vector PDFs.

These tests are designed to run in under 30 seconds on a GitHub-Actions
ubuntu-latest CPU runner. They do NOT verify model accuracy - that would
require running the full training pipeline (~2 hours).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make the project root importable regardless of where pytest is invoked
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------------
# 1) Module imports
# ----------------------------------------------------------------------------
def test_model_imports():
    """Catches syntax errors and missing dependencies in the model module."""
    from fastgrnn_model import (
        FastGRNNCell,
        LowRankFastGRNNCell,
        SparseLowRankFastGRNNCell,
        FastGRNNClassifier,
    )
    assert FastGRNNCell is not None
    assert LowRankFastGRNNCell is not None
    assert SparseLowRankFastGRNNCell is not None
    assert FastGRNNClassifier is not None


def test_quantize_imports():
    from quantize import (
        q15_round,
        quantize_weights,
        calibrate_activations,
        model_size_bytes,
    )
    assert callable(q15_round)
    assert callable(quantize_weights)


def test_numpy_reference_imports():
    from fastgrnn_numpy import sigmoid, fastgrnn_step, run_sequence
    # Quick numerical sanity: sigmoid(0) = 0.5
    assert abs(sigmoid(0.0) - 0.5) < 1e-6


# ----------------------------------------------------------------------------
# 2) Cell forward passes
# ----------------------------------------------------------------------------
def test_vanilla_cell_forward():
    from fastgrnn_model import FastGRNNCell

    cell = FastGRNNCell(input_size=3, hidden_size=16)
    x_t = torch.randn(4, 3)
    h_prev = torch.zeros(4, 16)
    h_t = cell(x_t, h_prev)
    assert h_t.shape == (4, 16)
    assert not torch.isnan(h_t).any()


def test_lowrank_cell_forward():
    from fastgrnn_model import LowRankFastGRNNCell

    cell = LowRankFastGRNNCell(input_size=3, hidden_size=16, r_w=2, r_u=8)
    x_t = torch.randn(4, 3)
    h_prev = torch.zeros(4, 16)
    h_t = cell(x_t, h_prev)
    assert h_t.shape == (4, 16)


def test_classifier_forward():
    from fastgrnn_model import FastGRNNClassifier

    model = FastGRNNClassifier(
        input_size=3, hidden_size=16, num_classes=6,
        r_w=2, r_u=8, sparse=True,
    )
    X = torch.randn(2, 128, 3)
    logits = model(X)
    assert logits.shape == (2, 6)
    assert not torch.isnan(logits).any()


# ----------------------------------------------------------------------------
# 3) Sparsification
# ----------------------------------------------------------------------------
def test_sparsification_reduces_nonzero_count():
    from fastgrnn_model import SparseLowRankFastGRNNCell

    cell = SparseLowRankFastGRNNCell(input_size=3, hidden_size=16, r_w=2, r_u=8)
    full = cell.effective_params()

    cell.apply_pruning(target_sparsity=0.5)
    sparse = cell.effective_params()

    assert sparse < full, "Pruning should reduce the effective parameter count"
    # Sparsity ratios per masked tensor should hover around 0.5
    ratios = cell.current_sparsity()
    for name, ratio in ratios.items():
        assert 0.4 < ratio < 0.6, f"{name} ratio {ratio} out of expected band"


# ----------------------------------------------------------------------------
# 4) Q15 round-trip
# ----------------------------------------------------------------------------
def test_q15_round_trip():
    from quantize import q15_round

    x = torch.randn(64)
    x_q, scale = q15_round(x)
    assert scale > 0
    # The rounded tensor should be close to the original
    assert (x - x_q).abs().max() < 2 * scale
    # And values should land on a discrete grid of multiples of `scale`
    q_int = (x_q / scale).round()
    assert (q_int.abs() <= 32767).all()


# ----------------------------------------------------------------------------
# 5) Paper figures - JSON parsing + matplotlib smoke test
# ----------------------------------------------------------------------------
def test_experiment_jsons_parse():
    """Every JSON in experiments/ should be valid JSON."""
    exp_dir = ROOT / "experiments"
    assert exp_dir.exists(), "experiments/ directory missing"
    jsons = list(exp_dir.glob("*.json"))
    assert len(jsons) >= 20, f"Expected 20+ experiment JSONs, found {len(jsons)}"
    for p in jsons:
        with open(p, "r", encoding="utf-8") as f:
            json.load(f)


def test_make_figures_runs():
    """The figure-generation pipeline should consume committed JSONs and
    produce vector PDFs without raising."""
    import subprocess

    out_dir = ROOT / "paper" / "en" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(ROOT / "paper" / "scripts" / "make_figures.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"make_figures.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Expect at least the seven canonical paper figures
    expected = [
        "saturation.pdf", "lowrank_seeds.pdf", "sparsity_curve.pdf",
        "quant_modes.pdf", "per_class_f1.pdf", "deploy_latency.pdf",
        "warmup_curve.pdf",
    ]
    for fname in expected:
        path = out_dir / fname
        assert path.exists(), f"Expected figure {fname} was not produced"
        assert path.stat().st_size > 1000, f"{fname} is suspiciously small"
