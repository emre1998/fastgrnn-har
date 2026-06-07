#!/usr/bin/env bash
# run_reproduce.sh - End-to-end reproduction of the deployed FastGRNN model.
#
# Reproduces the headline numbers from the paper:
#   - macro F1 = 0.918
#   - 566 bytes of weights, 283 nonzero parameters
#   - 100% prediction agreement with the deployed C inference engine
#
# Wall-clock budget: ~2 hours on a standard desktop CPU.
#
# Usage:  ./run_reproduce.sh [--skip-data] [--skip-train] [--skip-export]

set -euo pipefail

SKIP_DATA=0
SKIP_TRAIN=0
SKIP_EXPORT=0
for arg in "$@"; do
    case "$arg" in
        --skip-data)   SKIP_DATA=1 ;;
        --skip-train)  SKIP_TRAIN=1 ;;
        --skip-export) SKIP_EXPORT=1 ;;
        -h|--help)
            echo "Usage: $0 [--skip-data] [--skip-train] [--skip-export]"
            exit 0
            ;;
    esac
done

echo "==============================================================="
echo " FastGRNN-HAR end-to-end reproduction"
echo "==============================================================="

# ---------------------------------------------------------------------------
# 1) Dataset
# ---------------------------------------------------------------------------
if [ "$SKIP_DATA" -eq 0 ]; then
    echo
    echo "[1/4] Downloading and preprocessing the HAPT dataset..."
    python download_hapt.py
    python build_dataset.py
else
    echo "[1/4] Skipped (dataset assumed ready in data/processed/)"
fi

# ---------------------------------------------------------------------------
# 2) Training (MLP baseline + FastGRNN H=16 + low-rank + sparse)
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo
    echo "[2/4] Training the deployed model (seed 0)..."

    # Float baseline (FastGRNN H=16, full-rank)
    python train_fastgrnn.py --hidden 16 --epochs 100 --seed 0 --tag_seed

    # + Low-rank (r_w=2, r_u=8)
    python train_fastgrnn.py --hidden 16 --r_w 2 --r_u 8 --epochs 100 \
        --seed 0 --tag_seed

    # + Sparsity (s=0.5, IHT cubic ramp, warm-start from low-rank ckpt)
    python train_sparse.py \
        --target_sparsity 0.5 --epochs 100 --ramp_epochs 50 \
        --seed 0 --best_after_ramp \
        --init_ckpt fastgrnn_h16_rw2_ru8_s0_e100_best.pt
else
    echo "[2/4] Skipped (using existing checkpoints)"
fi

# ---------------------------------------------------------------------------
# 3) Q15 export + cross-check
# ---------------------------------------------------------------------------
if [ "$SKIP_EXPORT" -eq 0 ]; then
    echo
    echo "[3/4] Exporting Q15 weights and verifying PyTorch <-> C agreement..."
    (
        cd arduino
        python export_to_c.py
        python generate_lut.py
        python test_inference_python.py
        python generate_test_data.py
    )
else
    echo "[3/4] Skipped"
fi

# ---------------------------------------------------------------------------
# 4) Paper figures (from the committed experiment JSONs)
# ---------------------------------------------------------------------------
echo
echo "[4/4] Regenerating paper figures..."
python paper/scripts/make_figures.py

echo
echo "==============================================================="
echo " Reproduction complete."
echo "==============================================================="
echo " Expected outputs:"
echo "   - experiments/sparse_h16_rw2_ru8_sp50_s0_e100.json"
echo "   - arduino/fastgrnn_har/model_weights.h (566 bytes of Q15 weights)"
echo "   - arduino/test_vectors.json (PyTorch reference predictions)"
echo "   - paper/en/figures/*.pdf  (regenerated)"
echo
echo " Deploy to hardware:"
echo "   Arduino:  open arduino/fastgrnn_har/fastgrnn_har.ino in Arduino IDE"
echo "   MSP430:   import msp/ccs_fastgrnn_har/ into Code Composer Studio"
