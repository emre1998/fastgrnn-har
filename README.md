# From Compression to Deployment: Real-Time and Energy-Efficient FastGRNN on Ultra-Constrained Microcontrollers

[![arXiv](https://img.shields.io/badge/arXiv-2606.17249-b31b1b.svg)](https://arxiv.org/abs/2606.17249)

End-to-end reproduction of **FastGRNN** (Kusupati et al., NeurIPS 2018) for
real-time Human Activity Recognition (HAR), deployed on two bare-metal
microcontroller targets:

- **Arduino Uno R3** (ATmega328P) — 8-bit AVR, 32 KB Flash, 2 KB SRAM
- **MSP430G2553** (TI LaunchPad) — 16-bit, 16 KB Flash, 512 B SRAM,
  **no hardware multiplier**

A single portable C inference engine compiles unmodified on both
targets and produces **bit-equivalent** predictions, matching a PyTorch
reference at **100% agreement** across 3,399 test windows.

## Headline Numbers

| Metric                              | Value          |
|-------------------------------------|----------------|
| Test macro F1 (HAPT, 6 classes)     | **0.918**      |
| Deployed weight storage             | **566 bytes**  |
| Nonzero parameters                  | **283**        |
| Real-time per-sample latency (Arduino) | **9.21 ms** (46% of 20 ms budget) |
| Real-time per-sample latency (MSP430)  | **13 ms**   (65% of 20 ms budget) |
| LUT-based speedup vs `expf`/`tanhf` (MSP430) | **30.5×** |
| Cross-platform prediction agreement | **100% / 3,399 windows** |

## Compression Pipeline (L-S-Q)

```
 Float training  →  Low-rank  →  IHT sparsity  →  Q15 + calib.  →  C inference
   H=16, d=3        r_w=2, r_u=8     s=0.5           per-tensor       portable
                                                     scale            (AVR + MSP430)
```

**Preprint:** [arXiv:2606.17249](https://arxiv.org/abs/2606.17249)

See [`paper/en/fastgrnn-har-en.pdf`](paper/en/fastgrnn-har-en.pdf) for the full write-up,
including a deployable LUT recipe for multiplier-less MCUs and a
characterization of the recurrent warm-up latency (median 74 samples / 1.48 s).

## Quick Start

### Prerequisites
- Python 3.10+
- PyTorch 2.x, NumPy, Matplotlib
- (Optional) Arduino IDE 2.x for AVR deployment
- (Optional) Code Composer Studio 12.x for MSP430 deployment

### Train and reproduce

```bash
# Install dependencies
pip install torch numpy matplotlib scikit-learn

# Download and prepare the three datasets
python download_hapt.py   && python build_dataset.py
python download_wisdm.py  && python build_wisdm.py
python download_pamap2.py && python build_pamap2.py

# Confirm the rebuild matches the data the published results came from
python verify_data.py

# Train the deployed model (single seed)
python train_sparse.py --rw 2 --ru 8 --sparsity 0.5 --seed 0

# Cross-check the deployed Q15 inference against PyTorch
python arduino/test_inference_python.py
# Expect: 100% prediction agreement on 3,399 test windows

# Regenerate paper figures from experiment JSON files
python paper/scripts/make_figures.py
```

**Thread count is pinned to 1.** PyTorch's CPU BLAS changes its reduction order with
the thread count, which changes floating-point results, which over a few hundred
epochs changes where training lands. `repro.py` pins it and stamps the environment
into every result file; override only for a deliberate experiment, never to
reproduce a published number:

```bash
FASTGRNN_THREADS=4 python run_baseline_tier1.py ...   # not comparable
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the audit that established this,
including which results reproduce bit-exactly and which do not.

### Which script produces which result

| Script | Writes | Used by |
|---|---|---|
| `run_baseline_tier1.py` | `baseline_{ds}_{cell}_h16_s*_e120.json` | equal-capacity half of the ranking figure; size/accuracy frontier |
| `run_pareto_sweep.py` | `pareto_*.json`, `pareto_summary.json` | `pareto_bytes.pdf` |
| `run_deploy_budget.py` | `deploy_{ds}_s*.json` | FastGRNN at the byte budget |
| `run_rnn_epoch_check.py` | `deploy_rnn200_{ds}_s*.json` | GRU/LSTM shrink-*H* route at 200 epochs |
| `run_baseline_tier2_pruned.py` | `tier2pruned_{ds}_*.json` | GRU/LSTM pruned-*H*16 route |
| `analyze_best_route.py` | `best_route_summary.json` | equal-byte half of the ranking figure |
| `run_multiseed_sweep.py` | `multiseed_summary.json` | `lowrank_seeds.pdf` |
| `run_sparsity_sweep.py`, `run_sparse_multiseed.py` | `sparse_*.json` | `sparsity_curve.pdf` |
| `ptq_full_eval.py` | `ptq_full_multiseed.json` | `quant_modes.pdf`; the Q15-losslessness result |
| `epoch_saturation.py` | `saturation_h16.json` | `saturation.pdf` |
| `run_lowrank_stage.py`, `analyze_ablation.py` | `ablation_summary.json` | compression ablation |
| `analyze_footprint.py` | — (prints) | analytical Flash/SRAM budget |
| `build_deploy_firmware.py` | `arduino/{cell}_har/model_weights.h` | the flashed weights |
| `verify_firmware.py` | — (checks) | host-side C/PyTorch parity |
| `paper/scripts/make_figures.py` | `paper/en/figures/*.pdf` | every data figure |

Hardware numbers come from the firmware in `msp/`, not from these scripts:
`TEST_MODE` selects bench latency (1), bench energy (3), live latency with the
sensor (4), or live energy (5). Results are collected in
[B2_RESULTS.md](B2_RESULTS.md); the measurement protocol is in
[docs/energy_measurement.md](docs/energy_measurement.md).

### Deploy

```bash
# Export Q15 weights to a C header
cd arduino && python export_to_c.py

# Generate sigmoid/tanh LUT
python generate_lut.py

# Generate embedded test vectors
python generate_test_data.py
```

Then open `arduino/fastgrnn_har/fastgrnn_har.ino` in Arduino IDE
**or** `msp/ccs_fastgrnn_har/` as a Code Composer Studio project
and upload.

## Repository Layout

```
fastgrnn-har/
├── paper/                       Full LaTeX source and PDF
│   ├── en/                      English (canonical)
│   │   ├── fastgrnn-har-en.tex, fastgrnn-har-en.pdf
│   │   ├── sections/            Per-section .tex sources
│   │   ├── figures/             Auto-generated vector PDFs
│   │   └── references.bib
│   ├── tr/                      Turkish translation (fastgrnn-har-tr.pdf)
│   └── scripts/make_figures.py  Figure generation from JSON results
├── arduino/                     Arduino Uno deployment + Python tools
│   ├── fastgrnn_har/            Arduino sketch + headers
│   ├── export_to_c.py           Q15 weight export
│   ├── generate_lut.py          Activation LUT generator
│   └── test_inference_python.py PyTorch ↔ C parity check
├── msp/                         MSP430 deployment
│   ├── ccs_fastgrnn_har/        Code Composer Studio bare-metal project
│   └── fastgrnn_har_msp/        Energia alternative
├── experiments/                 All training/eval JSON results (30+)
├── notes/                       Weekly memory notes (TR)
└── data/                        HAPT dataset (downloaded)
```

## Citation

If you use this work, please cite:

```bibtex
@misc{kizilates2026fastgrnnharmcu,
  author        = {Kızılateş, Emre Can},
  title         = {From Compression to Deployment: Real-Time and Energy-Efficient
                   FastGRNN on Ultra-Constrained Microcontrollers},
  year          = {2026},
  eprint        = {2606.17249},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  url           = {https://arxiv.org/abs/2606.17249}
}
```

The original FastGRNN algorithm is by Kusupati et al.:

```bibtex
@inproceedings{kusupati2018fastgrnn,
  title     = {{FastGRNN}: A Fast, Accurate, Stable and Tiny Kilobyte
               Sized Gated Recurrent Neural Network},
  author    = {Kusupati, Aditya and Singh, Manish and Bhatia, Kush and
               Kumar, Ashish and Jain, Prateek and Varma, Manik},
  booktitle = {NeurIPS},
  year      = {2018}
}
```

## License

Code and configuration in this repository are released under the
[Apache License 2.0](LICENSE).

The HAPT dataset is the property of the original authors
(Reyes-Ortiz et al., 2015) and is redistributed for reproducibility
under the dataset's original UCI Machine Learning Repository terms.

## Contact

Emre Can Kızılateş — `kizilatesemrecan@gmail.com`
