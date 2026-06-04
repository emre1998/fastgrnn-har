# FastGRNN on Bare-Metal Microcontrollers

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

See [`paper/en/main.pdf`](paper/en/fastgrnn-har-en.pdf) for the full write-up,
including a deployable LUT recipe for multiplier-less MCUs and a
characterization of the ~2 s recurrent warm-up latency.

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

# Download and prepare the HAPT dataset
python download_hapt.py
python build_dataset.py

# Train the deployed model (single seed)
python train_sparse.py --rw 2 --ru 8 --sparsity 0.5 --seed 0

# Cross-check the deployed Q15 inference against PyTorch
python test_inference_python.py
# Expect: 100% prediction agreement on 3,399 test windows

# Regenerate paper figures from experiment JSON files
python paper/scripts/make_figures.py
```

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
│   │   ├── main.tex, main.pdf
│   │   ├── sections/            Per-section .tex sources
│   │   ├── figures/             Auto-generated vector PDFs
│   │   └── references.bib
│   ├── tr/, de/                 Turkish / German translations (in progress)
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
  author = {Kızılateş, Emre Can},
  title  = {FastGRNN on Bare-Metal Microcontrollers: An End-to-End
            Reproduction with Low-Rank, Sparse, Quantized Inference
            for Real-Time Human Activity Recognition},
  year   = {2026},
  howpublished = {\url{https://github.com/emre1998/fastgrnn-har}}
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
