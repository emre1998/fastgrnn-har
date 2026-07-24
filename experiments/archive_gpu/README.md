# GPU-era results

These are the Tier-1 and Pareto results as they were produced before the device
was pinned, when four training scripts selected
`cuda if torch.cuda.is_available() else cpu` and this machine had an RTX 3060.

They are **also still present under their canonical names** in `experiments/`, so
the repository is self-consistent: the committed figures were built from these
files and can be regenerated from them. Nothing here is orphaned.

## Why they are being replaced

Not because they are wrong, but because they cannot be reproduced and because the
device was not consistent across the comparison. cuDNN's RNN backward pass uses
atomics and is non-deterministic by default, and `cudnn.allow_tf32` is on by
default — those kernels compute with a 10-bit mantissa against FP32's 23. One
configuration measured on both devices:

| HAPT GRU pruned, seed 0 | F1 |
|---|---|
| as committed | 0.9128 |
| re-run on GPU | 0.9036 |
| re-run on CPU | 0.8867 |

The GPU re-run lands closer but does not return the committed number either. The
CPU, with the thread count pinned, reproduces exactly.

In the Pareto sweep the inconsistency reached inside a single figure: FastGRNN ran
on the CPU while GRU and LSTM ran on the GPU, so cells plotted against each other
were not trained under the same arithmetic.

## Pending

`run_cpu_reproduce.py` regenerates Tier-1 and the Pareto sweep on CPU into the
canonical filenames. It has not been run yet — the machine did not have the
virtual memory for it (this torch build maps ~2.2 GB of CUDA libraries per worker
even for a CPU-only run, and three workers exhausted the pagefile).

Until it runs, `experiments/` holds GPU-era numbers and this directory holds the
same files as a record of what they were. Afterwards, `experiments/` holds CPU
numbers and this directory becomes the before-picture.
