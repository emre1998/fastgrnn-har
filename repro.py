"""
Reproducibility helpers: pin the thread count, stamp the environment.

Why this exists
---------------
Thread count is part of the experimental configuration. PyTorch's CPU BLAS
changes its reduction order with the number of threads, which changes
floating-point results, which over a few hundred epochs changes where training
lands. It was never recorded, and a re-run at a different thread count looked
exactly like the results not reproducing. See REPRODUCIBILITY.md.

With the count pinned to 1 -- how the committed results were produced -- the
FastGRNN and shrink-H routes reproduce bit-exactly on HAPT and WISDM.

Usage
-----
    import repro
    repro.pin_threads()          # before building any model

    res = {...,  "env": repro.env_stamp()}

Override for a deliberate experiment (not for producing paper numbers):

    FASTGRNN_THREADS=4 python run_baseline_tier1.py ...

Deliberately NOT done here: torch.use_deterministic_algorithms(). Switching it
on would change results relative to everything already committed, which is the
opposite of what this module is for.
"""
from __future__ import annotations

import os
import platform
import subprocess

DEFAULT_THREADS = 1


def pin_threads(threads: int | None = None) -> int:
    """Pin BLAS/OMP threads. Returns the count actually set."""
    n = int(os.environ.get("FASTGRNN_THREADS", DEFAULT_THREADS)) \
        if threads is None else int(threads)
    # for any library that reads these at import time
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        import torch
        torch.set_num_threads(n)            # the one that actually binds
        torch.set_num_interop_threads(n)
    except (ImportError, RuntimeError):
        # interop threads can only be set once per process; harmless if already set
        pass
    return n


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def env_stamp() -> dict:
    """Everything needed to tell whether a re-run is comparable."""
    stamp = {
        "threads": int(os.environ.get("OMP_NUM_THREADS", DEFAULT_THREADS)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    try:
        import torch
        stamp["torch"] = torch.__version__
        stamp["threads_torch"] = torch.get_num_threads()
        # Several scripts select torch.device("cuda" if available else "cpu"), so
        # the device is not a constant of this repository and must be recorded,
        # not assumed. GPU and CPU produce different floating-point results, by
        # more than the thread count does.
        stamp["cuda_available"] = torch.cuda.is_available()
        stamp["device_would_be"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            stamp["gpu"] = torch.cuda.get_device_name(0)
            stamp["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
            stamp["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    except ImportError:
        pass
    try:
        import numpy
        stamp["numpy"] = numpy.__version__
    except ImportError:
        pass
    return stamp


if __name__ == "__main__":
    import json
    pin_threads()
    print(json.dumps(env_stamp(), indent=2))
