"""
Sparsity 50% + multi-seed validation.
Each seed warm-starts from its own r_u=8 checkpoint.
Five runs, four parallel workers.
"""

import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SEEDS = [0, 1, 2, 3, 4]
TARGET = 0.5
EPOCHS = 100
RAMP = 50
PATIENCE = 30
MAX_PARALLEL = 4

PYTHON = str(Path("venv/Scripts/python.exe").resolve())
LOG_DIR = Path("logs/sparse_multiseed")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_one(seed):
    init = f"fastgrnn_h16_rw2_ru8_s{seed}_e100_best.pt"
    if not Path(init).exists():
        return f"s{seed}", f"FAIL: {init} not found", 0
    tag = f"sp50_s{seed}"
    log_path = LOG_DIR / f"{tag}.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        PYTHON, "train_sparse.py",
        "--target_sparsity", str(TARGET),
        "--epochs", str(EPOCHS),
        "--ramp_epochs", str(RAMP),
        "--patience", str(PATIENCE),
        "--seed", str(seed),
        "--init_ckpt", init,
        "--best_after_ramp",
    ]
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"# Command: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return tag, "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})", time.time() - t0

if __name__ == "__main__":
    print(f"Sparse multi-seed: target={TARGET}, seeds={SEEDS}, parallel {MAX_PARALLEL}")
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = [pool.submit(run_one, s) for s in SEEDS]
        for fut in as_completed(futures):
            tag, status, dt = fut.result()
            elapsed = time.time() - t_start
            print(f"[{tag}]  {status:10s}  ({dt:.0f}s)  total={elapsed:.0f}s")
    print(f"\n=== Sweep complete: {(time.time()-t_start):.0f}s ===")
