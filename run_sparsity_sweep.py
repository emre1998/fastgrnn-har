"""
Sparsity sweep - target in {0.3, 0.5, 0.7, 0.9}, four parallel workers.
Each run: r_u=8 init, IHT 50-epoch ramp + 50-epoch fine-tune, best_after_ramp.
"""

import subprocess
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TARGETS = [0.3, 0.5, 0.7, 0.9]
SEED = 0
INIT_CKPT = "fastgrnn_h16_rw2_ru8_s0_e100_best.pt"
EPOCHS = 100
RAMP = 50
PATIENCE = 30
MAX_PARALLEL = 4

PYTHON = str(Path("venv/Scripts/python.exe").resolve())
LOG_DIR = Path("logs/sparsity_sweep")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_one(target):
    tag = f"sp{int(target*100)}"
    log_path = LOG_DIR / f"{tag}.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        PYTHON, "train_sparse.py",
        "--target_sparsity", str(target),
        "--epochs", str(EPOCHS),
        "--ramp_epochs", str(RAMP),
        "--patience", str(PATIENCE),
        "--seed", str(SEED),
        "--init_ckpt", INIT_CKPT,
        "--best_after_ramp",
    ]
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"# Command: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return tag, "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})", time.time() - t0

if __name__ == "__main__":
    print(f"Sparsity sweep: {TARGETS}, parallel {MAX_PARALLEL}")
    print(f"Logs: {LOG_DIR}/\n")
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = [pool.submit(run_one, t) for t in TARGETS]
        for fut in as_completed(futures):
            tag, status, dt = fut.result()
            elapsed = time.time() - t_start
            print(f"[{tag}]  {status:10s}  ({dt:.0f}s)  total_elapsed={elapsed:.0f}s")
    print(f"\n=== Sweep complete: {(time.time()-t_start):.0f}s ===")
