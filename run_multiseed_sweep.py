"""
Multi-seed grid sweep - academic-grade variance estimation.
Config grid: r_u in {4, 6, 8, 12}, seed in {0, 1, 2, 3, 4}
Total: 4 x 5 = 20 runs, max 4 in parallel (each with 2 threads).
Expected wall-clock: ~1.5 hours (8 physical, 16 logical cores).

Output: experiments/fastgrnn_h16_rw2_ru{R}_s{S}_e100.json (20 files)
Aggregate the results afterwards with:  python aggregate_seeds.py
"""

import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

R_U_VALUES = [4, 6, 8, 12]
SEEDS      = [0, 1, 2, 3, 4]
R_W        = 2
HIDDEN     = 16
EPOCHS     = 100
PATIENCE   = 30
MAX_PARALLEL = 4   # 4 parallel x 2 threads = 8 cores

PYTHON = str(Path("venv/Scripts/python.exe").resolve())
LOG_DIR = Path("logs/multiseed")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_one(r_u, seed):
    tag = f"ru{r_u}_s{seed}"
    log_path = LOG_DIR / f"{tag}.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        PYTHON, "train_fastgrnn.py",
        "--hidden", str(HIDDEN),
        "--r_w", str(R_W),
        "--r_u", str(r_u),
        "--epochs", str(EPOCHS),
        "--patience", str(PATIENCE),
        "--seed", str(seed),
        "--tag_seed",
    ]
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"# Command: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    return tag, status, dt

if __name__ == "__main__":
    jobs = [(r_u, seed) for r_u in R_U_VALUES for seed in SEEDS]
    print(f"Total runs: {len(jobs)}, max parallel: {MAX_PARALLEL}.")
    print(f"Logs: {LOG_DIR}/")
    print(f"Result JSONs: experiments/fastgrnn_h16_rw2_ru*_s*_e100.json\n")
    t_start = time.time()

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_one, r_u, seed): (r_u, seed) for r_u, seed in jobs}
        for fut in as_completed(futures):
            tag, status, dt = fut.result()
            completed += 1
            elapsed = time.time() - t_start
            print(f"[{completed:2d}/{len(jobs)}] {tag:12s}  {status:10s}  ({dt:.0f}s)  "
                  f"total_elapsed={elapsed:.0f}s")

    total = time.time() - t_start
    print(f"\n=== Sweep complete: {total:.0f}s = {total/60:.1f} minutes ===")
    print(f"Aggregate the results: python aggregate_seeds.py")
