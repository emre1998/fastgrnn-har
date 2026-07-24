"""
Re-run the magnitude-pruned route at 5 seeds with the thread count pinned.

The pruned route is the one the reproducibility audit found unstable: drift was
never zero on it, and it reached +0.248 / -0.268 on single seeds (see
REPRODUCIBILITY.md). It also happens to be the GRU's winning route on both HAPT
and WISDM, so the equal-byte comparison rests on it.

Writes to the tag {ds}V2, leaving the committed tier2pruned_{ds}_* files intact so
the two can be compared rather than one silently replacing the other.

  python run_pruned_v2.py
  python analyze_pruned_v2.py
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable

DATASETS = ["hapt", "wisdm", "pamap2"]
SEEDS = [0, 1, 2, 3, 4]

# Concurrency is bounded by virtual memory, not by cores. This torch build ships
# ~2.2 GB of CUDA libraries (torch_cuda 913 MB, cudnn 562 MB, cublasLt 451 MB) and
# loads them at import even for a CPU-only run, so each worker commits a large
# chunk of address space. Six workers exhausted the pagefile and every job died
# with WinError 1455. Do not "fix" this by installing CPU-only torch: the torch
# build is part of the pinned environment (see REPRODUCIBILITY.md).
WORKERS = 3

jobs = [(ds, s) for ds in DATASETS for s in SEEDS]
LOGS = ROOT / "_pruned_v2_logs"
LOGS.mkdir(exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")   # repro.pin_threads() handles threads
t_start = time.time()
done = 0


def run(job):
    global done
    ds, seed = job
    log = LOGS / f"{ds}_s{seed}.log"
    cmd = [PY, "run_baseline_tier2_pruned.py",
           "--data", f"data/processed/{ds}_windows.npz",
           "--tag", f"{ds}V2", "--epochs", "200", "--keep_cell", "181",
           "--seeds", str(seed), "--models", "gru", "lstm"]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    done += 1
    print(f"[{done}/{len(jobs)}] {ds}/s{seed}  rc={r.returncode}  "
          f"{(time.time()-t0)/60:.1f} min  (elapsed {(time.time()-t_start)/60:.0f} min)",
          flush=True)
    return f"{ds}/s{seed}", r.returncode


print(f"{len(jobs)} jobs, {WORKERS} workers, threads pinned by repro.py\n", flush=True)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    results = list(ex.map(run, jobs))

# Memory-pressure failures are transient: the scripts skip finished outputs, so a
# serial retry costs only the jobs that actually died.
failed = [j for j, (_, rc) in zip(jobs, results) if rc != 0]
if failed:
    print(f"\nretrying {len(failed)} failed job(s) one at a time\n", flush=True)
    results += [run(j) for j in failed]

bad = sorted({n for n, rc in results if rc != 0} -
             {n for n, rc in results if rc == 0})
print(f"\nfinished in {(time.time()-t_start)/60:.0f} min")
print("FAILED:", bad if bad else "none")
