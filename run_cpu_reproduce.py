"""
Re-run on CPU every experiment that used to take the GPU when one was present.

Four training scripts selected `cuda if torch.cuda.is_available() else cpu`, so
the device was a property of whichever machine ran them. Some results came from an
RTX 3060 and others from the CPU, and cells being compared against one another
were not always trained on the same device. CPU also reproduces bit-exactly while
the GPU does not (cuDNN RNN backward is non-deterministic and cudnn.allow_tf32 is
on by default). So: one device for everything, and it is the reproducible one.

The GPU-era files were moved to experiments/archive_gpu/ rather than deleted, so
the two sets can be compared afterwards. This writes the canonical filenames, so
no analysis script needs to change.

  python run_cpu_reproduce.py                 # everything
  python run_cpu_reproduce.py --only tier1    # or pareto
"""
import argparse
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

DATASETS = {"hapt": "data/processed/hapt_windows.npz",
            "wisdm": "data/processed/wisdm_windows.npz",
            "pamap2": "data/processed/pamap2_windows.npz"}
SEEDS = [0, 1, 2, 3, 4]

# Concurrency is bounded by virtual memory, not cores: this torch build loads
# ~2.2 GB of CUDA libraries at import even for a CPU run. Six workers exhausted
# the pagefile; three did not.
WORKERS = 3

ap = argparse.ArgumentParser()
ap.add_argument("--only", choices=["tier1", "pareto"], default=None)
ap.add_argument("--workers", type=int, default=WORKERS)
args = ap.parse_args()

jobs = []

# --- Tier 1: equal capacity, H=16, all three cells -------------------------
if args.only in (None, "tier1"):
    for ds, path in DATASETS.items():
        tag = None if ds == "hapt" else ds        # hapt writes unprefixed names
        for seed in SEEDS:
            cmd = [PY, "run_baseline_tier1.py", "--data", path,
                   "--seeds", str(seed), "--models", "fastgrnn", "gru", "lstm"]
            if tag:
                cmd += ["--tag", tag]
            jobs.append((f"tier1/{ds}/s{seed}", cmd))

# --- Pareto sweep: HAPT only, several hidden sizes per cell ----------------
if args.only in (None, "pareto"):
    GRID = {"fastgrnn": [8, 12], "gru": [4, 5, 6, 8, 10, 12],
            "lstm": [4, 5, 6, 8, 10, 12]}
    for cell, hiddens in GRID.items():
        for h in hiddens:
            for seed in SEEDS:
                jobs.append((f"pareto/{cell}/h{h}/s{seed}",
                             [PY, "run_pareto_sweep.py",
                              "--data", DATASETS["hapt"],
                              "--hiddens", str(h), "--models", cell,
                              "--seeds", str(seed)]))

LOGS = ROOT / "_cpu_repro_logs"
LOGS.mkdir(exist_ok=True)
env = dict(os.environ)          # repro.device()/pin_threads() handle the rest
t0 = time.time()
done = 0


def run(job):
    global done
    name, cmd = job
    log = LOGS / (name.replace("/", "_") + ".log")
    t = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    done += 1
    print(f"[{done}/{len(jobs)}] {name}  rc={r.returncode}  {(time.time()-t)/60:.1f} min"
          f"  (elapsed {(time.time()-t0)/60:.0f} min)", flush=True)
    return name, r.returncode


print(f"{len(jobs)} jobs, {args.workers} workers, CPU pinned by repro.py\n", flush=True)
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    results = list(ex.map(run, jobs))

failed = [j for j, (_, rc) in zip(jobs, results) if rc != 0]
if failed:
    print(f"\nretrying {len(failed)} failed job(s) serially\n", flush=True)
    results += [run(j) for j in failed]

ok = {n for n, rc in results if rc == 0}
bad = sorted({n for n, rc in results if rc != 0} - ok)
print(f"\nfinished in {(time.time()-t0)/60:.0f} min")
print("FAILED:", bad if bad else "none")
print("\nnext: python analyze_tier1_multi.py ; python analyze_best_route.py ;"
      " python paper/scripts/make_figures.py")
