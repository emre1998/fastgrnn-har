"""
Run-to-run / environment drift probe.

Re-runs the three script families that feed best_route_summary.json, in THIS
environment, writing to tagged filenames so nothing committed is touched:

  FastGRNN   run_deploy_budget.py        -> deploy_{ds}RF_s{seed}.json
  shrink-H   run_rnn_epoch_check.py      -> deploy_rnn200_{ds}RS_s{seed}.json
  pruned-H16 run_baseline_tier2_pruned.py-> tier2pruned_{ds}RP_{cell}_h16_s{seed}_e200.json

Every flag is set to match the configuration recorded in the committed results,
so any difference is environment, not configuration.

  python run_drift_probe.py                 # datasets x seeds 0,1
  python run_drift_probe.py --seeds 0       # quicker
  python analyze_drift.py                   # the comparison table
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--datasets", nargs="+", default=["hapt", "wisdm", "pamap2"])
parser.add_argument("--workers", type=int, default=14)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable


def data_of(ds):
    return f"data/processed/{ds}_windows.npz"


jobs = []
for ds in args.datasets:
    for s in args.seeds:
        # --- FastGRNN (low-rank + IHT + calibrated Q15) -----------------------
        jobs.append((f"fastgrnn/{ds}/s{s}", [
            PY, "run_deploy_budget.py", "--data", data_of(ds),
            "--tag", f"{ds}RF", "--seed", str(s)]))

        # --- GRU/LSTM shrink-H at the byte budget ----------------------------
        # the script reads the budget from deploy_{TAG}_s{seed}.json, so seed
        # the lookup from the committed FastGRNN result (identical budget).
        src, dst = EXP / f"deploy_{ds}_s{s}.json", EXP / f"deploy_{ds}RS_s{s}.json"
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
        jobs.append((f"shrink/{ds}/s{s}", [
            PY, "run_rnn_epoch_check.py", "--data", data_of(ds),
            "--tag", f"{ds}RS", "--seed", str(s), "--epochs", "200"]))

        # --- GRU/LSTM magnitude-pruned H=16 ----------------------------------
        jobs.append((f"pruned/{ds}/s{s}", [
            PY, "run_baseline_tier2_pruned.py", "--data", data_of(ds),
            "--tag", f"{ds}RP", "--epochs", "200", "--keep_cell", "181",
            "--seeds", str(s), "--models", "gru", "lstm"]))

# FastGRNN runs are by far the longest; start them first so they overlap the rest
jobs.sort(key=lambda j: 0 if j[0].startswith("fastgrnn") else 1)

env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           CUDA_VISIBLE_DEVICES="-1")     # CPU only: matches how these were run
LOGS = ROOT / "_drift_logs"
LOGS.mkdir(exist_ok=True)
t_start = time.time()
done = 0


def run(job):
    global done
    name, cmd = job
    log = LOGS / (name.replace("/", "_") + ".log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                           stderr=subprocess.STDOUT)
    done += 1
    mins = (time.time() - t0) / 60
    print(f"[{done}/{len(jobs)}] {name}  rc={r.returncode}  {mins:.1f} min "
          f"(elapsed {(time.time()-t_start)/60:.0f} min)", flush=True)
    return name, r.returncode


print(f"{len(jobs)} jobs, {args.workers} workers, CPU-only, 1 thread each")
print(f"logs: {LOGS.relative_to(ROOT)}/\n", flush=True)

with ThreadPoolExecutor(max_workers=args.workers) as ex:
    results = list(ex.map(run, jobs))

bad = [n for n, rc in results if rc != 0]
print(f"\nfinished in {(time.time()-t_start)/60:.0f} min")
print("FAILED:", bad if bad else "none")
print("next:  python analyze_drift.py")
