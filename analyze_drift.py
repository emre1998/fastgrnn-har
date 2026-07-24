"""
Compare the drift-probe re-runs against the committed results.

Same seed, same configuration, same code -- only the environment differs, so any
difference here is run-to-run/environment drift rather than a change of method.

The question this answers: the equal-byte margins in best_route_summary.json are
0.033 (HAPT, GRU ahead) and 0.033 (WISDM, FastGRNN ahead). Is the drift small
enough that those margins mean anything?

  python run_drift_probe.py && python analyze_drift.py
"""
import glob
import json
from pathlib import Path

import numpy as np

EXP = Path("experiments")
DATASETS = ["hapt", "wisdm", "pamap2"]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pair(old_pat, new_pat, field):
    """Return {seed: (old, new)} for seeds present on both sides."""
    def collect(pat):
        out = {}
        for p in sorted(glob.glob(str(EXP / pat))):
            s = p.split("_s")[-1].split(".")[0].split("_")[0]
            try:
                out[int(s)] = jload(p)
            except (ValueError, json.JSONDecodeError):
                pass
        return out

    o, n = collect(old_pat), collect(new_pat)
    res = {}
    for s in sorted(set(o) & set(n)):
        try:
            res[s] = (field(o[s]), field(n[s]))
        except KeyError:
            pass
    return res


ROUTES = [
    ("FastGRNN",      "deploy_{ds}_s*.json",
                      "deploy_{ds}RF_s*.json",
                      lambda d: d["fastgrnn"]["q15_f1"]),
    ("GRU shrink",    "deploy_rnn200_{ds}_s*.json",
                      "deploy_rnn200_{ds}RS_s*.json",
                      lambda d: d["gru"]["q15_f1"]),
    ("LSTM shrink",   "deploy_rnn200_{ds}_s*.json",
                      "deploy_rnn200_{ds}RS_s*.json",
                      lambda d: d["lstm"]["q15_f1"]),
    ("GRU pruned",    "tier2pruned_{ds}_gru_h16_s*_e200.json",
                      "tier2pruned_{ds}RP_gru_h16_s*_e200.json",
                      lambda d: d["q15_macro_f1"]),
    ("LSTM pruned",   "tier2pruned_{ds}_lstm_h16_s*_e200.json",
                      "tier2pruned_{ds}RP_lstm_h16_s*_e200.json",
                      lambda d: d["q15_macro_f1"]),
]

MARGINS = {"hapt": ("GRU ahead", 0.033),
           "wisdm": ("FastGRNN ahead", 0.033),
           "pamap2": ("FastGRNN ahead", 0.090)}

print("=" * 74)
print("DRIFT: committed results vs re-run in this environment (same seed/config)")
print("=" * 74)

all_d = []
for ds in DATASETS:
    who, margin = MARGINS[ds]
    print(f"\n--- {ds.upper()}   (equal-byte margin {margin:+.3f}, {who}) ---")
    print(f"  {'route':<13} {'seed':>4} {'committed':>10} {'re-run':>9} {'drift':>8}")
    for label, op, np_, field in ROUTES:
        got = pair(op.format(ds=ds), np_.format(ds=ds), field)
        if not got:
            print(f"  {label:<13} {'--':>4} {'(pending)':>10}")
            continue
        for s, (o, n) in got.items():
            d = n - o
            all_d.append(abs(d))
            flag = "  <-- exceeds margin" if abs(d) > margin else ""
            print(f"  {label:<13} {s:>4} {o:>10.4f} {n:>9.4f} {d:>+8.4f}{flag}")

if all_d:
    a = np.array(all_d)
    print("\n" + "=" * 74)
    print(f"|drift|  mean {a.mean():.4f}   median {np.median(a):.4f}   "
          f"max {a.max():.4f}   n={len(a)}")
    print("=" * 74)
    print("\nReading it:")
    print("  Drift well under the margins  -> margins are real; stamp the")
    print("     environment, report drift as measured uncertainty, no full re-run.")
    print("  Drift comparable to a margin  -> that ranking is not resolvable at")
    print("     this precision; state it as a tie rather than a win.")
    print("\nNote: each committed headline is a 5-seed mean, so per-seed drift")
    print("partly averages out. Compare against the 5-seed std, not a single seed.")
else:
    print("\nNo pairs yet -- run run_drift_probe.py first.")
