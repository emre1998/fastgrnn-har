"""
Bootstrap confidence intervals on the 5-seed results, and CIs on the margins that
the ranking claims rest on.

Why this exists
---------------
Reporting mean +/- std hides the shape of a 5-seed distribution -- most sharply on
HAPT FastGRNN, where one seed collapses (the low-rank instability located by the
ablation) and drags the mean while the other four sit high. A reader cannot tell
from "0.86 +/- 0.08" whether that is a tight cluster or an outlier.

So, per dataset and regime, this reports the per-seed values, a 95% bootstrap CI of
the mean, and -- for the equal-byte comparison the reversal claim depends on -- a
bootstrap CI of the *margin* between FastGRNN and the best baseline. If that CI
excludes zero the ranking is statistically supported; if it straddles zero the
honest word is "tie", not "win".

n = 5 is small. Bootstrap CIs from five points are wide and approximate, and this is
stated rather than hidden: the per-seed values are printed alongside every interval
so the reader sees the raw distribution, not only a derived band.

  python analyze_bootstrap.py            # prints + writes BOOTSTRAP.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

EXP = Path("experiments")
RNG = np.random.default_rng(0)      # pinned so the intervals reproduce
N_BOOT = 20000
DATASETS = ["hapt", "wisdm", "pamap2"]
CELLS = ["gru", "lstm", "fastgrnn"]
CELL_LBL = {"gru": "GRU", "lstm": "LSTM", "fastgrnn": "FastGRNN"}


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---- per-seed extractors --------------------------------------------------
def tier1_seeds(ds, cell):
    pfx = "" if ds == "hapt" else f"{ds}_"
    files = sorted(glob.glob(str(EXP / f"baseline_{pfx}{cell}_h16_s*_e120.json")))
    return np.array([jload(f)["test_macro_f1"] for f in files])


def fastgrnn_budget_seeds(ds):
    files = sorted(glob.glob(str(EXP / f"deploy_{ds}_s*.json")))
    return np.array([jload(f)["fastgrnn"]["q15_f1"] for f in files])


def shrink_seeds(ds, cell):
    files = sorted(glob.glob(str(EXP / f"deploy_rnn200_{ds}_s*.json")))
    return np.array([jload(f)[cell]["q15_f1"] for f in files])


def pruned_seeds(ds, cell):
    files = sorted(glob.glob(str(EXP / f"tier2pruned_{ds}_{cell}_h16_s*_e200.json")))
    return np.array([jload(f)["q15_macro_f1"] for f in files])


def best_route_seeds(ds, cell):
    """The per-seed array of whichever route has the higher mean (matches the
    best-of choice the equal-byte table makes)."""
    sh, pr = shrink_seeds(ds, cell), pruned_seeds(ds, cell)
    if pr.size and pr.mean() >= sh.mean():
        return pr, "pruned"
    return sh, "shrink"


# ---- bootstrap ------------------------------------------------------------
def ci_mean(x, alpha=0.05):
    if x.size == 0:
        return (np.nan, np.nan)
    boot = RNG.choice(x, size=(N_BOOT, x.size), replace=True).mean(axis=1)
    return tuple(np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def ci_diff(a, b, alpha=0.05):
    """Two-sample bootstrap CI of mean(a) - mean(b). Unpaired: the seed-1
    collapse is a FastGRNN-specific effect, not a shared data-shuffle one, so
    pairing by seed would understate the baseline's independence."""
    da = RNG.choice(a, size=(N_BOOT, a.size), replace=True).mean(axis=1)
    db = RNG.choice(b, size=(N_BOOT, b.size), replace=True).mean(axis=1)
    d = da - db
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (a.mean() - b.mean()), lo, hi


def seed_str(x):
    return "[" + ", ".join(f"{v:.3f}" for v in x) + "]"


# ---- report ---------------------------------------------------------------
def main():
    L = ["# Bootstrap confidence intervals (5 seeds)",
         "",
         f"95% percentile bootstrap, {N_BOOT} resamples, RNG seed 0. Per-seed values "
         "are shown next to every interval: with n = 5 the interval is wide and "
         "approximate, and the raw distribution is the honest primary record.",
         ""]

    # ---- equal capacity ---------------------------------------------------
    L += ["## Equal capacity (Tier-1, H=16)", "",
          "| Dataset | Cell | Mean [95% CI] | Per-seed |", "|---|---|---|---|"]
    for ds in DATASETS:
        for cell in CELLS:
            x = tier1_seeds(ds, cell)
            lo, hi = ci_mean(x)
            flag = ""
            if x.size and (x.max() - x.min()) > 0.15:
                flag = f" ⚠ spread {x.max()-x.min():.2f}"
            L.append(f"| {ds.upper()} | {CELL_LBL[cell]} | "
                     f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}] | {seed_str(x)}{flag} |")
    L.append("")

    # ---- equal byte, margins ----------------------------------------------
    L += ["## Equal byte budget — margin of FastGRNN over the best baseline", "",
          "The margin is FastGRNN minus the stronger of the two baseline routes "
          "(shrink-H / pruned-H16). A 95% CI that excludes zero supports the ranking; "
          "one that straddles zero is a statistical tie.", "",
          "| Dataset | FastGRNN | Best baseline | Margin [95% CI] | Verdict |",
          "|---|---|---|---|---|"]
    verdicts = {}
    for ds in DATASETS:
        fg = fastgrnn_budget_seeds(ds)
        # the strongest baseline cell at the budget
        cand = {}
        for cell in ("gru", "lstm"):
            arr, route = best_route_seeds(ds, cell)
            cand[cell] = (arr, route)
        base_cell = max(cand, key=lambda c: cand[c][0].mean())
        barr, broute = cand[base_cell]
        m, lo, hi = ci_diff(fg, barr)
        excl = lo > 0 or hi < 0
        verdict = ("**supported**" if lo > 0 else
                   ("baseline ahead" if hi < 0 else "tie (CI spans 0)"))
        verdicts[ds] = (m, lo, hi, verdict)
        L.append(f"| {ds.upper()} | {fg.mean():.3f} {seed_str(fg)} | "
                 f"{CELL_LBL[base_cell]} {broute} {barr.mean():.3f} | "
                 f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {verdict} |")
    L.append("")

    # ---- honest reading ---------------------------------------------------
    L += ["## What the intervals say", ""]
    for ds in DATASETS:
        m, lo, hi, verdict = verdicts[ds]
        L.append(f"- **{ds.upper()}**: margin {m:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] "
                 f"— {verdict}.")
    L += ["",
          "The equal-byte win should be reported at the strength the intervals "
          "actually support: where the CI excludes zero it is a result, where it "
          "spans zero it is a tie and must be called one. HAPT FastGRNN's wide "
          "interval comes from a single collapsing seed (the low-rank instability, "
          "characterized in the ablation), not from broad noise — which is why the "
          "per-seed column, not the mean alone, is the honest record.",
          ""]

    text = "\n".join(L) + "\n"
    Path("BOOTSTRAP.md").write_text(text, encoding="utf-8")
    print(text)
    print("wrote BOOTSTRAP.md")


if __name__ == "__main__":
    main()
