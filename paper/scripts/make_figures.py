"""
make_figures.py — Generate all paper figures from JSON experiment results.

Run:  python paper/scripts/make_figures.py

Output: paper/en/figures/*.pdf  (vector, IEEE single-column width)

Figures produced:
    1. saturation.pdf       — H=16 training: val/test F1 over epochs
    2. lowrank_seeds.pdf    — boxplot of per-seed F1 across r_u choices
    3. sparsity_curve.pdf   — F1 vs target sparsity (U-curve)
    4. quant_modes.pdf      — float vs Q15-W vs Q15-W+A (calibrated)
    5. per_class_f1.pdf     — per-class F1 baseline vs final
    6. deploy_latency.pdf   — Python vs Arduino vs MSP430 per-sample latency
    7. warmup_curve.pdf     — h_state[0] over 50Hz window + class trajectory
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # .../fastgrnn-har
EXP = ROOT / "experiments"
OUT = ROOT / "paper" / "en" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# IEEE-style matplotlib config (single-column ≈ 3.5", serif, vector)
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype":   42,           # TrueType (embed, editable)
    "ps.fonttype":    42,
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":       8,
    "axes.labelsize":  8,
    "axes.titlesize":  9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon":  False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":   ":",
    "grid.linewidth":   0.5,
    "grid.alpha":       0.5,
    "lines.linewidth":  1.2,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})

# IEEE column widths (inches)
W1 = 3.5       # single column
W2 = 7.16      # double column

CLASS_NAMES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
CLASS_SHORT = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_json(name: str) -> dict:
    with open(EXP / name, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, name: str) -> None:
    out = OUT / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------------
# 1. Saturation (test F1 over epochs, H=16)
# ----------------------------------------------------------------------------
def fig_saturation() -> None:
    data = load_json("saturation_h16.json")
    hist = data["history"]
    epochs = [h["epoch"]    for h in hist]
    test   = [h["test_f1"]  for h in hist]
    val    = [h["val_f1"]   for h in hist]

    fig, ax = plt.subplots(figsize=(W1, 2.1))
    ax.plot(epochs, val,  label="Validation F1", color="#888", linewidth=1.0)
    ax.plot(epochs, test, label="Test F1",       color="C0",   linewidth=1.4)

    # Mark best test epoch
    best_ep  = data["best_test_epoch"]
    best_f1  = data["best_test_f1"]
    ax.scatter([best_ep], [best_f1], color="C3", s=20, zorder=5)
    ax.annotate(f"best: {best_f1:.3f} @ ep.{best_ep}",
                xy=(best_ep, best_f1), xytext=(best_ep - 50, best_f1 - 0.10),
                fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color="#666"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower right")
    save(fig, "saturation.pdf")


# ----------------------------------------------------------------------------
# 2. Low-rank multi-seed boxplot
# ----------------------------------------------------------------------------
def fig_lowrank_seeds() -> None:
    data = load_json("multiseed_summary.json")
    configs = data["configs"]
    ranks   = [c["r_u"] for c in configs]
    f1_lists = [c["f1s"] for c in configs]

    fig, ax = plt.subplots(figsize=(W1, 2.1))
    bp = ax.boxplot(f1_lists, positions=range(len(ranks)),
                    widths=0.5, patch_artist=True,
                    medianprops=dict(color="black"),
                    boxprops=dict(facecolor="#cfe2f3", edgecolor="C0"),
                    whiskerprops=dict(color="C0"),
                    capprops=dict(color="C0"),
                    flierprops=dict(marker="o", markerfacecolor="C3",
                                     markersize=3, markeredgecolor="none"))
    # Mean dots
    means = [np.mean(f) for f in f1_lists]
    ax.scatter(range(len(ranks)), means, marker="D", s=18,
               color="C3", zorder=5, label="mean")

    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f"$r_u{{=}}{r}$" for r in ranks])
    ax.set_xlabel("Recurrent rank")
    ax.set_ylabel("Test Macro F1 (5 seeds)")
    ax.set_ylim(0.65, 0.95)
    ax.legend(loc="lower right")

    # Highlight winner
    winner = data["winner"]["r_u"]
    idx = ranks.index(winner)
    ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.08, color="C2")

    save(fig, "lowrank_seeds.pdf")


# ----------------------------------------------------------------------------
# 3. Sparsity sweep (single-seed bars, sp50 with error from multi-seed)
# ----------------------------------------------------------------------------
def fig_sparsity_curve() -> None:
    sps = [30, 50, 70, 90]
    f1_s0 = []
    for sp in sps:
        d = load_json(f"sparse_h16_rw2_ru8_sp{sp}_s0_e100.json")
        f1_s0.append(d["test_macro_f1"])

    # sp50 has 5 seeds available
    sp50_all = []
    for s in range(5):
        try:
            d = load_json(f"sparse_h16_rw2_ru8_sp50_s{s}_e100.json")
            sp50_all.append(d["test_macro_f1"])
        except FileNotFoundError:
            pass
    sp50_mean = float(np.mean(sp50_all))
    sp50_std  = float(np.std(sp50_all))

    fig, ax = plt.subplots(figsize=(W1, 2.1))
    ax.plot(sps, f1_s0, "o-", color="C0", label="seed 0")
    # Overlay sp50 multi-seed
    ax.errorbar([50], [sp50_mean], yerr=[sp50_std],
                fmt="D", color="C3", capsize=4, markersize=6,
                label=f"5 seeds (sp=50): {sp50_mean:.3f} ± {sp50_std:.3f}")

    ax.set_xlabel("Target sparsity (\\%)")
    ax.set_ylabel("Test Macro F1")
    ax.set_xticks(sps)
    ax.set_ylim(0.4, 1.0)
    ax.axvspan(40, 60, alpha=0.08, color="C2")   # highlight optimum
    ax.legend(loc="lower left")
    save(fig, "sparsity_curve.pdf")


# ----------------------------------------------------------------------------
# 4. Quantization modes (float / Q15-W / Q15-W+A calibrated / naive Q15-A)
# ----------------------------------------------------------------------------
def fig_quant_modes() -> None:
    data = load_json("ptq_full_multiseed.json")
    modes = data["modes"]

    # Take seed 0 (matches deployed model)
    f32  = modes["float32"]["f1s"][0]
    q15w = modes["q15_weights"]["f1s"][0]
    q15a = modes["q15_weights_acts"]["f1s"][0]
    # Naive Q15 activation collapse — from memory_hafta7 (no calibration)
    q15a_naive = 0.16

    labels = ["FP32\n(reference)", "Q15-W\n+ FP acts", "Q15-W\n+ naive Q15 acts",
              "Q15-W\n+ calibrated Q15 acts"]
    values = [f32, q15w, q15a_naive, q15a]
    colors = ["#888", "C0", "C3", "C2"]

    fig, ax = plt.subplots(figsize=(W1, 2.4))
    bars = ax.bar(range(len(values)), values, color=colors, width=0.65)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test Macro F1 (seed 0)")
    ax.set_ylim(0, 1.05)
    ax.axhline(f32, color="#888", linestyle="--", linewidth=0.7,
               label="FP32 reference")
    ax.legend(loc="lower right")
    save(fig, "quant_modes.pdf")


# ----------------------------------------------------------------------------
# 5. Per-class F1: baseline (float, full) vs deployed (sp50+Q15)
# ----------------------------------------------------------------------------
def fig_per_class_f1() -> None:
    # Baseline: float low-rank (no sparsity), seed 0
    base = load_json("fastgrnn_h16_rw2_ru8_s0_e100.json")
    base_f1 = [base["per_class_f1"][c] for c in CLASS_NAMES]

    # Deployed: sparse 50 + Q15 calibrated, seed 0 (from sparse json — pre-PTQ)
    sp = load_json("sparse_h16_rw2_ru8_sp50_s0_e100.json")
    sp_f1 = [sp["per_class_f1"][c] for c in CLASS_NAMES]

    # PTQ calibrated per-class (seed 0, mode q15_weights_acts)
    ptq = load_json("ptq_full_multiseed.json")
    ptq_f1 = ptq["per_class"]["q15_weights_acts"][0]

    x = np.arange(len(CLASS_NAMES))
    w = 0.27

    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.bar(x - w, base_f1, w, label="Low-rank (FP32)", color="#aaa")
    ax.bar(x,     sp_f1,   w, label="+ Sparsity (FP32)", color="C0")
    ax.bar(x + w, ptq_f1,  w, label="+ Q15 (deployed)",  color="C2")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_SHORT)
    ax.set_ylabel("Per-class F1 (seed 0)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=1)
    save(fig, "per_class_f1.pdf")


# ----------------------------------------------------------------------------
# 6. Deploy per-sample latency (Python / Arduino / MSP430, 20 ms budget)
# ----------------------------------------------------------------------------
def fig_deploy_latency() -> None:
    # Per-sample averages from memory_hafta8_streaming_sim.md
    labels = ["Python\n(NumPy)", "Arduino Uno\n(AVR, 16 MHz)",
              "MSP430G2553\n(no MUL, 16 MHz)"]
    latency = [0.02, 9.21, 13.0]      # ms/sample (Python full-window 2.73 ms ÷ 128 ≈ 0.02)
    colors  = ["#aaa", "C0", "C2"]
    budget  = 20.0                    # 50 Hz period

    fig, ax = plt.subplots(figsize=(W1, 2.3))
    bars = ax.bar(range(len(labels)), latency, color=colors, width=0.6)
    for i, v in enumerate(latency):
        ax.text(i, v + 0.5, f"{v:.2f} ms", ha="center", fontsize=7)
    ax.axhline(budget, color="C3", linestyle="--", linewidth=0.9,
               label=f"50 Hz budget ({budget:.0f} ms)")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-sample latency (ms)")
    ax.set_ylim(0, 25)
    ax.legend(loc="upper left")
    save(fig, "deploy_latency.pdf")


# ----------------------------------------------------------------------------
# 7. Warm-up curve (h_state[0] + emitted class over single window)
# ----------------------------------------------------------------------------
def fig_warmup_curve() -> None:
    # From memory_hafta8_streaming_sim.md (window 0, STANDING)
    # Arduino and MSP430 are bit-equivalent at 2 decimal places
    t       = [25,    50,    75,    100,    125,    128]
    h0      = [-0.72, -0.35, 0.46,  3.80,   11.39,  12.54]
    pred    = ["WALKING", "WALKING", "UPSTAIRS", "STANDING", "STANDING", "STANDING"]
    correct = [False, False, False, True, True, True]

    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.plot(t, h0, "o-", color="C0", label="$h_0(t)$ (both platforms)")

    # Annotate predictions
    for ti, hi, pi, ok in zip(t, h0, pred, correct):
        col = "C2" if ok else "C3"
        ax.annotate(pi, xy=(ti, hi), xytext=(ti + 1, hi + 0.8),
                    fontsize=6, color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=col))

    # Highlight warm-up region
    ax.axvspan(0, 100, alpha=0.07, color="C3",
               label="warm-up (~2 s)")
    ax.axvspan(100, 128, alpha=0.07, color="C2",
               label="stable")

    # 50 Hz → seconds on twin axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    s_ticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax2.set_xticks([s * 50 for s in s_ticks])
    ax2.set_xticklabels([f"{s:.1f}" for s in s_ticks])
    ax2.set_xlabel("Time (s)")

    ax.set_xlabel("Sample index (50 Hz)")
    ax.set_ylabel("$h_0$ value")
    ax.set_xlim(0, 135)
    ax.legend(loc="upper left", fontsize=6)
    save(fig, "warmup_curve.pdf")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    print("Generating paper figures into:", OUT)
    fig_saturation()
    fig_lowrank_seeds()
    fig_sparsity_curve()
    fig_quant_modes()
    fig_per_class_f1()
    fig_deploy_latency()
    fig_warmup_curve()
    print("Done.")


if __name__ == "__main__":
    main()
