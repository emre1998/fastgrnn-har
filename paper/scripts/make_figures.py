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
    8. realtime_budget.pdf  — sensor-in-loop: sensor+inference vs 20 ms deadline
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

    ax.set_xlabel("Target sparsity (%)")
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
# 8. Real-time budget: sensor + inference vs the 20 ms deadline (sensor in loop)
# ----------------------------------------------------------------------------
def fig_realtime_budget() -> None:
    data     = load_json("sensor_in_loop_matrix.json")
    runs     = data["runs"]
    deadline = data["deadline_ms"]

    CELLS   = ["GRU", "LSTM", "FastGRNN"]
    C_SENS  = "#8c8c8c"      # sensor acquisition segment
    C_PASS  = "#4c9f70"      # inference segment, deadline met
    C_FAIL  = "#d1615d"      # inference segment, deadline missed
    YMAX    = 38.0

    def pick(cell, lut, khz):
        return next(r for r in runs
                    if r["cell"] == cell and r["lut"] == lut and r["i2c_khz"] == khz)

    # x layout: LUT group at 0,1.2,2.4 — no-LUT group at 4.4,5.6,6.8
    xs   = [0, 1.2, 2.4, 4.4, 5.6, 6.8]
    grid = [(c, l) for l in (1, 0) for c in CELLS]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)

    for ax, khz, tag in zip(axes, (100, 10), ("a", "b")):
        ax.set_axisbelow(True)
        # infeasible region
        ax.axhspan(deadline, YMAX, color=C_FAIL, alpha=0.07, zorder=0, lw=0)
        ax.axhline(deadline, color=C_FAIL, linestyle="--", linewidth=1.0, zorder=1)

        for x, (cell, lut) in zip(xs, grid):
            r    = pick(cell, lut, khz)
            e2e  = r["e2e_ms"]
            cinf = C_PASS if r["realtime"] else C_FAIL

            if r["split_resolved"]:
                s = r["sensor_ms"]
                ax.bar(x, s,       0.72, color=C_SENS, edgecolor="none", zorder=2)
                ax.bar(x, e2e - s, 0.72, bottom=s, color=cinf,
                       edgecolor="none", zorder=2)
                star = ""
            else:
                # Sensor/inference split below the 1 ms timer tick: report the
                # measured end-to-end value only, do not invent a split.
                ax.bar(x, e2e, 0.72, color=cinf, edgecolor="none", zorder=2)
                star = "*"

            ax.text(x, e2e + 0.7, f"{e2e:.2f}{star}", ha="center", fontsize=6.5,
                    color=cinf, fontweight="bold" if r["realtime"] else "normal")

        # group separator + group labels (below the cell tick labels)
        ax.axvline(3.4, color="#d0d0d0", linewidth=0.7, zorder=1)
        for gx, gl in ((1.2, "LUT activations"), (5.6, "no LUT (libm expf/tanhf)")):
            ax.annotate(gl, xy=(gx, -0.145), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=7)

        ax.set_xticks(xs)
        ax.set_xticklabels([c for c, _ in grid], fontsize=6.5)
        ax.set_xlim(-0.9, 7.7)
        ax.set_ylim(0, YMAX)
        ax.set_title(f"({tag})  I$^2$C @ {khz} kHz", fontsize=8.5)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("End-to-end latency per sample (ms)")

    # (a) the only feasible corner of the whole design space
    axes[0].text(1.2, 17.6, "only real-time-feasible\nconfiguration",
                 ha="center", va="center", fontsize=6.5, color=C_PASS)

    # (b) the sensor floor that sinks every configuration
    axes[1].axhline(8.41, color="#4d4d4d", linestyle=":", linewidth=0.8, zorder=3)
    axes[1].text(3.4, 9.0, "8.4 ms", ha="center", va="bottom", fontsize=6.5,
                 color="#4d4d4d", zorder=4,
                 bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2))

    handles = [
        mpl.patches.Patch(color=C_SENS, label="Sensor acquisition (MPU6050, I$^2$C)"),
        mpl.patches.Patch(color=C_PASS, label="Inference — deadline met"),
        mpl.patches.Patch(color=C_FAIL, label="Inference — deadline missed"),
        mpl.lines.Line2D([], [], color=C_FAIL, linestyle="--", lw=1.0,
                         label=f"{deadline:.0f} ms deadline (50 Hz)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.19), fontsize=7)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "realtime_budget.pdf")


# ----------------------------------------------------------------------------
# 9. Ranking flip: equal capacity (H=16) vs equal deployment byte budget
# ----------------------------------------------------------------------------
def fig_ranking_flip() -> None:
    """Slopegraph — the crossing of the lines *is* the ranking reversal.

    Left  regime: equal hidden size H=16 (architecture quality).
    Right regime: equal deployment footprint (~283 nonzero params / 566 B).

    For GRU/LSTM the right-hand point is the BEST of the two compression routes
    (shrink-H at 200 epochs, or magnitude-pruned H=16), i.e. the strongest
    baseline we could build at the budget — not the most convenient one.
    """
    import glob

    TIER1_PAT = {
        "hapt":   "baseline_{c}_h16_s*_e120.json",
        "wisdm":  "baseline_wisdm_{c}_h16_s*_e120.json",
        "pamap2": "baseline_pamap2_{c}_h16_s*_e120.json",
    }
    DS_TITLE = {"hapt": "HAPT", "wisdm": "WISDM", "pamap2": "PAMAP2"}
    CELLS    = ["gru", "lstm", "fastgrnn"]
    CELL_LBL = {"gru": "GRU", "lstm": "LSTM", "fastgrnn": "FastGRNN"}
    COLOR    = {"gru": "#4c72b0", "lstm": "#8c8c8c", "fastgrnn": "#dd8452"}

    best = load_json("best_route_summary.json")

    def tier1(ds, c):
        files = sorted(glob.glob(str(EXP / TIER1_PAT[ds].format(c=c))))
        v = np.array([json.load(open(f, encoding="utf-8"))["test_macro_f1"]
                      for f in files])
        return float(v.mean()), float(v.std())

    def budget(ds, c):
        d = best[ds][c]
        if c == "fastgrnn":
            return d["mean"], d["std"]
        r = d["best_route"]                       # "shrink" or "pruned"
        return d["best_mean"], d[f"{r}_std"]

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.7))

    def spread(vals: dict, gap: float) -> dict:
        """Push near-identical label positions apart so they stay readable.
        Only the label moves; the plotted point stays at its true value."""
        order = sorted(vals, key=vals.get)
        out, prev = {}, None
        for c in order:
            y = vals[c] if prev is None else max(vals[c], prev + gap)
            out[c], prev = y, y
        return out

    for ax, ds in zip(axes, TIER1_PAT):
        pts = {c: (tier1(ds, c), budget(ds, c)) for c in CELLS}

        lo = min(min(m - s for m, s in p) for p in pts.values())
        hi = max(max(m + s for m, s in p) for p in pts.values())
        pad = 0.10 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

        for c in CELLS:
            (m0, s0), (m1, s1) = pts[c]
            ax.errorbar([0, 1], [m0, m1], yerr=[s0, s1], color=COLOR[c],
                        marker="o", markersize=4, linewidth=1.4,
                        elinewidth=0.7, capsize=2.0, alpha=0.95, zorder=3)

        gap  = 0.055 * (hi - lo + 2 * pad)
        left  = spread({c: pts[c][0][0] for c in CELLS}, gap)
        right = spread({c: pts[c][1][0] for c in CELLS}, gap)
        for c in CELLS:
            ax.annotate(f"{pts[c][0][0]:.3f}", (0, left[c]),
                        textcoords="offset points", xytext=(-6, 0),
                        ha="right", va="center", fontsize=6.5, color=COLOR[c])
            ax.annotate(f"{pts[c][1][0]:.3f}", (1, right[c]),
                        textcoords="offset points", xytext=(6, 0),
                        ha="left", va="center", fontsize=6.5, color=COLOR[c],
                        fontweight="bold" if best[ds]["winner"] == c else "normal")

        # ring the cell that wins at the byte budget
        w = best[ds]["winner"]
        ax.scatter([1], [pts[w][1][0]], s=70, facecolors="none",
                   edgecolors=COLOR[w], linewidths=1.1, zorder=4)

        ax.set_xlim(-0.62, 1.62)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["equal\ncapacity", "equal\nbyte budget"], fontsize=7)
        ax.set_title(DS_TITLE[ds], fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=6.5)

    axes[0].set_ylabel("Test macro F1 (5 seeds)")

    handles = [mpl.lines.Line2D([], [], color=COLOR[c], marker="o",
                                markersize=4, lw=1.4, label=CELL_LBL[c])
               for c in CELLS]
    handles.append(mpl.lines.Line2D([], [], color="#555555", marker="o",
                                    markerfacecolor="none", markersize=8,
                                    lw=0, label="winner at the byte budget"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.13), fontsize=7)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "ranking_flip.pdf")


# ----------------------------------------------------------------------------
# 10. Compiler -O insensitivity (the no-FPU mechanism)
# ----------------------------------------------------------------------------
def fig_opt_insensitivity() -> None:
    """36-run sweep: identical source, only the CCS optimization level changes.

    (a) absolute latency, y from zero — two flat bands separated by the LUT gap.
    (b) the same data as speedup relative to -O off, so the size of the compiler
        effect is stated explicitly rather than hidden by the absolute scale.
    """
    data   = load_json("opt_level_sweep.json")
    levels = data["levels"]
    lat    = data["latency_ms"]
    CELLS  = ["GRU", "LSTM", "FastGRNN"]
    COLOR  = {"GRU": "#4c72b0", "LSTM": "#8c8c8c", "FastGRNN": "#dd8452"}
    x      = np.arange(len(levels))
    plateau = levels.index(data["plateau_level"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W2, 2.7))

    # ---- (a) absolute -------------------------------------------------------
    for cell in CELLS:
        axA.plot(x, lat["lut"][cell], "-o", color=COLOR[cell], markersize=3.5,
                 label=f"{cell}, LUT")
        axA.plot(x, lat["no_lut"][cell], "--s", color=COLOR[cell], markersize=3.5,
                 markerfacecolor="none", label=f"{cell}, no LUT")

    axA.axhline(20.0, color="#d1615d", linestyle=":", linewidth=0.9)
    axA.annotate("20 ms budget", xy=(5.35, 20.0), fontsize=6, color="#d1615d",
                 ha="right", va="bottom")

    # the gap the compiler cannot close, vs the gap it can
    axA.annotate("", xy=(4.0, lat["no_lut"]["GRU"][4]), xytext=(4.0, lat["lut"]["GRU"][4]),
                 arrowprops=dict(arrowstyle="<->", lw=0.8, color="#333333"))
    axA.annotate("activation\nimplementation\n$-37$ to $-46\\%$",
                 xy=(4.15, 15.7), fontsize=6.2, color="#333333", va="center")

    axA.set_ylim(0, 31)
    axA.set_ylabel("Latency per recurrent step (ms)")
    axA.set_title("(a)  measured latency", fontsize=8.5)
    axA.legend(loc="lower left", ncol=2, fontsize=5.8, columnspacing=0.8,
               handlelength=1.6, handletextpad=0.4)

    # ---- (b) relative to -O off --------------------------------------------
    for cell in CELLS:
        for key, style, fill in (("lut", "-o", COLOR[cell]),
                                 ("no_lut", "--s", "none")):
            v = np.asarray(lat[key][cell], dtype=float)
            axB.plot(x, 100.0 * (v / v[0] - 1.0), style, color=COLOR[cell],
                     markersize=3.5, markerfacecolor=fill)

    axB.axhline(0.0, color="#999999", linewidth=0.8)
    axB.axvline(plateau, color="#4c9f70", linestyle="--", linewidth=0.9)
    axB.annotate("plateau at $-$O2\n(deployed: $-$O3)", xy=(plateau + 0.12, 1.1),
                 fontsize=6.2, color="#4c9f70", va="bottom")
    axB.annotate("no FPU, no hardware multiplier: every float MAC and\n"
                 "every $\\mathtt{expf}$/$\\mathtt{tanhf}$ is a precompiled soft-float library call",
                 xy=(5.3, -12.3), fontsize=6.0, color="#555555",
                 ha="right", va="bottom")
    axB.set_ylim(-13, 3)
    axB.set_ylabel("Change vs. $-$O off (%)")
    axB.set_title("(b)  what the compiler actually buys", fontsize=8.5)

    for ax in (axA, axB):
        ax.set_xticks(x)
        ax.set_xticklabels([f"$-$O{l}" if l != "off" else "$-$O off"
                            for l in levels], fontsize=6.5)
        ax.set_xlabel("CCS optimization level")
        ax.set_xlim(-0.35, 5.35)
        ax.grid(axis="x", visible=False)

    fig.subplots_adjust(wspace=0.30)
    save(fig, "opt_insensitivity.pdf")


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
    fig_realtime_budget()
    fig_ranking_flip()
    fig_opt_insensitivity()
    print("Done.")


if __name__ == "__main__":
    main()
