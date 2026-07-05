"""
Memory-budget accounting (K3, analytical) -- no training.

A4b: equal-brain (H=16) footprint, dense FP32 -> deployed (Q15+compression).
A4 : two-part MCU budget per cell -> Flash (weights) + SRAM (streaming working set),
     checked against MSP430 (512 B SRAM / 16 KB Flash) and Arduino Uno (2 KB / 32 KB).

Streaming inference is what makes it fit: storing the whole window (T*D int16) would
exceed 512 B; streaming keeps only the recurrent state + one input sample + scratch.
The SRAM numbers are analytical estimates to be confirmed by the Faz B firmware .map.
"""
import json
from pathlib import Path

D = 3            # tri-axial accelerometer input
NUM_CLASSES = {"hapt": 6, "wisdm": 6, "pamap2": 12}
T = {"hapt": 128, "wisdm": 50, "pamap2": 128}   # window length (samples)
MSP_SRAM, MSP_FLASH = 512, 16 * 1024
UNO_SRAM, UNO_FLASH = 2 * 1024, 32 * 1024


def params_dense(kind, H, C):
    if kind == "fastgrnn":                      # vanilla full-rank cell
        cell = H * D + H * H + 2 * H + 2        # W + U + b_z + b_h + zeta + nu
    elif kind == "gru":
        cell = 3 * (H * D + H * H + 2 * H)
    else:                                       # lstm
        cell = 4 * (H * D + H * H + 2 * H)
    head = H * C + C
    return cell + head


def sram_working_set(kind, H, C):
    """Streaming inference SRAM (bytes, int16=2B). Persistent state + one scratch gate + input + logits."""
    x = D                                       # current input sample
    logits = C                                  # classifier output accumulators
    if kind == "fastgrnn":
        state = H                               # h_t
        scratch = 2 * H                         # pre-activation + candidate (z, h_tilde reuse)
    elif kind == "gru":
        state = H                               # h_t
        scratch = 3 * H                         # r, z, n gates
    else:                                       # lstm
        state = 2 * H                           # h_t + c_t
        scratch = 4 * H                         # i, f, g, o gates
    return (state + scratch + x + logits) * 2


def fit(label, sram, flash):
    ms = "OK" if sram <= MSP_SRAM else "OVER"
    mf = "OK" if flash <= MSP_FLASH else "OVER"
    us = "OK" if sram <= UNO_SRAM else "OVER"
    print(f"  {label:22s} SRAM {sram:5d} B (MSP {ms} / Uno {us}) | Flash {flash:5d} B (MSP {mf})")


def main():
    print("=== A4b: EQUAL-BRAIN (H=16) FOOTPRINT: dense FP32 -> compressed (Q15), same H ===")
    dv = json.load(open("experiments/deploy_budget_summary.json"))
    print(f"{'cell':9s} {'dense params':>12s} {'dense FP32 B':>13s} {'compressed B':>13s} {'ratio':>6s}")
    for kind in ("fastgrnn", "gru", "lstm"):
        p = params_dense(kind, 16, 6)                       # HAPT, 6 classes, H=16
        if kind == "fastgrnn":
            comp = dv["hapt"]["fastgrnn"]["total_nonzero"] * 2   # low-rank+IHT+Q15 = 566 B (still H=16)
        else:
            comp = p * 2                                    # GRU/LSTM natural compression at H16 = Q15 only
        print(f"{kind:9s} {p:12d} {p*4:13d} {comp:13d} {p*4/comp:5.1f}x")
    print("  At EQUAL brain (H=16): FastGRNN compresses to 566 B (low-rank+IHT+Q15);")
    print("  GRU/LSTM only to Q15 (no natural low-rank/sparse) -> 4-5x larger. THIS is why,")
    print("  at a fixed byte budget, GRU/LSTM must SHRINK H while FastGRNN keeps it.")

    print("\n=== A4: TWO-PART BUDGET per cell, streaming (H=16 worst case) ===")
    print(f"MSP430: {MSP_SRAM} B SRAM / {MSP_FLASH} B Flash | Uno: {UNO_SRAM} B SRAM / {UNO_FLASH} B Flash")
    for ds in ("hapt", "wisdm", "pamap2"):
        C = NUM_CLASSES[ds]
        win = T[ds] * D * 2
        print(f"\n{ds.upper()}  (store-whole-window would need {win} B SRAM -> "
              f"{'OVER' if win > MSP_SRAM else 'OK'} on MSP430; streaming avoids this)")
        for kind in ("fastgrnn", "gru", "lstm"):
            sram = sram_working_set(kind, 16, C)
            flash = dv[ds][kind]["total_nonzero"] * 2       # weights in Flash (Q15)
            fit(kind, sram, flash)

    print("\nNote: SRAM = streaming working set (state + scratch + input + logits), int16.")
    print("Weights live in Flash, not SRAM. All cells fit MSP430 512 B SRAM via streaming.")


if __name__ == "__main__":
    main()
