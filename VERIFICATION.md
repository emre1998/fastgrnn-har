# Verification of two reviewer-flagged concerns (#2 warm-up, #5 bit-equivalence)

Both were investigated against the actual firmware and results, not argued from the
abstract. Neither is a defect; each resolves to a precise statement the writing
phase should use. Framing is deferred to August by decision — this file is the
answer that phase will draw on.

## #2 — Warm-up latency is not added latency, but it does bound the window length

**What the firmware does.** `msp/ccs_{cell}_production/main.cpp` resets the hidden
state at the start of every window (`gru_reset()`), streams `WINDOW_T = 128` samples,
and reads a class **only at the end of the window**:

```c
while (1) {
    gru_reset();
    for (t = 0; t < 128; t++) { gru_step(sample); pace_to_50Hz(); }
    cls = gru_predict();           // read once, at sample 128
}
```

So windows are non-overlapping and each classification is emitted after a full
window.

**The warm-up numbers** (`experiments/warmup_distribution.json`, deployed seed-0
model, 100 windows): the hidden state settles to the stable class at a median of
**74 samples (1.48 s)**, q3 86, **worst case 125**, min 0.

**The resolution.** The class is read at sample 128, and the worst observed warm-up
is 125, so in all 100 windows the emitted class sits on a **settled** state. The
intra-window settling is therefore **invisible to the output** — the device does not
emit 1.48 s of wrong labels. A reader who takes the 1.48 s (or 2.5 s) figure as
end-user response lag — as the alphaxiv summary did for fall detection — is misreading
the design: user-facing output is one class per 2.56 s window, each on a warm state.

**But the honest caveat, which is the real finding:** the margin is only **3 samples**
(128 − 125). The window length is effectively **lower-bounded by the worst-case
warm-up** — shorten the window, or feed an input that settles more slowly, and the
classification point could fall inside the warm-up. So warm-up is a genuine design
constraint on window length; it is not an added latency in the deployed configuration.

This is narrower than the hoped-for "one-time boot transient" reframe (the firmware
does reset per window, so warm-up recurs), and stronger than the criticism assumed
(no wrong-output latency). State it exactly: *warm-up does not degrade the
once-per-window output because the window exceeds the worst-case settling time, with
a 3-sample margin that lower-bounds the window length.*

## #5 — "Bit-equivalent" names two different claims; the seed dependence is the harmless one

The word "bit-equivalent" in v1 slides between two measurements. Separating them
dissolves the concern.

**Claim A — cross-platform inference.** One portable C source
(`model_weights.h` + `{cell}.cpp`, `#ifdef __AVR__` for PROGMEM) compiles on both the
8-bit AVR and the 16-bit MSP430 and produces matching output. This is a property of
identical integer Q15 arithmetic under two compilers, so it is **seed-independent by
construction**. It is demonstrated on the **deployed seed-0 model**; it does not need
per-seed repetition, because nothing in it depends on the weight values.

**Claim B — quantization fidelity.** FP32 (PyTorch) vs Q15 (`agreement_5seed.json`)
argmax agreement, per seed:

| seed | agree | of 3399 |
|---|---|---|
| 0 | 99.97% | 3398 |
| 1 | 99.91% | 3396 |
| 2 | 99.97% | 3398 |
| 3 | 100.0% | 3399 |
| 4 | 99.97% | 3398 |

This is **seed-dependent by nature**: different trained weights sit at different
distances from the class boundaries, so quantization flips a different handful of
borderline windows (1–3 out of 3399). Sub-100% here is the **expected behaviour of
quantization**, and 99.9%+ is a **strong** result, not a weakness.

**The resolution.** The critique "bit-equivalence is only exact at seed 0" cites
Claim B's seed variation as if it undercut Claim A. It does not — they are different
measurements. The fix is entirely in the writing:

- Use distinct terms: **"cross-platform bit-identical inference"** (Claim A) versus
  **"Q15–FP32 argmax agreement"** or **"quantization fidelity"** (Claim B). Never let
  one phrase carry both.
- State Claim B's sub-100% as the expected, small cost of quantization (1–3 windows of
  3399), and 99.9%+ as strong.
- State Claim A is demonstrated on the deployed configuration and is structural, so it
  carries no per-seed dependence.

No experiment is needed for either. Both are closed as writing-phase clarifications.
