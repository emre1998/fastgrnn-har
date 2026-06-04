"""
Pre-compute 256-entry look-up tables for sigmoid and tanh and write them
to a C header.

The index maps an input from the range [-8, +8] to one of 256 buckets.

Output: arduino/fastgrnn_har/lut.h + msp/.../lut.h
"""

import math
from pathlib import Path

LUT_SIZE = 256
INPUT_MIN = -8.0
INPUT_MAX = 8.0
INPUT_RANGE = INPUT_MAX - INPUT_MIN  # 16.0
BUCKET_WIDTH = INPUT_RANGE / LUT_SIZE  # 0.0625

# Sample each LUT entry at the CENTER of its bucket (no extra interpolation needed)
sigmoid_lut = [1.0 / (1.0 + math.exp(-(INPUT_MIN + (i + 0.5) * BUCKET_WIDTH))) for i in range(LUT_SIZE)]
tanh_lut    = [math.tanh(INPUT_MIN + (i + 0.5) * BUCKET_WIDTH)              for i in range(LUT_SIZE)]

def fmt_array(arr, name, indent=2):
    lines = []
    indent_str = " " * indent
    line = indent_str
    for i, v in enumerate(arr):
        s = f"{v:+.6f}f, "
        if len(line) + len(s) > 90:
            lines.append(line.rstrip())
            line = indent_str
        line += s
    if line.strip():
        lines.append(line.rstrip(", "))
    return "\n".join(lines)

header = f"""/*
 * lut.h - 256-entry float LUTs for sigmoid and tanh
 *
 * Input range: [{INPUT_MIN}, {INPUT_MAX}) - {INPUT_RANGE} wide, {LUT_SIZE} buckets
 * Bucket width: {BUCKET_WIDTH:.6f}
 * Index computation: idx = (int)((x - INPUT_MIN) * LUT_SCALE)
 *   LUT_SCALE = 1 / BUCKET_WIDTH = {1.0/BUCKET_WIDTH:.4f}
 *
 * Saturating: x < INPUT_MIN -> idx = 0, x >= INPUT_MAX -> idx = LUT_SIZE - 1
 *
 * Footprint: 256 * 4 bytes * 2 tables = 2 KB Flash (PROGMEM/const)
 * Speed: O(1) lookup instead of expf/tanhf (~3-5x speedup on AVR,
 *        ~30x on multiplier-less MSP430).
 */

#ifndef LUT_H
#define LUT_H

#include <stdint.h>

#ifdef __AVR__
  #include <avr/pgmspace.h>
#else
  #ifndef PROGMEM
    #define PROGMEM
  #endif
#endif

#define LUT_SIZE         {LUT_SIZE}
#define LUT_INPUT_MIN    ({INPUT_MIN:.1f}f)
#define LUT_INPUT_MAX    ({INPUT_MAX:.1f}f)
#define LUT_INPUT_SCALE  ({1.0/BUCKET_WIDTH:.6f}f)   // 1 / bucket_width

const float SIGMOID_LUT[LUT_SIZE] PROGMEM = {{
{fmt_array(sigmoid_lut, "SIGMOID_LUT")}
}};

const float TANH_LUT[LUT_SIZE] PROGMEM = {{
{fmt_array(tanh_lut, "TANH_LUT")}
}};

#endif // LUT_H
"""

paths = [
    "fastgrnn_har/lut.h",
    "../msp/fastgrnn_har_msp/lut.h",
    "../msp/ccs_fastgrnn_har/lut.h",
]
for p in paths:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(header)
    print(f"Saved: {p} ({Path(p).stat().st_size} bytes)")

print(f"\nLUT spec: 256 entries, [{INPUT_MIN}, {INPUT_MAX}], bucket = {BUCKET_WIDTH:.6f}")
print(f"Expected runtime footprint: 2 KB Flash (sigmoid + tanh, float32)")
