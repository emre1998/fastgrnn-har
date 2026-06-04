"""
Convert test_vectors.json into a C header so we can validate inference
on the Arduino without the sensor attached.

Output: arduino/fastgrnn_har/test_data.h
"""

import json
from pathlib import Path

vectors = json.loads(Path("test_vectors.json").read_text())

# Embed just two samples (Flash budget: 2 * 128 * 3 * 4 = 3 KB)
N = 2
samples = vectors[:N]

lines = [
    "/* test_data.h - inference validation without a sensor",
    " * 2 test windows + their expected class indices",
    " * PORTABLE: AVR (Arduino) and MSP430 (Energia)",
    " */",
    "#ifndef TEST_DATA_H",
    "#define TEST_DATA_H",
    "",
    "#include <stdint.h>",
    "",
    "#ifdef __AVR__",
    "  #include <avr/pgmspace.h>",
    "#else",
    "  #ifndef PROGMEM",
    "    #define PROGMEM",
    "  #endif",
    "#endif",
    "",
    f"#define N_TEST_SAMPLES {N}",
    "",
    "// Expected class indices (0-5)",
    "const uint8_t TEST_EXPECTED[N_TEST_SAMPLES] = {",
    "  " + ", ".join(str(s["c_pred"]) for s in samples),
    "};",
    "",
    "const uint8_t TEST_TRUE[N_TEST_SAMPLES] = {",
    "  " + ", ".join(str(s["true_label"]) for s in samples),
    "};",
    "",
    "// Test windows - PROGMEM (Flash)",
]

for idx, s in enumerate(samples):
    window = s["window"]  # (128, 3) list
    lines.append(f"const float TEST_WINDOW_{idx}[128][3] PROGMEM = {{")
    for row in window:
        lines.append(f"  {{ {row[0]:+.6f}f, {row[1]:+.6f}f, {row[2]:+.6f}f }},")
    lines.append("};")
    lines.append("")

lines.append("// Pointer array so the main loop can pick a window by index")
lines.append("const float (* const TEST_WINDOWS[N_TEST_SAMPLES])[3] PROGMEM = {")
for idx in range(N):
    lines.append(f"  TEST_WINDOW_{idx},")
lines.append("};")
lines.append("")
lines.append("#endif // TEST_DATA_H")

content = "\n".join(lines)
for out in ("fastgrnn_har/test_data.h",
            "../msp/fastgrnn_har_msp/test_data.h"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(content)
    print(f"Saved: {out}")
print(f"  {N} test windows embedded")
print(f"  Estimated Flash usage: {N * 128 * 3 * 4} bytes = {N * 128 * 3 * 4 / 1024:.2f} KB")
