"""
B1 verification: compile the ACTUAL C firmware on the host (g++) and check it
reproduces the PyTorch/NumPy predictions bit-exactly on the test vectors.

Stronger than the NumPy C-equivalent check in build_deploy_firmware.py: this
compiles gru.cpp / lstm.cpp themselves and runs the compiled binary.

Usage:  python verify_firmware.py --cell gru
        python verify_firmware.py --cell lstm
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cell", choices=["gru", "lstm"], required=True)
args = parser.parse_args()

d = Path(f"arduino/{args.cell}_har")
tv = json.loads((d / "test_vectors.json").read_text())
N = len(tv)
T = len(tv[0]["window"])
D = len(tv[0]["window"][0])
C = len(tv[0]["logits"])

# --- emit test_data.h (windows + expected predictions) ---
def fmt_window(w):
    rows = ",\n    ".join("{ " + ", ".join(f"{v:.8f}f" for v in row) + " }" for row in w)
    return "  {\n    " + rows + "\n  }"
windows = ",\n".join(fmt_window(s["window"]) for s in tv)
expected = ", ".join(str(s["c_pred"]) for s in tv)
(d / "test_data.h").write_text(
    f"#ifndef TEST_DATA_H\n#define TEST_DATA_H\n#include \"model_weights.h\"\n"
    f"#define N_TEST {N}\n"
    f"const float TEST_X[N_TEST][WINDOW_T][INPUT_DIM] = {{\n{windows}\n}};\n"
    f"const int TEST_EXPECTED[N_TEST] = {{ {expected} }};\n#endif\n"
)

# --- emit host_main.cpp ---
cell = args.cell
(d / "host_main.cpp").write_text(f"""#include <cstdio>
#include "{cell}.h"
#include "test_data.h"
int main() {{
  for (int n = 0; n < N_TEST; n++) {{
    unsigned char p = {cell}_classify_window(TEST_X[n]);
    const float* lg = {cell}_get_logits();
    printf("%d", (int)p);
    for (int c = 0; c < NUM_CLASSES; c++) printf(" %.6f", lg[c]);
    printf("\\n");
  }}
  return 0;
}}
""")

# --- compile with g++ (deployed config: USE_LUT=1) ---
exe = d / ("host_test.exe" if sys.platform == "win32" else "host_test")
cmd = ["g++", "-O2", "-DUSE_LUT=1", f"-I{d}", str(d / "host_main.cpp"),
       str(d / f"{cell}.cpp"), "-o", str(exe)]
print("Compiling:", " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("COMPILE FAILED:\n", r.stderr); sys.exit(1)

# --- run and compare ---
out = subprocess.run([str(exe)], capture_output=True, text=True).stdout.strip().splitlines()
print(f"\n=== {cell.upper()} compiled-C vs reference ({N} vectors) ===")
ok = 0
for n, line in enumerate(out):
    parts = line.split()
    cpred = int(parts[0]); clogits = [float(x) for x in parts[1:]]
    exp = tv[n]["c_pred"]; explog = tv[n]["logits"]
    dmax = max(abs(a - b) for a, b in zip(clogits, explog))
    match = "OK" if cpred == exp else "MISMATCH"
    if cpred == exp: ok += 1
    print(f"  vec {n}: C_pred={cpred} expected={exp} [{match}]  logit_maxdiff={dmax:.6f}")
print(f"\nPrediction agreement: {ok}/{N} ({100*ok/N:.0f}%)")
print("BIT-EXACT OK" if ok == N else "MISMATCH -- firmware logic differs")
