"""
Download and unpack the UCI PAMAP2 Physical Activity Monitoring dataset.

100 Hz IMUs on 3 body locations + HR; 9 subjects, 12 protocol activities.
We use the hand-IMU tri-axial accelerometer (16g range) to mirror the
single tri-axial accelerometer of HAPT/WISDM.

Usage:  python download_pamap2.py
"""
import ssl
import zipfile
import urllib.request
from pathlib import Path

import certifi

URL = ("https://archive.ics.uci.edu/static/public/231/"
       "pamap2+physical+activity+monitoring.zip")
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "pamap2.zip"
EXTRACT_DIR = DATA_DIR / "pamap2"

DATA_DIR.mkdir(exist_ok=True)
ctx = ssl.create_default_context(cafile=certifi.where())

if not ZIP_PATH.exists():
    print(f"Downloading: {URL}  (large, ~650 MB)")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=600) as resp, open(ZIP_PATH, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            mb = done / 1e6
            if total:
                print(f"\r  {mb:.1f}/{total/1e6:.1f} MB", end="", flush=True)
            elif done % (1024 * 1024 * 20) < 1024 * 256:
                print(f"\r  {mb:.1f} MB", end="", flush=True)
    print("\nDownload complete.")
else:
    print(f"Already present: {ZIP_PATH}")

EXTRACT_DIR.mkdir(exist_ok=True)
print(f"Extracting to: {EXTRACT_DIR}")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(EXTRACT_DIR)

# PAMAP2 ships a nested zip
for iz in EXTRACT_DIR.rglob("*.zip"):
    print(f"Unpacking inner zip: {iz.name}")
    with zipfile.ZipFile(iz, "r") as zf:
        zf.extractall(iz.with_suffix(""))

print("\nProtocol .dat files found:")
for p in sorted(EXTRACT_DIR.rglob("subject*.dat")):
    print(f"  {p.relative_to(EXTRACT_DIR)}  ({p.stat().st_size/1e6:.1f} MB)")
