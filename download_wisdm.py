"""
Download and unpack the WISDM Actitracker v1.1 dataset.

Tri-axial phone accelerometer, 20 Hz, 36 users, 6 activities
(Walking, Jogging, Upstairs, Downstairs, Sitting, Standing).

Usage:  python download_wisdm.py
"""
import ssl
import tarfile
import urllib.request
from pathlib import Path

import certifi

URL = ("https://www.cis.fordham.edu/wisdm/includes/datasets/latest/"
       "WISDM_ar_latest.tar.gz")
DATA_DIR = Path("data")
TAR_PATH = DATA_DIR / "wisdm_ar_latest.tar.gz"
EXTRACT_DIR = DATA_DIR / "wisdm"

DATA_DIR.mkdir(exist_ok=True)
ctx = ssl.create_default_context(cafile=certifi.where())

if not TAR_PATH.exists():
    print(f"Downloading: {URL}  (~11 MB)")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp, open(TAR_PATH, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print("\nDownload complete.")
else:
    print(f"Already present: {TAR_PATH}")

EXTRACT_DIR.mkdir(exist_ok=True)
print(f"Extracting to: {EXTRACT_DIR}")
with tarfile.open(TAR_PATH, "r:gz") as tf:
    tf.extractall(EXTRACT_DIR)

print("\nContents:")
for p in sorted(EXTRACT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(EXTRACT_DIR)}  ({p.stat().st_size/1e6:.2f} MB)")
