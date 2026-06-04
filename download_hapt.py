"""
Download and unpack the UCI HAPT dataset.

Usage:  python download_hapt.py
"""

import os
import urllib.request
import zipfile
import ssl
import certifi
from pathlib import Path

URL = ("https://archive.ics.uci.edu/static/public/341/"
       "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip")

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "hapt.zip"
EXTRACT_DIR = DATA_DIR / "hapt"

DATA_DIR.mkdir(exist_ok=True)

if not ZIP_PATH.exists():
    print(f"Downloading: {URL}")
    print("(may take a few minutes, ~60 MB)")
    # NOTE: UCI's certificate has expired. Since this is a public dataset we
    # bypass verification here as a pragmatic workaround. Do not do this in
    # production code; remove the bypass once UCI fixes the certificate.
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(URL, context=ctx) as resp, open(ZIP_PATH, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print("\nDownload complete.")
else:
    print(f"Already present: {ZIP_PATH}")

print(f"\nExtracting to: {EXTRACT_DIR}")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(EXTRACT_DIR)

print("\nTop-level contents:")
for entry in sorted(EXTRACT_DIR.iterdir()):
    print("  ", entry.name)

# Most UCI archives are nested zips - unpack one more level if needed
inner_zips = list(EXTRACT_DIR.rglob("*.zip"))
if inner_zips:
    print("\nUnpacking inner zips:")
    for iz in inner_zips:
        target = iz.with_suffix("")
        print(f"  {iz.name} -> {target.name}/")
        with zipfile.ZipFile(iz, "r") as zf:
            zf.extractall(target)

# Final sanity check: locate RawData/ and activity_labels.txt
print("\nLooking for the RawData/ directory...")
raw_dirs = list(EXTRACT_DIR.rglob("RawData"))
for rd in raw_dirs:
    files = sorted(rd.iterdir())
    print(f"  Found: {rd}")
    print(f"  {len(files)} files inside; first five:")
    for f in files[:5]:
        print(f"    {f.name}")

print("\nLooking for activity_labels.txt...")
for p in EXTRACT_DIR.rglob("activity_labels.txt"):
    print(f"  {p}")
    with open(p) as f:
        print("    Contents:")
        for line in f:
            print(f"      {line.rstrip()}")
