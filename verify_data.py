"""
Verify that the processed datasets match the ones the published results were
produced from.

The .npz files are too large to commit and are rebuilt by download_*.py +
build_*.py. Those builders are deterministic, but nothing checked that a rebuild
actually reproduced the same bytes -- so a silently different dataset would have
looked exactly like results failing to reproduce. This closes that gap.

  python verify_data.py            # check against the committed manifest
  python verify_data.py --write    # regenerate the manifest (maintainers only)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("data/processed")
MANIFEST = Path("data/processed_manifest.json")
DATASETS = ["hapt", "wisdm", "pamap2"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path) -> dict:
    """Shape/label summary as well as the hash: a mismatching hash with matching
    shapes points at a different build, not a different dataset."""
    d = np.load(path, allow_pickle=True)
    return {
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "X_train": list(d["X_train"].shape),
        "X_test": list(d["X_test"].shape),
        "classes": int(d["y_train"].max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="regenerate the manifest from the local files")
    args = ap.parse_args()

    present = {ds: DATA / f"{ds}_windows.npz" for ds in DATASETS}
    missing = [ds for ds, p in present.items() if not p.exists()]

    if args.write:
        if missing:
            print(f"cannot write manifest, missing: {', '.join(missing)}")
            return 1
        man = {ds: describe(p) for ds, p in present.items()}
        MANIFEST.write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(f"wrote {MANIFEST}")
        for ds, m in man.items():
            print(f"  {ds:8s} {m['sha256'][:16]}  {m['X_train'][0]:6d} train windows")
        return 0

    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST} -- run with --write to create one")
        return 1
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))

    bad = 0
    for ds in DATASETS:
        path = present[ds]
        if not path.exists():
            print(f"  MISSING  {ds:8s} -- run download_{ds}.py then build_{ds}.py")
            bad += 1
            continue
        want, got = man.get(ds), describe(path)
        if want is None:
            print(f"  UNKNOWN  {ds:8s} not in manifest")
            bad += 1
        elif want["sha256"] == got["sha256"]:
            print(f"  OK       {ds:8s} {got['sha256'][:16]}")
        else:
            bad += 1
            print(f"  MISMATCH {ds:8s}")
            print(f"           expected {want['sha256'][:16]}  {want['X_train']}")
            print(f"           got      {got['sha256'][:16]}  {got['X_train']}")
            if want["X_train"] == got["X_train"]:
                print("           shapes agree, so this is a different build of the "
                      "same source, not a different dataset")

    print()
    if bad:
        print(f"{bad} dataset(s) do not match the published results. Numbers "
              f"produced from them are not comparable to the committed ones.")
    else:
        print("all datasets match the manifest.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
