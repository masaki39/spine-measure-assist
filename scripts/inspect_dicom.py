"""
DICOM inspection CLI tool.
Usage:
  uv run --with pydicom python scripts/inspect_dicom.py <dicom_file>
  uv run --with pydicom python scripts/inspect_dicom.py --scan data/
"""

import argparse
import sys
from pathlib import Path


def inspect_file(path: Path):
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    print(f"\n=== {path} ===")

    for tag in ["Modality", "ImageType", "Rows", "Columns", "PixelSpacing", "SeriesDescription", "StudyDescription"]:
        val = getattr(ds, tag, None)
        if val is not None:
            print(f"  {tag}: {val}")

    if hasattr(ds, "PixelData"):
        arr = ds.pixel_array
        print(f"  pixel_array shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"  min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
    else:
        print("  (no pixel data)")


def scan_dir(root: Path):
    import pydicom

    for p in sorted(root.rglob("*")):
        if p.is_file():
            try:
                inspect_file(p)
            except Exception as e:
                print(f"  [skip] {p}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect DICOM file(s)")
    parser.add_argument("path", help="DICOM file or directory to scan")
    parser.add_argument("--scan", action="store_true", help="Recursively scan directory")
    args = parser.parse_args()

    p = Path(args.path)
    if args.scan or p.is_dir():
        scan_dir(p)
    else:
        inspect_file(p)


if __name__ == "__main__":
    main()
