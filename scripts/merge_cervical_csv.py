"""
Merge cervical angle CSVs (lateral / flexion / extension) into one summary CSV.

Usage:
  uv run python scripts/merge_cervical_csv.py
  uv run python scripts/merge_cervical_csv.py --omuro /path/to/omuro --output summary.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import pydicom


def build_study_map(dicom_dir: Path) -> dict[str, str]:
    """Return {study_instance_uid: case_id} for DICOMs in dicom_dir."""
    study_map: dict[str, str] = {}
    for dcm in sorted(f for f in dicom_dir.glob("*.dcm") if not f.name.startswith("._")):
        ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
        uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
        if uid:
            study_map[uid] = dcm.stem
    return study_map


def read_csv(dataset_dir: Path) -> dict[str, dict]:
    """Return {case_id: row_dict} from angles.csv."""
    result: dict[str, dict] = {}
    csv_path = dataset_dir / "dataset" / "angles.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["case_id"]] = row
    return result


def safe_float(val: str) -> str:
    try:
        return str(round(float(val), 2))
    except (ValueError, TypeError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge cervical angle CSVs into a summary CSV")
    parser.add_argument("--omuro", default="/Volumes/T7 Shield/dicom/omuro",
                        help="Root directory containing lateral/flexion/extension subdirectories")
    parser.add_argument("--output", default="/Users/masaki/Downloads/cervical_summary.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    omuro = Path(args.omuro)
    lateral_dir = omuro / "頸椎XP（側面)"
    flexion_dir = omuro / "頸椎XP（前屈）"
    extension_dir = omuro / "頸椎XP（後屈）"
    output = Path(args.output)

    print("Building study maps...")
    lat_study = build_study_map(lateral_dir)
    flex_study = build_study_map(flexion_dir)
    ext_study = build_study_map(extension_dir)

    print("Reading CSVs...")
    lat_data = read_csv(lateral_dir)
    flex_data = read_csv(flexion_dir)
    ext_data = read_csv(extension_dir)

    # studies present in all three views
    all_uids = sorted(set(lat_study) & set(flex_study) & set(ext_study))
    missing = (set(lat_study) | set(flex_study) | set(ext_study)) - set(all_uids)
    print(f"Matched studies: {len(all_uids)}")
    if missing:
        print(f"  Warning: {len(missing)} studies missing from some view (skipped)", file=sys.stderr)

    fieldnames = ["study_uid_suffix", "cl", "cl_flex", "cl_ext", "rom", "t1s", "c2_7sva"]
    rows = []
    for uid in all_uids:
        lat_row = lat_data.get(lat_study[uid], {})
        flex_row = flex_data.get(flex_study[uid], {})
        ext_row = ext_data.get(ext_study[uid], {})

        cl = safe_float(lat_row.get("C2C7_angle", ""))
        cl_flex = safe_float(flex_row.get("C2C7_angle", ""))
        cl_ext = safe_float(ext_row.get("C2C7_angle", ""))
        t1s = safe_float(lat_row.get("T1S", ""))
        sva = safe_float(lat_row.get("C2C7_SVA", ""))

        try:
            rom = str(round(float(cl_ext) - float(cl_flex), 2))
        except (ValueError, TypeError):
            rom = ""

        rows.append({
            "study_uid_suffix": uid[-12:],
            "cl": cl,
            "cl_flex": cl_flex,
            "cl_ext": cl_ext,
            "rom": rom,
            "t1s": t1s,
            "c2_7sva": sva,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {output}")


if __name__ == "__main__":
    main()
