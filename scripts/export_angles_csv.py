"""Export angle data from landmark JSON files to CSV."""
import csv
import json
import sys
from pathlib import Path

ANGLE_KEYS = ["C2C7_angle", "T1S", "C2C7_SVA"]


def main(dataset_dir: str) -> None:
    dataset_path = Path(dataset_dir)
    # exclude macOS resource fork files (._*)
    json_files = sorted(
        f for f in dataset_path.glob("*_landmarks.json")
        if not f.name.startswith("._")
    )

    if not json_files:
        print(f"No landmark JSON files found in {dataset_path}", file=sys.stderr)
        sys.exit(1)

    output_path = dataset_path / "angles.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id"] + ANGLE_KEYS)
        writer.writeheader()
        for jf in json_files:
            with open(jf, "rb") as jfile:
                data = json.loads(jfile.read().decode("utf-8", errors="replace"))
            angles = data.get("angles_deg", {})
            row = {"case_id": data.get("case_id", jf.stem.replace("_landmarks", ""))}
            for key in ANGLE_KEYS:
                row[key] = angles.get(key, "")
            writer.writerow(row)

    print(f"Wrote {len(json_files)} rows → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_angles_csv.py <dataset_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
