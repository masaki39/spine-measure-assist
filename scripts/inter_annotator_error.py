"""
Inter-annotator error between train/dataset/original/ (5 points) and train/dataset/l1pa/ (6 points).
Metrics:
  - MRE (Mean Radial Error) in mm per landmark and overall
  - MAE (Mean Absolute Error) in degrees per angle
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent / "train/dataset"
ORIG = ROOT / "original"
L1PA = ROOT / "l1pa"

COMMON_LANDMARKS = ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH"]
COMMON_ANGLES = ["PI", "PT", "SS", "LL"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def ijk_to_mm(pt: dict, spacing: list) -> tuple[float, float]:
    return pt["i"] * spacing[0], pt["j"] * spacing[1]


def radial_error_mm(a: dict, b: dict, spacing: list) -> float:
    ax, ay = ijk_to_mm(a, spacing)
    bx, by = ijk_to_mm(b, spacing)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def main():
    cases = sorted(
        [f.stem.replace("_landmarks", "") for f in ORIG.glob("*_landmarks.json")],
        key=lambda x: int(x[1:]),
    )

    lm_errors: dict[str, list[float]] = {k: [] for k in COMMON_LANDMARKS}
    angle_errors: dict[str, list[float]] = {k: [] for k in COMMON_ANGLES}

    missing = 0
    for cid in cases:
        p2 = L1PA / f"{cid}_landmarks.json"
        if not p2.exists():
            missing += 1
            continue

        d1 = load_json(ORIG / f"{cid}_landmarks.json")
        d2 = load_json(p2)

        spacing = d1["metadata"]["spacing"]
        lm1 = d1["landmarks_ijk"]
        lm2 = d2["landmarks_ijk"]

        for lm in COMMON_LANDMARKS:
            if lm in lm1 and lm in lm2:
                lm_errors[lm].append(radial_error_mm(lm1[lm], lm2[lm], spacing))

        ang1 = d1.get("angles_deg", {})
        ang2 = d2.get("angles_deg", {})
        for ang in COMMON_ANGLES:
            if ang in ang1 and ang in ang2:
                angle_errors[ang].append(abs(ang1[ang] - ang2[ang]))

    n = len(cases) - missing
    print(f"Cases analyzed: {n}  (missing in l1pa: {missing})\n")

    print("=== Landmark MRE (mm) ===")
    all_errors: list[float] = []
    for lm in COMMON_LANDMARKS:
        errs = lm_errors[lm]
        if errs:
            mean = sum(errs) / len(errs)
            std = math.sqrt(sum((e - mean) ** 2 for e in errs) / len(errs))
            print(f"  {lm:<12}  MRE={mean:.2f}  SD={std:.2f}  max={max(errs):.2f}  (n={len(errs)})")
            all_errors.extend(errs)

    if all_errors:
        overall = sum(all_errors) / len(all_errors)
        std_all = math.sqrt(sum((e - overall) ** 2 for e in all_errors) / len(all_errors))
        print(f"  {'Overall':<12}  MRE={overall:.2f}  SD={std_all:.2f}  max={max(all_errors):.2f}")

    print("\n=== Angle MAE (degrees) ===")
    for ang in COMMON_ANGLES:
        errs = angle_errors[ang]
        if errs:
            mean = sum(errs) / len(errs)
            std = math.sqrt(sum((e - mean) ** 2 for e in errs) / len(errs))
            print(f"  {ang:<6}  MAE={mean:.2f}  SD={std:.2f}  max={max(errs):.2f}  (n={len(errs)})")


if __name__ == "__main__":
    main()
