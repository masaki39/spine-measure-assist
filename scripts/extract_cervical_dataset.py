"""
SQLite DB から頚椎側面画像を抽出し、CervicalMeasureAssist用データセットを生成する。

使用法:
  uv run python scripts/extract_cervical_dataset.py
  uv run python scripts/extract_cervical_dataset.py --limit 3   # テスト用
  uv run python scripts/extract_cervical_dataset.py --dry-run   # 処理内容の確認のみ
  uv run python scripts/extract_cervical_dataset.py --region CERVICAL_SPINE  # region値を指定

出力 (/Volumes/T7 Shield/dicom/kch-organized/dataset/cervical/):
  K001_image.npy         -- 2D float32 画像 (H, W)
  K001_landmarks.json    -- アノテーションテンプレート (全ランドマーク null)

注意: DBの region フィールド値はDB固有。
  最初に --dry-run で対象件数を確認し、0件の場合は --region で正しい値を指定すること。
  候補例: 'CERVICAL_SPINE', 'C_SPINE', 'WHOLE_SPINE' など。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.dataset_cervical import LANDMARK_ORDER

SSD_BASE = "/Volumes/T7 Shield/dicom/kch-organized"
DEFAULT_DB = f"{SSD_BASE}/dicom_index.db"
DEFAULT_BASE = SSD_BASE
DEFAULT_OUT = Path(SSD_BASE) / "dataset" / "cervical"
DEFAULT_REGION = "CERVICAL_SPINE"


def make_cervical_landmark_template() -> dict:
    """全8頚椎ランドマークを null で初期化したテンプレートを返す。"""
    return {k: {"i": None, "j": None, "k": 0} for k in LANDMARK_ORDER}


def _load_dicom_pixel(dcm_path: Path) -> tuple[np.ndarray, list]:
    """DICOMを読み込み (H, W) float32 配列と spacing を返す。"""
    import pydicom

    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D pixel array, got shape {arr.shape}")
    spacing = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    return arr, spacing


def _build_metadata(spacing: list[float]) -> dict:
    sp_i, sp_j = spacing[0], spacing[1]
    return {
        "spacing": [sp_i, sp_j, 1.0],
        "ijk_to_ras": [
            [-sp_i, 0.0, 0.0],
            [0.0, -sp_j, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "origin_ras": [0.0, 0.0, 0.0],
    }


def extract_one(
    row: dict,
    base_dir: Path,
    out_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """1ケースを処理する。成功したら True を返す。"""
    case_id = row["patient_id"]
    dcm_path = base_dir / row["path"]

    if not dcm_path.exists():
        print(f"  [SKIP] DICOM not found: {dcm_path}")
        return False

    npy_path = out_dir / f"{case_id}_image.npy"
    json_path = out_dir / f"{case_id}_landmarks.json"

    if npy_path.exists() and json_path.exists() and not force:
        print(f"  [SKIP] Already exists: {case_id}")
        return True

    print(f"  {case_id}: {dcm_path}")

    if dry_run:
        return True

    arr, spacing = _load_dicom_pixel(dcm_path)
    H, W = arr.shape

    np.save(str(npy_path), arr)

    meta = {
        "case_id": case_id,
        "db_id": row["id"],
        "image_shape": [H, W],
        "metadata": _build_metadata(spacing),
        "landmarks_ijk": make_cervical_landmark_template(),
    }

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract cervical spine dataset from SQLite DB")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path")
    parser.add_argument("--base", default=DEFAULT_BASE, help="DICOM root directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--region", default=DEFAULT_REGION,
                        help=f"DB region value (default: {DEFAULT_REGION}). "
                             "Run --dry-run first to confirm case count.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N cases (テスト用)")
    parser.add_argument("--case", type=str, default=None, help="特定ケースIDのみ処理 (例: K004)")
    parser.add_argument("--force", action="store_true", help="既存ファイルを上書きして再生成")
    parser.add_argument("--dry-run", action="store_true", help="DBクエリのみ実行、ファイル書き込みなし")
    args = parser.parse_args()

    db_path = Path(args.db)
    base_dir = Path(args.base)
    out_dir = Path(args.out)

    if not db_path.exists():
        sys.exit(f"DB not found: {db_path}")
    if not base_dir.exists():
        sys.exit(f"DICOM root not found: {base_dir}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    params: list = []
    query = f"SELECT id, patient_id, path FROM images WHERE region=? AND view='LAT'"
    params.append(args.region)
    if args.case:
        query += " AND patient_id = ?"
        params.append(args.case)
    query += " ORDER BY patient_id"
    if args.limit:
        query += f" LIMIT {args.limit}"
    rows = con.execute(query, params).fetchall()
    con.close()

    print(f"対象 (region='{args.region}'): {len(rows)} 件")
    if len(rows) == 0:
        print("ヒント: --region オプションで正しい region 値を指定してください。")
        print("  例: --region WHOLE_SPINE, --region C_SPINE")
    if args.dry_run:
        print("[dry-run] ファイル書き込みはスキップします")

    ok = 0
    for row in rows:
        if extract_one(dict(row), base_dir, out_dir, dry_run=args.dry_run, force=args.force):
            ok += 1

    print(f"\n完了: {ok}/{len(rows)} 件")


if __name__ == "__main__":
    main()
