# CLAUDE.md

## プロジェクト概要

脊椎側面X線（DICOM）から椎体ランドマークを検出し、矢状面アライメント角度を計測する支援ツール。
- `SagittalMeasureAssist/` — UI付き計測アプリ（PySimpleGUI）
- `train/` — ランドマーク検出モデルの訓練・ONNX出力
- `data/` — DICOMファイル置き場（Patient's ID でリネーム済み）

## コマンド

```bash
# テスト
uv run -m pytest

# アプリ起動
uv run python SagittalMeasureAssist/SagittalMeasureAssist.py

# 訓練（ML依存が必要）
uv sync --extra ml
uv run python train/train.py

# ONNX出力
uv run python train/export_onnx.py
```

## データ準備

データ構造が一定でないため、DICOMファイルは手動で `data/` に配置する。

### 手動リネーム

`magick` でDICOMメタデータのPatient's IDを読んでリネームする：

```bash
for f in data/*; do
  id=$(magick identify -verbose "$f" 2>/dev/null | grep "dcm:Patient'sID:" | awk -F': ' '{print $2}' | tr -d ' ')
  mv "$f" "$(dirname $f)/$id"
done
```

主なDICOMフィールド参照コマンド：

```bash
magick identify -verbose <file> | grep -i "patient\|study\|dcm:"
```

### DICOMファイル調査

```bash
uv run --with pydicom python scripts/inspect_dicom.py <file>
uv run --with pydicom python scripts/inspect_dicom.py --scan data/
```
