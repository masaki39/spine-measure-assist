# CLAUDE.md

## プロジェクト概要

脊椎側面X線（DICOM）から椎体ランドマークを検出し、矢状面アライメント角度を計測する支援ツール。
- `SagittalMeasureAssist/` — UI付き計測アプリ（3D Slicer拡張）
- `train/` — ランドマーク検出モデルの訓練・ONNX出力
  - `train/data/` — DICOMファイル置き場（K001形式でリネーム済み）
  - `train/dataset/` — 訓練用 npy/json/nrrd ファイル（K001形式）
  - `train/runs/` — 訓練済みチェックポイント・ONNXモデル
  - `train/colabs/` — Google Colab用ノートブック（探索的）
  - `train/learning_curve/` — 学習曲線実験（探索的）

## コマンド

```bash
# テスト
uv run -m pytest

# アプリ起動
uv run python SagittalMeasureAssist/SagittalMeasureAssist.py

# 訓練（ML依存が必要）— 標準: SmallUNet σ=5
uv sync --extra ml
uv run python train/train.py --data-dir train/dataset --backbone smallunet --sigma 5

# ONNX出力
uv run python train/export_onnx.py
```

## データ準備

データ構造が一定でないため、DICOMファイルは手動で `train/data/` に配置する。
ファイル名は `K001`〜`K999` の3桁ゼロ埋め形式を使用する。

### 手動リネーム

`magick` でDICOMメタデータのPatient's IDを読んでリネームする：

```bash
for f in train/data/*; do
  id=$(magick identify -verbose "$f" 2>/dev/null | grep "dcm:Patient'sID:" | awk -F': ' '{print $2}' | tr -d ' ')
  n=$(echo "$id" | sed 's/K//')
  mv "$f" "$(dirname $f)/K$(printf '%03d' $n)"
done
```

主なDICOMフィールド参照コマンド：

```bash
magick identify -verbose <file> | grep -i "patient\|study\|dcm:"
```

### DICOMファイル調査

```bash
uv run --with pydicom python scripts/inspect_dicom.py <file>
uv run --with pydicom python scripts/inspect_dicom.py --scan train/data/
```

## 計測パラメータ

### ランドマーク点（6点）
| 名前 | 説明 |
|------|------|
| L1_ant | L1椎体上終板 前縁 |
| L1_post | L1椎体上終板 後縁 |
| S1_ant | S1椎体上終板 前縁 |
| S1_post | S1椎体上終板 後縁 |
| FH | 大腿骨頭中心（両側平均） |
| L1_center | L1椎体中心 |

### 計測角度（5角度）
| 名前 | 定義 |
|------|------|
| PI | Pelvic Incidence |
| PT | Pelvic Tilt |
| SS | Sacral Slope |
| LL | Lumbosacral Lordosis（L1-S1 Cobb角） |
| L1PA | FH→S1中点ベクトルとFH→L1_centerベクトルの符号付き角度（L1_centerが前方で正） |
