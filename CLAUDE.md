# CLAUDE.md

## プロジェクト概要

脊椎側面X線（DICOM）から椎体ランドマークを検出し、矢状面アライメント角度を計測する支援ツール。
- `SagittalMeasureAssist/` — UI付き計測アプリ（3D Slicer拡張）
- `train/` — ランドマーク検出モデルの訓練・ONNX出力

---

## ディレクトリ構成と役割

```
train/
  train.py          訓練（CLI）
  model.py          SmallUNet
  dataset.py        HeatmapDataset
  export_onnx.py    PyTorch → ONNX
  infer_onnx.py     推論・評価（MRE/角度MAE/Bland-Altman/ICC）
  colab/            Google Colab GPU ノートブック（train / finetune）
  learning_curve/   学習曲線実験（探索的、Colabで実行）
  data/             DICOMファイル（gitignore）
  dataset/original/ 5点アノテーション（L1PA なし）
  dataset/l1pa/     6点アノテーション（現在の標準、L1_center 追加）
  runs/             訓練済みモデル・splits.json

SagittalMeasureAssist/lib/
  logic_angles.py     角度計算の源泉（PI/PT/SS/LL/L1PA）
  logic_inference.py  ONNX推論・信頼度・解剖チェック
  logic_export.py     エクスポート処理

scripts/
  inspect_dicom.py          DICOM調査
  inter_annotator_error.py  アノテーション間誤差（human baseline）

tests/                193テスト
```

---

## コマンド

```bash
# テスト
uv run -m pytest

# 訓練（標準: SmallUNet σ=5, AWL, augment, seed=42）
uv sync --extra ml
uv run python train/train.py \
  --data-dir train/dataset/l1pa --backbone smallunet --sigma 5 \
  --augment --loss awl --split-seed 42 --epochs 50

# ONNX出力
uv run python train/export_onnx.py \
  --checkpoint train/runs/best.pt --output train/runs/best.onnx

# 定量評価（全指標）
uv run python train/infer_onnx.py \
  --model train/runs/best.onnx --dir train/dataset/l1pa

# テストセット限定評価
uv run python train/infer_onnx.py \
  --model train/runs/best.onnx --dir train/dataset/l1pa \
  --splits train/runs/splits.json --subset test

# Human baseline
uv run python scripts/inter_annotator_error.py
```

---

## データ準備

### 手動リネーム（DICOMファイル）

```bash
for f in train/data/*; do
  id=$(magick identify -verbose "$f" 2>/dev/null | grep "dcm:Patient'sID:" | awk -F': ' '{print $2}' | tr -d ' ')
  n=$(echo "$id" | sed 's/K//')
  mv "$f" "$(dirname $f)/K$(printf '%03d' $n)"
done
```

### DICOM調査

```bash
uv run --with pydicom python scripts/inspect_dicom.py <file>
uv run --with pydicom python scripts/inspect_dicom.py --scan train/data/
```

---

## 計測パラメータ

### ランドマーク点（6点、`dataset/l1pa/` 標準）

| 名前 | 説明 |
|---|---|
| L1_ant | L1椎体上終板 前縁 |
| L1_post | L1椎体上終板 後縁 |
| S1_ant | S1椎体上終板 前縁 |
| S1_post | S1椎体上終板 後縁 |
| FH | 大腿骨頭中心（両側平均） |
| L1_center | L1椎体中心 |

### 計測角度（5角度）

| 名前 | 定義 |
|---|---|
| PI | Pelvic Incidence |
| PT | Pelvic Tilt |
| SS | Sacral Slope |
| LL | Lumbosacral Lordosis（L1-S1 Cobb角） |
| L1PA | FH→S1中点ベクトルとFH→L1_centerベクトルの符号付き角度 |

PI = SS + PT の恒等式が成立する（`test_compute_angles_pi_ss_pt_relationship` で保護）。

---

## 実装ルール

### 角度計算の一貫性

`logic_angles.py` が角度計算の**唯一の源泉**。
`infer_onnx.py` の `compute_angles()` は同一ロジックを再実装しており、`test_compute_angles_matches_logic_angles` で一致を保証している。
どちらかを変更したら必ず両方を同期させること。

### train/val/test 分割

- `--split-seed 42` がデフォルト（再現可能性のため固定）
- `splits.json` が訓練ごとに `runs/` 以下に保存される
- テストセット評価は `infer_onnx.py --splits splits.json --subset test` で実行

### データフォーマット

JSONの `landmarks_ijk` フィールドのキー名・座標系を変更する場合は
`scripts/inter_annotator_error.py` と `train/dataset.py` の両方を更新すること。

### Colabノートブック

- `train/colab/*.ipynb` は Google Colab（GPU）専用。ローカルでのテスト不要。
- `train/learning_curve/*.ipynb` は探索的実験用。本番パイプラインとは独立。

### Python実行

- `pip3 install` 禁止。`uvx --with <pkg> python3 script.py` を使うこと。
- ML依存: `uv sync --extra ml` でインストール。

### テスト方針

テストの目的は「実装時の意図がその後に意図せず変更されるのを防ぐ」こと。
- 角度計算の数学的正確性（`test_logic_angles.py`）
- 統計関数の正確性（`test_infer_statistics.py`）
- モデルの入出力shape（`test_model_shape.py`）
- 信頼度・解剖チェックのAPI（`test_logic_inference_utils.py`）
