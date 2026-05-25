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

### ランドマーク命名規則（Phase 2 設計）

```
{椎体}_{終板}_{前後}
例: L3_sup_ant（L3椎体上終板前縁）、T4_inf_post（T4椎体下終板後縁）
```

- **椎体**: `C2`-`C7`, `T1`-`T12`, `L1`-`L5`, `S1`
- **終板**: `sup`（上終板）/ `inf`（下終板）
- **前後**: `ant`（前縁）/ `post`（後縁）
- **大腿骨頭**: `FH_L`（左）/ `FH_R`（右）
- **合計**: 24椎体 × 4点 + 2点 = **98点**

### Phase 1 との対応（後方互換）

| Phase 1（現行） | Phase 2（予定） |
|---|---|
| `L1_ant` | `L1_sup_ant` |
| `L1_post` | `L1_sup_post` |
| `S1_ant` | `S1_sup_ant` |
| `S1_post` | `S1_sup_post` |
| `FH` | `(FH_L + FH_R) / 2` で派生 |
| `L1_center` | 4点平均で派生 |

### 計測角度

**Phase 1（現行、5角度）**

| 名前 | 定義 |
|---|---|
| PI | Pelvic Incidence |
| PT | Pelvic Tilt |
| SS | Sacral Slope |
| LL | Lumbosacral Lordosis（L1-S1 Cobb角） |
| L1PA | FH→S1中点とFH→L1_centerの符号付き角度 |

PI = SS + PT の恒等式が成立する（`test_compute_angles_pi_ss_pt_relationship` で保護）。

**Phase 2 追加予定角度**

| 名前 | 定義 | 使用点 |
|---|---|---|
| TK | 胸椎後弯（T4-T12 Cobb） | T4_sup, T12_inf |
| CL | 頸椎前弯（C2-C7 Cobb） | C2_sup, C7_inf |
| T1S | T1傾斜（水平との角度） | T1_sup |
| SVA | 矢状鉛直軸（C7-S1水平距離） | C7_sup, S1_sup_post |
| TPA | T1骨盤角 | T1_sup, S1_sup, FH |
| PI-LL | 骨盤不一致 | 派生 |
| 各分節角度 | 任意2椎体間のCobb角 | 自由設定 |

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
