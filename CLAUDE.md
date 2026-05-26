# CLAUDE.md

## プロジェクト概要

脊椎側面X線（DICOM）から椎体ランドマークを検出し、矢状面アライメント角度を計測する支援ツール。
- `SagittalMeasureAssist/` — UI付き計測アプリ（3D Slicer拡張）
- `train/` — ランドマーク検出モデルの訓練・ONNX出力

---

## ディレクトリ構成と役割

```
train/
  # Phase 1（現行・変更しない）
  train.py          訓練（CLI）
  model.py          SmallUNet
  dataset.py        HeatmapDataset（6点）
  export_onnx.py    PyTorch → ONNX
  infer_onnx.py     推論・評価（MRE/角度MAE/Bland-Altman/ICC）

  # Phase 2（新規）
  landmark_scheme.py   96点定義・BBox導出 の唯一の源泉
  dataset_phase2.py    Phase2 Dataset（mode=full/stage1/stage2）
  model_hrnet.py       HRNet-W32 + heatmap head（Stage2）
  export_onnx_phase2.py  Stage2 PT → ONNX
  infer_phase2.py      2段階推論・評価

  colab/
    train.ipynb           Phase1 訓練（GPU）
    finetune.ipynb        Phase1 fine-tune（GPU）
    train_detector.ipynb  Stage1 YOLOv8 訓練（GPU）
    train_phase2.ipynb    Stage2 HRNet 訓練（GPU）
  learning_curve/   学習曲線実験（探索的、Colabで実行）
  data/             DICOMファイル（gitignore）
  dataset/original/ 5点アノテーション（Phase1）
  dataset/l1pa/     6点アノテーション（Phase1 現行標準）
  dataset/phase2/   96点アノテーション（Phase2、gitignore）
  runs/             訓練済みモデル・splits.json

SagittalMeasureAssist/lib/
  logic_angles.py     角度計算の源泉（PI/PT/SS/LL/L1PA）
  logic_inference.py  ONNX推論・信頼度・解剖チェック
  logic_export.py     エクスポート処理

scripts/
  inspect_dicom.py          DICOM調査
  inter_annotator_error.py  アノテーション間誤差（human baseline）
  extract_dataset.py        DB → npy + JSON テンプレート生成（Phase2）

tests/                193+テスト
```

---

## コマンド

```bash
# テスト
uv run -m pytest

# --- Phase 1（現行）---

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

# --- Phase 2（新規）---

# DB から画像抽出 + アノテーションテンプレート生成
uv run python scripts/extract_dataset.py
uv run python scripts/extract_dataset.py --limit 3  # テスト用

# Phase2 ONNX変換（訓練後）
uv run python train/export_onnx_phase2.py \
  --checkpoint train/runs/phase2_best.pt \
  --output     train/runs/phase2_best.onnx

# 2段階推論評価（GT BBoxを使用）
uv run python train/infer_phase2.py \
  --stage1 train/runs/detector.onnx \
  --stage2 train/runs/phase2_best.onnx \
  --dir    train/dataset/phase2 \
  --use-gt-boxes
```

---

## データ準備

### Phase 2 データ抽出（DB ベース）

```bash
# 全件抽出（初回）
uv run python scripts/extract_dataset.py

# テスト抽出（3件）
uv run python scripts/extract_dataset.py --limit 3

# dry-run（DBクエリのみ、ファイル書き込みなし）
uv run python scripts/extract_dataset.py --dry-run
```

出力先: `train/dataset/phase2/`
- `{case_id}_image.npy` — 2D float32 画像
- `{case_id}_landmarks.json` — アノテーションテンプレート（landmarks_ijk は全 null）

### DICOM調査

```bash
uv run python scripts/inspect_dicom.py <file>
uv run python scripts/inspect_dicom.py --scan train/data/
```

---

## 計測パラメータ

### ランドマーク命名規則（Phase 2 実装済み）

`train/landmark_scheme.py` が**唯一の定義源**。命名は番号付き角点方式。

```
番号の意味:
  1 = 上前縁 (sup_ant)
  2 = 上後縁 (sup_post)
  3 = 下後縁 (inf_post)
  4 = 下前縁 (inf_ant)
```

| グループ | キー例 | 点数 |
|---|---|---|
| 頭蓋骨 | `EAC` | 1 |
| C2（下縁のみ） | `C2_3`, `C2_4` | 2 |
| C3–L5（4角点） | `C3_1`〜`C3_4`, ... | 88 |
| L6（optional） | `L6_1`〜`L6_4` | 0〜4 |
| S1（上縁のみ） | `S1_1`, `S1_2` | 2 |
| 大腿骨頭 | `FH` | 1 |
| 大腿骨 | `femur_prox`, `femur_dist` | 2 |
| **合計** | | **96（+L6 最大100）** |

### 解剖バリアント

JSON の `lumbar_variant` フィールドで管理:
- `"normal"`: 標準（L5 が最下位腰椎）
- `"lumbarization"`: S1 が腰椎化し L6 が存在
- `"sacralization"`: L5 が仙骨化し L5 キーが null

### BBox領域（Stage 1 検出対象、27クラス）

`skull`, `C2`, `C3`-`C7`, `T1`-`T12`, `L1`-`L5`, `L6`（任意）, `S1`, `pelvis`

BBoxはランドマーク座標から自動導出（`derive_bboxes()` in `landmark_scheme.py`）。

### データベース

`/Volumes/T7 Shield/kch-organized/dicom_index.db`
- `images` テーブル: `region='WHOLE_SPINE' AND view='LAT'` で 186件抽出

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

| 名前 | 定義 | 使用キー |
|---|---|---|
| TK | 胸椎後弯（T4-T12 Cobb） | `T4_1/T4_2`, `T12_3/T12_4` |
| CL | 頸椎前弯（C2-C7 Cobb） | `C2_3/C2_4`, `C7_3/C7_4` |
| T1S | T1傾斜 | `T1_1`, `T1_2` |
| SVA | 矢状鉛直軸 | `C7_1`, `S1_1` |
| TPA | T1骨盤角 | `T1_1`, `S1_1`, `FH` |
| PI-LL | 骨盤不一致 | 派生 |

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

**Phase 1（`dataset/l1pa/`）**: JSONキー変更時は `scripts/inter_annotator_error.py` と `train/dataset.py` を両方更新。

**Phase 2（`dataset/phase2/`）**: ランドマーク定義は `train/landmark_scheme.py` のみを変更する。他のモジュール（`dataset_phase2.py`, `infer_phase2.py`）はそこからインポートしている。

### Phase 2 固有ルール

- `landmark_scheme.py` は **すべての定義の唯一の源泉**。キーセット・BBoxクラス・チャネルマッピングをここ以外で定義しない
- `null` ランドマーク（未アノテーション）は heatmap=0、loss 計算から除外される
- BBox は常に `derive_bboxes()` で自動導出し、JSON には保存しない
- L6 等のオプションランドマークは JSON テンプレートに含まれるが null のまま保持できる

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
