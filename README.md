# Sagittal Measure Assist

脊椎側面X線（DICOM）から椎体ランドマークを検出し、矢状面アライメント角度（PI/PT/SS/LL/L1PA）を計測する 3D Slicer 拡張 + AIモデル訓練パイプライン。

---

## セットアップ

### 3D Slicer 拡張

1. Slicer → `Edit > Application Settings > Modules > Additional module paths` に `SagittalMeasureAssist/` を追加
2. Slicer 再起動

### ローカル開発環境（ML 依存あり）

```bash
uv sync --extra ml
uv run -m pytest        # テスト実行
```

---

## 主なコマンド

```bash
# 訓練（ローカル CPU）
uv run python train/train.py \
  --data-dir train/dataset/l1pa \
  --backbone smallunet --sigma 5 --augment --loss awl \
  --split-seed 42 --epochs 50

# ONNX 変換
uv run python train/export_onnx.py \
  --checkpoint train/runs/best.pt --output train/runs/best.onnx

# 定量評価（ランドマーク MRE + 角度 MAE + Bland-Altman）
uv run python train/infer_onnx.py \
  --model train/runs/best.onnx --dir train/dataset/l1pa

# テストセット限定評価（再現可能）
uv run python train/infer_onnx.py \
  --model train/runs/best.onnx --dir train/dataset/l1pa \
  --splits train/runs/splits.json --subset test

# Human baseline（アノテーション間誤差）
uv run python scripts/inter_annotator_error.py
```

---

## ディレクトリ構成

```
SagittalMeasureAssist/        3D Slicer 拡張本体
  SagittalMeasureAssist.py    エントリーポイント
  lib/
    logic_angles.py           角度計算（PI/PT/SS/LL/L1PA）
    logic_inference.py        ONNX 推論・信頼度スコア・解剖チェック
    logic_export.py           学習データエクスポート
    assist_controller.py      UI ↔ ロジック橋渡し
    ui_*.py                   各パネルの UI

train/                        訓練・評価スクリプト（Slicer 外部）
  train.py                    訓練メイン（train/val/test 3 分割、固定 seed）
  model.py                    SmallUNet アーキテクチャ
  dataset.py                  HeatmapDataset（拡張あり）
  export_onnx.py              PyTorch → ONNX 変換
  infer_onnx.py               推論・MRE/角度 MAE/Bland-Altman/ICC 評価
  colab/
    train.ipynb               本番訓練（Google Colab GPU 用）
    finetune.ipynb            継続学習（Google Colab GPU 用）
  learning_curve/             学習曲線実験（探索的、Colab GPU 用）
    00_prepare_folds.ipynb    folds.json 生成（1 回のみ）
    resnet34/01_train.ipynb   ResNet-34 系バリアント
    smallunet/01_train.ipynb  SmallUNet 系バリアント
    02_plot_curve.ipynb       学習曲線プロット
    README.md

scripts/
  inspect_dicom.py            DICOM メタデータ調査
  inter_annotator_error.py    アノテーション間誤差計算（human baseline）

tests/                        テストスイート（193 テスト）

train/data/                   DICOM ファイル（gitignore、~2.4 GB）
train/dataset/                エクスポート済み npy/json（gitignore、~5.7 GB）
train/runs/                   訓練済みチェックポイント・ONNX・splits.json
```

---

## 計測パラメータ

### ランドマーク命名規則（Phase 2 設計）

```
{椎体}_{終板}_{前後}  例: L3_sup_ant, T4_inf_post
```

C2–S1 全椎体 × 4 点（上下終板 × 前後縁）+ FH_L / FH_R = **98 点**

> **Phase 1（現行）** は 6 点のみ: `L1_ant`, `L1_post`, `S1_ant`, `S1_post`, `FH`, `L1_center`
> 詳細は `RESEARCH.md` の Phase 設計を参照。

### Phase 1 計測角度（5 角度）

| 名前 | 定義 |
|---|---|
| PI | Pelvic Incidence |
| PT | Pelvic Tilt |
| SS | Sacral Slope |
| LL | Lumbosacral Lordosis（L1–S1 Cobb 角） |
| L1PA | FH→S1_mid と FH→L1_center の符号付き角度 |

### Phase 2 追加予定角度

TK（胸椎後弯）、CL（頸椎前弯）、T1S、SVA、TPA、PI-LL、各分節 Cobb 角

---

## Colab を使った GPU 訓練

1. `train/colab/train.ipynb` を Google Colab（T4 GPU）で開く
2. Google Drive の任意フォルダに `*_image.npy` / `*_landmarks.json` をアップロード
3. `DRIVE_DATA_DIR` を設定して全セル実行
4. 出力 ONNX をダウンロードして `train/runs/` に配置

学習曲線実験（5-fold CV）は `train/learning_curve/` の手順を参照。
