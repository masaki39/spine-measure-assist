# 🦴 Sagittal Measure Assist

脊椎側面X線（DICOM）から椎体ランドマークを手動で配置し、脊椎矢状面アライメント（PI/PT/SS/LL）を計測するための **3D Slicer 拡張**。
計測データを学習用にエクスポートし、AIモデルを訓練してランドマーク配置の自動化も目指せる。

<details>
<summary>💡 ONNXモデルとは？</summary>

ONNX（Open Neural Network Exchange）は、異なるフレームワーク間でAIモデルを共有するための標準フォーマット。
PyTorchで訓練したモデルをONNXに変換することで、`onnxruntime` を使ってCPUのみの環境（3D Slicer内など）でも高速に推論できる。
GPU不要・依存が少ない点が特徴。

</details>

---

## 🔄 全体の流れ

```
[1. 計測のみ]
DICOM読み込み → ランドマーク配置 → 角度を確認

[2. 学習データを作る]
DICOM読み込み → ランドマーク配置 → エクスポート（.npy/.nrrd/.json）

[3. モデルを訓練する]（Slicer外部）
エクスポートデータ → 学習 → ONNXモデル出力

[4. 自動推論を使う]
ONNXモデル + DICOM → ランドマーク自動配置 → 角度確認
```

---

## ⚙️ セットアップ

### Slicer拡張のインストール
1. 3D Slicerを起動
2. `Edit > Application Settings > Modules > Additional module paths` に `SagittalMeasureAssist/` フォルダを追加
3. Slicerを再起動 → モジュールリストに "Sagittal Measure Assist" が表示される

---

## 📋 使い方

### STEP 1: 📂 DICOMを読み込む

1. Slicerのメニューから `File > Add DICOM Data` を開く
2. DICOMファイルをインポートし、**Load** ボタンで読み込む
3. モジュール「Sagittal Measure Assist」を開き、**Volume** セレクターで対象を選ぶ
   - スライスビューが自動更新される
   - エクスポート用のケースIDにPatient IDが自動入力される

### STEP 2: 📍 ランドマークを配置する

「計測」セクションで **新規作成 / 1点追加** ボタンを使い、以下の順に5点を置く：

| 順番 | ランドマーク | 場所 |
|------|------------|------|
| 1 | L1_ant | L1頭側終板 前縁 |
| 2 | L1_post | L1頭側終板 後縁 |
| 3 | S1_ant | S1頭側終板 前縁 |
| 4 | S1_post | S1頭側終板 後縁 |
| 5 | FH | 両側大腿骨頭の中心 |

> **画像が左右逆の場合**: 「左右反転補正」にチェックを入れる。

### STEP 3: 📐 角度を確認する

**計測を更新** ボタンを押すと PI / PT / SS / LL（度）が表示される。

### STEP 4: 💾 学習データをエクスポートする（任意）

「エクスポート」セクションで：
1. **出力先フォルダ**を指定
2. **ケースID** を確認（Patient IDが自動入力済み）
3. **エクスポート** ボタンを押す

以下の3ファイルが保存される：

| ファイル | 内容 |
|---------|------|
| `{ケースID}_image.npy` | 画像配列（NumPy形式） |
| `{ケースID}_volume.nrrd` | ボリューム本体 |
| `{ケースID}_landmarks.json` | ランドマーク座標 + 角度 + メタデータ |

### STEP 5: 🧠 モデルを訓練する（任意・Slicer外部）

複数ケースをエクスポートしたら、以下の手順でモデルを訓練する。

```bash
# 依存インストール（初回のみ）
uv sync --extra ml

# 学習
uv run python train/train.py \
  --data-dir /path/to/exported \
  --save-dir runs \
  --epochs 20

# ONNXに変換
uv run python train/export_onnx.py \
  --checkpoint runs/best.pt \
  --output runs/best.onnx \
  --height 512 --width 512
```

既存モデルを出発点に追加データで継続学習（ファインチューニング）する場合：

```bash
uv run python train/train.py \
  --data-dir /path/to/new_data \
  --checkpoint runs/best.pt \
  --lr 1e-4 \
  --epochs 50
```

> ファインチューニングでは学習率を小さめ（`1e-4` 程度）にするのが推奨。新データのみで学習すると旧データのパターンを忘れやすい（破滅的忘却）ため、可能であれば旧データも混ぜて学習する。

> 💡 **GPU環境での学習**: `train/train_colab.ipynb` を使うとGoogle Colab上でGPUを使って訓練できる。追加データで継続学習する場合は `train/finetune_colab.ipynb` を使う。

学習の仕組み（概要）：
- 画像を512×512にリサイズ（縦横比を維持してパディング）
- 各ランドマークに2Dガウスを置いた5枚のヒートマップを教師信号に使用
- 軽量UNetでヒートマップを予測し、MSEで学習

### STEP 6: 🤖 自動推論を使う（任意）

訓練済みONNXモデルで5点を自動配置できる。`onnxruntime` は初回実行時に自動インストールされる。

1. 「自動推論 (ONNX)」セクションでモデルファイル（`.onnx`）を選択
2. 入力サイズを学習時と合わせる（デフォルト: 512×512）
3. **推論してMarkupsに配置** ボタンを押す → 5点が自動配置され、角度も更新される

推論後はモデルが出力したヒートマップをスライスビューに半透明でオーバーレイ表示できる：

### STEP 7: 📊 モデルを定量評価する（任意）

複数サンプルに対してMRE（Mean Radial Error）とSDR（Successful Detection Rate）を一括計算できる。

```bash
uv run python train/infer_onnx.py --model runs/model.onnx --dir dataset/
```

出力例：

```
=== MRE Evaluation (N=186 samples) ===
Landmark       MRE(px)   MRE(mm)   SDR@2mm   SDR@4mm
L1_ant           14.16      6.07     12.4%     40.9%
...
Overall          13.56      5.82     11.3%     39.2%
```

指標の詳細は `runs/evaluation_metrics.md` を参照。

| コントロール | 説明 |
|---|---|
| **Heatmapを表示** チェックボックス | オーバーレイのオン/オフ |
| **表示ランドマーク** コンボボックス | 全体（合成）または各ランドマーク（L1_ant / L1_post / S1_ant / S1_post / FH）を選択 |
| **Heatmap透明度** スライダー | 0〜100%で透明度を調整 |

---

## 🧪 テスト

```bash
uv run -m pytest
```

---

## 📁 ディレクトリ構成

```
SagittalMeasureAssist/   # Slicer拡張本体
  SagittalMeasureAssist.py  # エントリーポイント
  lib/
    assist_controller.py  # UIとロジックをつなぐ
    ui_measure.py         # 計測パネルUI
    ui_export.py          # エクスポートパネルUI
    ui_auto.py            # 自動推論パネルUI
    logic_angles.py       # 角度計算
    logic_export.py       # エクスポート処理
    logic_inference.py    # ONNX推論処理

train/                   # 学習・変換スクリプト（Slicer外部）
  train.py               # 訓練メインスクリプト
  model.py               # SmallUNetアーキテクチャ
  dataset.py             # HeatmapDataset
  export_onnx.py         # PyTorch → ONNX変換
  infer_onnx.py          # スタンドアロン推論・MRE評価スクリプト
  train_colab.ipynb      # Google Colab用ノートブック（初回学習）
  finetune_colab.ipynb   # Google Colab用ノートブック（追加学習）
  learning_curve/        # 学習曲線分析（5-fold CV × 6段階）
    README.md            # 進捗タスクリスト（30ジョブ）
    00_prepare_folds.ipynb   # フォールド割り当て生成（1回のみ）
    01_train_one_job.ipynb   # 1ジョブ学習（FOLD × SIZE を指定）
    02_plot_curve.ipynb      # 結果集計 & 学習曲線プロット

dataset/                 # エクスポート済み学習データ（.npy/.json/.nrrd）
runs/                    # 学習済みモデル（best.pt, model.onnx）+ 評価結果
data/                    # DICOMファイル置き場（Patient IDでリネーム済み）
tests/                   # テストスイート
scripts/                 # DICOMメタデータ調査スクリプト
```
