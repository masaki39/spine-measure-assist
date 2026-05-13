# Research Notes

## プロジェクトの着眼点

> 「少ないデータでも実用的な学習曲線を示せるか」

現代的手法（データ拡張 + 事前学習済みバックボーン + 改良損失関数）を組み合わせることで、
5-fold CV の学習曲線が実用水準に達するまでのデータ量を削減できるかを検証する。

**評価指標**: MRE(mm) / 角度MAE(°) を各訓練サイズ（15, 30, 60, 90, 120, 149症例）で比較。
人力計測誤差（inter-annotator）を臨床的ベースラインとして使用。

---

## 現状

- データ: 186症例、5ランドマーク（L1_ant, L1_post, S1_ant, S1_post, FH）
- 全データ学習（train評価）: MRE=2.46mm, SDR@4mm=91.3%（sigma=4, SmallUNet）→ 訓練セット評価のため過大評価
- **課題**: 5-fold CV にすると精度が大幅に低下 → 真の汎化性能はMRE≈10mm

### baseline の問題点

| 問題 | 詳細 |
|---|---|
| データ拡張なし | 実効的に186枚のみ |
| スクラッチ学習 | 事前学習なし、少データで汎化しにくい |
| MSE損失 | 背景ピクセルが支配的、ランドマーク付近の勾配が薄まる |

---

## 実験計画：2×2 要因比較

「何が効いているか」を分解するための最小構成。

| variant | backbone | augment | loss | sigma | 目的 |
|---|---|---|---|---|---|
| `smallunet_aug0_mse_s15` | SmallUNet | なし | MSE | 15 | **baseline** |
| `smallunet_aug1_awl_s15` | SmallUNet | あり | AWL | 15 | 拡張+AWLのみの効果確認 |
| `resnet34_aug0_mse_s15`  | ResNet-34 | なし | MSE | 15 | 事前学習のみの効果確認 |
| `resnet34_aug1_awl_s15`  | ResNet-34 | あり | AWL | 15 | **本命**（全手法） |

### sigma=15 を採用する理由

sigma=4 では 512×512 画像の約 0.006% にしか有意な勾配が生じず、少データでは収束しない（sigma=4 での実験は全サイズで MRE>70mm という結果で失敗）。
sigma=15 は学習の安定性を優先した選択。評価はヒートマップの argmax で行うため、sigma の大小は最終精度への影響は軽微。

### 追加候補（優先度低）

- `efficientnet-b3_aug1_awl_s15` : より効率的なバックボーンとの比較

---

## ノートブック構成

```
train/learning_curve/
  00_prepare_folds.ipynb          # folds.json 作成（1回だけ実行）
  resnet34/01_train.ipynb         # ResNet-34 系バリアント（全fold一括）
  smallunet/01_train.ipynb        # SmallUNet 系バリアント（全fold一括）
  02_plot_curve.ipynb             # 学習曲線プロット（結果比較）
```

各ノートブックは設定セルの `BACKBONE / AUGMENT / LOSS` を変えるだけで別バリアントを実行できる。
結果は Drive の `results/{VARIANT}/` に保存され、既存の結果は自動スキップされる。

### Colab 実行手順

1. GPU ランタイムで `resnet34/01_train.ipynb` または `smallunet/01_train.ipynb` を開く
2. 設定セルの `BACKBONE`, `AUGMENT`, `LOSS` を確認して実行
3. 全 fold × 全サイズ（30ジョブ）が一括で完走する
4. 必要なら設定を変えて別バリアントも実行

### ローカルでの単体テスト

```bash
uv run -m pytest
```

---

## 実験結果

### 人力計測誤差（inter-annotator baseline）

`scripts/inter_annotator_error.py` による original ↔ l1pa アノテーション比較（N=186）。

| 評価対象 | ランドマーク MRE | PI MAE | PT MAE | SS MAE | LL MAE |
|---|---|---|---|---|---|
| **Human (inter-annotator)** | **2.52mm** (SD=3.73, max=40.7mm) | **3.09°** | **1.03°** | **2.80°** | **3.57°** |

> SD・maxが大きい = 奇形や難症例で外れ値が存在。AIも同様に外れ値除外が必要。

### AI 学習曲線（5-fold CV、MRE mean±std over folds）

| variant | 15症例 | 60症例 | 149症例 | 角度MAE（149症例） |
|---|---|---|---|---|
| `smallunet_aug0_mse_s15` | 82.3±98.6mm | 15.0±2.1mm | 10.7±0.8mm | 未計測（再実行要） |
| `resnet34_aug1_awl_s15`  | 37.9±25.6mm | 12.4±0.7mm | 10.5±0.3mm | 未計測（再実行要） |
| `resnet34_aug1_awl` (sigma=4) | 失敗（収束せず） | - | - | - |

> MRE≈10mm はランドマーク単位の誤差。**角度への影響はColabの再実行後に確認**（修正済みノートブックが角度MAEを出力する）。

### resnet34_aug1_awl（sigma=4、失敗）

sigma=4 が小さすぎて全サイズで収束せず。
