# Research Notes

## プロジェクトの着眼点

> 「少ないデータでも実用的な学習曲線を示せるか」

現代的手法（データ拡張 + 事前学習済みバックボーン + 改良損失関数）を組み合わせることで、
5-fold CV の学習曲線が実用水準に達するまでのデータ量を削減できるかを検証する。

**評価指標**: MRE(mm) / SDR@4mm を各訓練サイズ（15, 30, 60, 90, 120, 149症例）で比較

---

## 現状

- データ: 186症例、5ランドマーク（L1_ant, L1_post, S1_ant, S1_post, FH）
- 全データ学習: MRE=2.46mm, SDR@4mm=91.3%（sigma=4, SmallUNet）
- **課題**: 5-fold CV にすると精度が大幅に低下 → データ効率が低い

### baseline の問題点

| 問題 | 詳細 |
|---|---|
| データ拡張なし | 実効的に186枚のみ |
| スクラッチ学習 | 事前学習なし、少データで汎化しにくい |
| MSE損失 | 背景ピクセルが支配的、ランドマーク付近の勾配が薄まる |

---

## 実験バリアント

| variant | backbone | augment | loss | 状態 |
|---|---|---|---|---|
| `smallunet_aug0_mse` | SmallUNet | なし | MSE | baseline（未実行） |
| `resnet34_aug1_awl` | ResNet-34 | あり | AWL | 新手法（未実行） |

### 拡張手法の詳細（`resnet34_aug1_awl`）

- **Backbone**: ResNet-34 (ImageNet事前学習) + U-Net デコーダ
- **Augmentation**: Rotation ±15°, ElasticTransform(α=50, σ=5), BrightnessContrast ±20%
- **Loss**: Adaptive Wing Loss — ランドマーク付近（heatmap値≈1）を強く、背景（≈0）を弱く学習

---

## 次のアクション

- [ ] Colab で `smallunet_aug0_mse` 30ジョブ実行（baseline）
- [ ] Colab で `resnet34_aug1_awl` 30ジョブ実行（新手法）
- [ ] `02_plot_curve.ipynb` で両者の学習曲線を比較プロット
- [ ] 結果を見て追加バリアント（augのみ、backboneのみ等）を検討

---

## 実験結果

| variant | 15症例 MRE | 60症例 MRE | 149症例 MRE | 全体SDR@4mm |
|---|---|---|---|---|
| smallunet_aug0_mse | - | - | - | - |
| resnet34_aug1_awl | - | - | - | - |

---

## 技術メモ

### Colab 実行手順

1. `01_train_one_job.ipynb` を開く
2. 設定セルの `BACKBONE`, `AUGMENT`, `LOSS` を変更
3. Fold 1〜5 を順次実行（結果は Drive の `results/{VARIANT}/` に保存）

### ローカルでの単体テスト

```bash
uv run -m pytest
```

### 新しいバリアントを追加する場合

`01_train_one_job.ipynb` の設定セルを変更するだけ。結果は `VARIANT` 名で自動的に別ディレクトリに保存される。
