# 学習曲線分析

5-fold CV × 6段階の訓練サイズで学習曲線を描き、「何症例あれば実用水準に達するか」を検証する。

## ノートブック構成

| ノートブック | 役割 | 実行頻度 |
|---|---|---|
| `00_prepare_folds.ipynb` | folds.json 生成 | **1回のみ** |
| `resnet34/01_train.ipynb` | ResNet-34系バリアントを全fold一括実行 | バリアントごとに1回 |
| `smallunet/01_train.ipynb` | SmallUNet系バリアントを全fold一括実行 | バリアントごとに1回 |
| `02_plot_curve.ipynb` | 全バリアントの結果を集計・プロット | 都度 |

## 実行手順

1. Google Colab（GPU推奨）で `resnet34/01_train.ipynb` または `smallunet/01_train.ipynb` を開く
2. 設定セル（Cell 1）の `BACKBONE / AUGMENT / LOSS` を確認して全セル実行
3. 全 fold × 全サイズ（30ジョブ）が一括完走（2〜4時間）
4. 結果は Drive の `anglist_learning_curve/results/{VARIANT}/` に保存される（既存はスキップ）
5. `02_plot_curve.ipynb` で学習曲線をプロット

## バリアント定義

| variant | backbone | augment | loss | 目的 |
|---|---|---|---|---|
| `smallunet_aug0_mse_s15` | SmallUNet | なし | MSE | baseline |
| `resnet34_aug1_awl_s15` | ResNet-34 | あり | AWL | 本命（事前学習+拡張+AWL） |

## 結果ファイル構造（Drive）

```
anglist_learning_curve/
  folds.json
  results/
    resnet34_aug1_awl_s15/
      fold1_size015.json   # angle_mae, outlier除外指標を含む
      ...
      fold5_size149.json
    smallunet_aug0_mse_s15/
      ...
```

## 進捗

詳細は `RESEARCH.md` を参照。
