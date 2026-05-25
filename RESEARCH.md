# Research Notes

## 研究目的

> 少ないデータで実用的な精度に達する脊椎矢状面アライメント自動計測モデルを構築し、学術・臨床適用可能なレベルまで検証する。

**評価指標**:
- ランドマーク検出: MRE(mm) ± SD、SDR@2mm / SDR@4mm、95%CI
- 角度精度: MAE(°)、Bland-Altman（bias, LoA）、ICC(3,1)
- 人力計測誤差（intra-annotator）を臨床 baseline として使用

---

## データセット

- N = 186 症例（単一施設）
- アノテーション: `train/dataset/l1pa/`（6点、L1_center追加版）
- human baseline: `scripts/inter_annotator_error.py` で計算（original ↔ l1pa 比較）

### Human Baseline（intra-annotator, N=186）

| 対象 | MRE(mm) | SD | PI MAE(°) | PT MAE(°) | SS MAE(°) | LL MAE(°) |
|---|---|---|---|---|---|---|
| Intra-annotator | 2.52 | 3.73 | 3.09 | 1.03 | 2.80 | 3.57 |

> SD・maxが大きい → 難症例（奇形等）で外れ値あり。AI も同様の傾向が予測される。

---

## 現状（AI モデル）

### 5-fold CV 学習曲線（MRE mean±std）

| variant | 15症例 | 60症例 | 149症例 | 角度MAE（149症例） |
|---|---|---|---|---|
| `smallunet_aug0_mse_s15` | 82.3±98.6mm | 15.0±2.1mm | 10.7±0.8mm | **未計測** |
| `resnet34_aug1_awl_s15` | 37.9±25.6mm | 12.4±0.7mm | 10.5±0.3mm | **未計測** |

> 149症例で MRE≈10mm。Human baseline（2.52mm）に対してまだ大きな差がある。
> 角度 MAE は Colab 再実行後に確認予定（ノートブック更新済み）。

---

## 実験計画

### 2×2 要因比較（学習曲線）

| variant | backbone | augment | loss | sigma | 目的 |
|---|---|---|---|---|---|
| `smallunet_aug0_mse_s15` | SmallUNet | なし | MSE | 15 | **baseline** |
| `resnet34_aug1_awl_s15` | ResNet-34 | あり | AWL | 15 | **本命**（全手法） |

sigma=15 を採用理由: sigma=4 では少データで収束しないことを実験で確認（全サイズで MRE>70mm）。

Colab実行: `train/learning_curve/` を参照。

---

## TODO

### 短期（次の Colab 実行前）

- [ ] **角度 MAE をノートブックで計測** — 現在の学習曲線には角度誤差がない。`infer_onnx.py` の評価コードを `01_train.ipynb` に組み込む
- [ ] **テストセット分割を本番訓練に適用** — `--split-seed 42 --test-ratio 0.1` で held-out test set を確保し、`splits.json` を保存

### 中期（論文・臨床適用に向けて）

- [ ] **Multi-rater（inter-rater）baseline の取得** — 現在の baseline は intra-annotator のみ。別観察者によるアノテーションで真の inter-rater 信頼性を測定する
- [ ] **外部検証データセット** — 別施設・別機器のデータで汎化性能を評価（論文必須要件）
- [ ] **ピクセル間隔の正規化** — 施設間でピクセル/mm が異なるため、前処理に spacing 正規化を追加
- [ ] **術後症例（金属インプラントあり）の対応** — 現データセットにはほぼ含まれていない可能性が高い

### 長期

- [ ] **公開モデルとの比較** — 同条件で先行研究との定量比較
- [ ] **学術論文執筆** — 評価指標: MRE, SDR, 角度 MAE, Bland-Altman, ICC

---

## ノートブック構成

```
train/colab/
  train.ipynb       本番訓練（Google Colab T4 GPU）
  finetune.ipynb    継続学習

train/learning_curve/
  00_prepare_folds.ipynb    folds.json 生成（1回のみ）
  resnet34/01_train.ipynb   ResNet-34 系（全fold × 全サイズ一括）
  smallunet/01_train.ipynb  SmallUNet 系
  02_plot_curve.ipynb       結果集計・学習曲線プロット
```

結果は Drive の `anglist_learning_curve/results/{VARIANT}/` に保存。

---

## 参考: sigma の選択

| sigma | 挙動 | 備考 |
|---|---|---|
| 4 | 512×512 画像の 0.006% にしか勾配が生じない | 少データで収束せず（全サイズ失敗） |
| 5 | 本番採用（SmallUNet） | 安定収束、推論精度は argmax で決まるため sigma は最終精度に軽微な影響 |
| 15 | 学習曲線実験で使用 | さらに安定、比較のため維持 |
