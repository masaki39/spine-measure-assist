# Research Notes

## 研究目的

> 少ないデータで実用的な精度に達する脊椎矢状面アライメント自動計測モデルを構築し、学術・臨床適用可能なレベルまで検証する。

---

## フェーズ構成

### Phase 1（現在）: 腰仙椎・骨盤パラメータモデル

- **ランドマーク**: 6点（L1_ant/post, S1_ant/post, FH, L1_center）
- **角度**: PI, PT, SS, LL, L1PA
- **目的**: 設計・学習パイプラインの確立とベースライン精度の把握

### Phase 2（次）: 全脊椎ランドマークモデル

- **ランドマーク**: 98点（C2-S1全椎体 × 4点 + FH左右）
- **角度**: 頸椎前弯・胸椎後弯・腰椎前弯・SVA・TPA等（多数）
- **目的**: 包括的な脊椎アライメント自動計測

---

## Phase 1 現状

### データセット

- N = 186症例（単一施設、6点アノテーション）
- `train/dataset/l1pa/` が現在の標準データ
- `train/dataset/original/` は旧5点データ（L1_center なし）

### Human Baseline（intra-annotator, N=186）

| 対象 | MRE(mm) | SD | PI MAE(°) | PT MAE(°) | SS MAE(°) | LL MAE(°) |
|---|---|---|---|---|---|---|
| Intra-annotator | 2.52 | 3.73 | 3.09 | 1.03 | 2.80 | 3.57 |

> SD・maxが大きい → 難症例で外れ値あり。AIも同様の傾向が予測される。

### AI モデル学習曲線（5-fold CV、MRE mean±std）

| variant | 15症例 | 60症例 | 149症例 | 角度MAE（149症例） |
|---|---|---|---|---|
| `smallunet_aug0_mse_s15` | 82.3±98.6mm | 15.0±2.1mm | 10.7±0.8mm | **未計測（再実行要）** |
| `resnet34_aug1_awl_s15` | 37.9±25.6mm | 12.4±0.7mm | 10.5±0.3mm | **未計測（再実行要）** |

> 149症例で MRE≈10mm。Human baseline（2.52mm）に対して大きな差が残る。
> 角度 MAE は Colab 再実行後に確認（ノートブックは更新済み）。

### Sigma 最適化結果（全データ評価、SmallUNet）

| sigma | MRE(mm) | SDR@2mm | SDR@4mm |
|---|---|---|---|
| 4 | 2.46 | 59.5% | **91.3%** |
| 5 | 4.90 | 26.7% | 68.3% |
| 15 | 5.82 | 11.3% | 39.2% |

> sigma=4 が最適だが全データ学習の過大評価。学習曲線実験では sigma=15 で安定収束を確認。
> 本番では sigma=5 を採用（収束安定性と精度のバランス）。

---

## Phase 2 設計

### 新ランドマーク命名規則

```
{椎体}_{終板}_{前後}

終板: sup（上終板）/ inf（下終板）
前後: ant（前縁）/ post（後縁）
```

| 部位 | 椎体 | 点数 | 命名例 |
|---|---|---|---|
| 頸椎 | C2-C7 | 6 × 4 = 24 | `C3_sup_ant`, `C3_sup_post`, `C3_inf_ant`, `C3_inf_post` |
| 胸椎 | T1-T12 | 12 × 4 = 48 | `T4_sup_ant`, `T4_sup_post`, ... |
| 腰椎 | L1-L5 | 5 × 4 = 20 | `L1_sup_ant`, `L1_sup_post`, ... |
| 仙骨 | S1 | 4 | `S1_sup_ant`, `S1_sup_post`, `S1_inf_ant`, `S1_inf_post` |
| 大腿骨頭 | - | 2 | `FH_L`（左）, `FH_R`（右） |
| **合計** | | **98点** | |

### Phase 1 → Phase 2 の後方互換性

| Phase 1 名 | Phase 2 名 | 変換 |
|---|---|---|
| `L1_ant` | `L1_sup_ant` | リネーム |
| `L1_post` | `L1_sup_post` | リネーム |
| `S1_ant` | `S1_sup_ant` | リネーム |
| `S1_post` | `S1_sup_post` | リネーム |
| `FH` | `(FH_L + FH_R) / 2` | 派生計算 |
| `L1_center` | `(L1_sup_ant + L1_sup_post + L1_inf_ant + L1_inf_post) / 4` | 派生計算 |

### Phase 2 で計算可能な主要角度

| 角度 | 定義 | 使用ランドマーク |
|---|---|---|
| **TK**（胸椎後弯） | T4-T12 Cobb角 | T4_sup, T12_inf |
| **LL**（腰椎前弯） | L1-S1 Cobb角 | L1_sup, S1_sup（現在と同等） |
| **CL**（頸椎前弯） | C2-C7 Cobb角 | C2_sup, C7_inf |
| **PI/PT/SS** | 骨盤パラメータ | FH_L, FH_R, S1_sup |
| **T1S**（T1傾斜） | T1上終板の水平との角度 | T1_sup |
| **TPA**（T1骨盤角） | T1とS1中点-FH軸の角度 | T1_sup, S1_sup, FH |
| **SVA**（矢状鉛直軸） | C7重心からS1後壁の水平距離 | C7_sup, S1_sup_post |
| **PI-LL**（不一致） | PI - LL | 派生 |
| **各分節角度** | 任意2椎体間のCobb角 | 自由設定 |

---

## TODO

### 短期（次回 Colab 実行）

- [ ] **学習曲線に角度 MAE を追加して再実行** — `01_train.ipynb` の評価セルに角度計算を追加済み（要確認）
- [ ] **テストセット分割で本番モデルを再評価** — `--split-seed 42 --test-ratio 0.1` を使い、`infer_onnx.py --subset test` でテストセット限定評価を取得

### 中期（Phase 2 への移行）

- [ ] **3D Slicer アノテーション UI の拡張** — 98点に対応した点配置インターフェース。椎体ごとにグループ化、一括配置機能
- [ ] **新ランドマーク命名への移行** — `logic_angles.py` + `dataset.py` + JSON フォーマットを Phase 2 命名規則に更新
- [ ] **角度計算モジュールの拡張** — `logic_angles.py` に TK, CL, T1S, SVA, TPA, PI-LL 等を追加
- [ ] **既存 186 症例の再アノテーション** — 現在の 6 点 → 98 点へ拡張（段階的に可：まず腰椎 4 点/椎体→胸椎→頸椎）
- [ ] **モデルのスケールアップ** — 出力チャネル数 6 → 98 に対応した訓練

### 長期

- [ ] **外部検証データセット** — 別施設・別機器での汎化性能評価（論文必須）
- [ ] **公開モデルとの比較** — 先行研究との定量比較
- [ ] **学術論文執筆**

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
