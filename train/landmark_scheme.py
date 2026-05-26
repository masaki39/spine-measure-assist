"""
Phase 2 ランドマーク定義の唯一の源泉。
他モジュールはここからインポートすること。

命名規則（番号付き角点）:
  1 = sup_ant（上前縁）
  2 = sup_post（上後縁）
  3 = inf_post（下後縁）
  4 = inf_ant（下前縁）
"""

from __future__ import annotations

# ---- 椎体グループ ----

CERVICAL = [f"C{i}" for i in range(3, 8)]    # C3-C7 (5椎体)
THORACIC = [f"T{i}" for i in range(1, 13)]   # T1-T12 (12椎体)
LUMBAR   = [f"L{i}" for i in range(1, 6)]    # L1-L5 (5椎体)

# 4点を持つ標準椎体 (22椎体)
FULL_4PT_VERTEBRAE: list[str] = CERVICAL + THORACIC + LUMBAR

# ---- ランドマーク定義 ----
# 基本 96点 + L6 任意 4点 = 最大 100点

BASE_LANDMARKS: list[str] = (
    ["EAC"]
    + ["C2_3", "C2_4"]
    + [f"{v}_{i}" for v in FULL_4PT_VERTEBRAE for i in range(1, 5)]
    + ["S1_1", "S1_2"]
    + ["FH", "femur_prox", "femur_dist"]
)

OPTIONAL_LANDMARKS: list[str] = [f"L6_{i}" for i in range(1, 5)]

ALL_LANDMARK_KEYS: list[str] = BASE_LANDMARKS + OPTIONAL_LANDMARKS  # 100キー

# ---- BBoxクラス ----
# 学習用ラベルはランドマークから自動導出するため、アノテーション不要

BBOX_CLASSES: list[str] = (
    ["skull"]          # EAC用頭部ROI
    + ["C2"]
    + CERVICAL         # C3-C7
    + THORACIC         # T1-T12
    + LUMBAR           # L1-L5
    + ["L6"]           # 任意
    + ["S1"]
    + ["pelvis"]       # FH + 大腿骨用骨盤ROI
)

BBOX_CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(BBOX_CLASSES)}

# ---- Stage 2 チャネルマッピング ----
# 各BBox領域が持つ最大4チャネル。None = 学習対象外

REGION_CHANNELS: dict[str, list[str | None]] = {
    "skull":  ["EAC",       None,         None,         None],
    "C2":     [None,        None,         "C2_3",       "C2_4"],
    "S1":     ["S1_1",      "S1_2",       None,         None],
    "pelvis": ["FH",        "femur_prox", "femur_dist", None],
}

for _v in FULL_4PT_VERTEBRAE + ["L6"]:
    REGION_CHANNELS[_v] = [f"{_v}_1", f"{_v}_2", f"{_v}_3", f"{_v}_4"]

# 各領域の有効チャネルインデックス（Noneを除く）
REGION_VALID_CHANNELS: dict[str, list[int]] = {
    region: [i for i, k in enumerate(channels) if k is not None]
    for region, channels in REGION_CHANNELS.items()
}

# ---- 解剖バリアント ----

LUMBAR_VARIANTS = ("normal", "lumbarization", "sacralization")


# ---- JSONテンプレート生成 ----

def make_landmark_template() -> dict[str, dict]:
    """全ランドマークをnullで初期化したテンプレートを返す。"""
    return {k: {"i": None, "j": None, "k": 0} for k in ALL_LANDMARK_KEYS}


# ---- BBox自動導出 ----

def derive_bboxes(
    landmarks_ij: dict[str, tuple[float, float] | None],
    image_hw: tuple[int, int],
    pad_ratio: float = 0.2,
    fixed_pad_px: int = 150,
) -> dict[str, tuple[int, int, int, int] | None]:
    """
    ランドマーク座標からBBox (x1, y1, x2, y2) を導出する。

    Args:
        landmarks_ij: {キー: (i座標=列, j座標=行)} または None
        image_hw: (H, W) 画像サイズ
        pad_ratio: 4点BBoxの外側に追加するパディング割合
        fixed_pad_px: C2/S1/skull など点数が少ない領域の固定パディング(px)

    Returns:
        {領域名: (x1, y1, x2, y2)} または None（全点が未アノテーション）
    """
    H, W = image_hw
    result: dict[str, tuple[int, int, int, int] | None] = {}

    def _box(pts: list[tuple[float, float]], pad_x: float, pad_y: float) -> tuple[int, int, int, int]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (
            max(0, int(min(xs) - pad_x)),
            max(0, int(min(ys) - pad_y)),
            min(W - 1, int(max(xs) + pad_x)),
            min(H - 1, int(max(ys) + pad_y)),
        )

    for region, channels in REGION_CHANNELS.items():
        pts: list[tuple[float, float]] = []
        for key in channels:
            if key is not None:
                coord = landmarks_ij.get(key)
                if coord is not None:
                    pts.append(coord)

        if not pts:
            result[region] = None
            continue

        if region in ("skull", "C2", "S1"):
            result[region] = _box(pts, fixed_pad_px, fixed_pad_px)
        else:
            # 4点BBox: span幅の pad_ratio 分を外側に追加
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pw = max((max(xs) - min(xs)) * pad_ratio, 10)
            ph = max((max(ys) - min(ys)) * pad_ratio, 10)
            result[region] = _box(pts, pw, ph)

    return result


# ---- ユーティリティ ----

def get_region_for_landmark(key: str) -> str | None:
    """ランドマークキーが属するBBox領域を返す。"""
    for region, channels in REGION_CHANNELS.items():
        if key in channels:
            return region
    return None


def is_optional(key: str) -> bool:
    return key in OPTIONAL_LANDMARKS
