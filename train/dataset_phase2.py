"""
Phase 2 用 Dataset クラス。

- 最大 100点（L6 は optional、null なら学習から除外）
- YOLO形式ラベル生成（Stage 1 訓練用）
- BBoxクロップ + heatmap（Stage 2 訓練用）
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from train.dataset import _build_augmentation, _percentile_clip_norm, _resize_with_padding
from train.landmark_scheme import (
    ALL_LANDMARK_KEYS,
    BBOX_CLASSES,
    BBOX_CLASS_TO_IDX,
    REGION_CHANNELS,
    derive_bboxes,
    make_landmark_template,
)

# Stage 2 crop サイズ（全領域共通）
CROP_SIZE = (256, 256)


def _make_heatmap_masked(
    coords_and_mask: List[Tuple[Optional[Tuple[float, float]], bool]],
    size: Tuple[int, int],
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    heatmap と valid_mask を返す。
    coords_and_mask: [(coord_or_None, is_valid), ...]
    - coord=None or is_valid=False のチャネルは heatmap=0, mask=0
    Returns:
      heatmap: (L, H, W)
      valid_mask: (L,) bool
    """
    h, w = size
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    sigma2 = 2 * sigma * sigma
    heatmaps = []
    masks = []
    for coord, valid in coords_and_mask:
        if valid and coord is not None:
            x, y = coord
            g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / sigma2)
            heatmaps.append(g)
            masks.append(True)
        else:
            heatmaps.append(torch.zeros(h, w))
            masks.append(False)
    return torch.stack(heatmaps, dim=0), torch.tensor(masks, dtype=torch.bool)


class Phase2Dataset(Dataset):
    """
    Phase2 ランドマーク（最大100点）と BBox を扱う Dataset。

    mode="full"  : 全体画像 + 全ランドマーク heatmap（単純ボトムアップ用）
    mode="stage1": YOLO形式 BBox ラベル生成（Stage1 訓練用）
    mode="stage2": BBoxクロップ + 4ch heatmap（Stage2 訓練用）
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "full",
        resize: Tuple[int, int] = (512, 512),
        sigma: float = 3.0,
        percentile_clip: Tuple[float, float] = (1.0, 99.0),
        augment: bool = False,
        region_filter: Optional[List[str]] = None,
    ):
        assert mode in ("full", "stage1", "stage2"), f"Unknown mode: {mode}"
        self.data_dir = data_dir
        self.mode = mode
        self.resize = resize
        self.sigma = sigma
        self.percentile_clip = percentile_clip
        self._transform = _build_augmentation() if augment else None
        self.region_filter = region_filter  # stage2: 訓練する領域名リスト

        self.samples = self._discover_samples()
        if mode == "stage2":
            # 各(画像, 領域)ペアを1サンプルとして展開
            self.samples_stage2 = self._expand_stage2()

    def _discover_samples(self) -> List[Tuple[str, str, str]]:
        out = []
        for fname in os.listdir(self.data_dir):
            if not fname.endswith("_image.npy"):
                continue
            base = fname.replace("_image.npy", "")
            npy_path = os.path.join(self.data_dir, fname)
            json_path = os.path.join(self.data_dir, f"{base}_landmarks.json")
            if os.path.exists(json_path):
                out.append((base, npy_path, json_path))
        out.sort()
        if not out:
            raise RuntimeError(f"No samples found in {self.data_dir}")
        return out

    def _expand_stage2(self) -> List[Tuple[str, str, str, str]]:
        """(case_id, npy_path, json_path, region_name) のリストを返す。"""
        regions = self.region_filter if self.region_filter else list(REGION_CHANNELS.keys())
        return [
            (case_id, npy, js, region)
            for case_id, npy, js in self.samples
            for region in regions
        ]

    def __len__(self) -> int:
        if self.mode == "stage2":
            return len(self.samples_stage2)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        if self.mode == "stage2":
            case_id, npy_path, json_path, region = self.samples_stage2[idx]
            return self._getitem_stage2(case_id, npy_path, json_path, region)
        case_id, npy_path, json_path = self.samples[idx]
        if self.mode == "stage1":
            return self._getitem_stage1(case_id, npy_path, json_path)
        return self._getitem_full(case_id, npy_path, json_path)

    # ------------------------------------------------------------------ #
    #  共通: 画像ロードと前処理
    # ------------------------------------------------------------------ #

    def _load_image(self, npy_path: str) -> np.ndarray:
        img = np.load(npy_path)
        if img.ndim == 3:
            img = img[0]
        if img.ndim != 2:
            raise ValueError(f"Unexpected shape {img.shape}: {npy_path}")
        return img.astype(np.float32)

    def _load_meta(self, json_path: str) -> Dict:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _landmarks_to_dict(self, lm: Dict) -> Dict[str, Optional[Tuple[float, float]]]:
        """landmarks_ijk を {key: (i, j) | None} に変換する。"""
        result: Dict[str, Optional[Tuple[float, float]]] = {}
        for key in ALL_LANDMARK_KEYS:
            entry = lm.get(key, None)
            if entry is None or entry.get("i") is None:
                result[key] = None
            else:
                result[key] = (float(entry["i"]), float(entry["j"]))
        return result

    # ------------------------------------------------------------------ #
    #  mode="full": 全体画像 + 全ランドマーク heatmap
    # ------------------------------------------------------------------ #

    def _getitem_full(self, case_id: str, npy_path: str, json_path: str) -> Dict:
        img_np = self._load_image(npy_path)
        meta = self._load_meta(json_path)
        lm_dict = self._landmarks_to_dict(meta.get("landmarks_ijk", {}))

        img_np = _percentile_clip_norm(img_np, *self.percentile_clip)
        img_t = torch.from_numpy(img_np).unsqueeze(0)
        img_t, scale, pad_x, pad_y = _resize_with_padding(img_t, self.resize)

        coords_and_mask = []
        for key in ALL_LANDMARK_KEYS:
            coord = lm_dict[key]
            if coord is not None:
                x_r = coord[0] * scale + pad_x
                y_r = coord[1] * scale + pad_y
                coords_and_mask.append(((x_r, y_r), True))
            else:
                coords_and_mask.append((None, False))

        if self._transform is not None:
            valid_coords = [(c[0], c[1]) if (c is not None and valid) else (0.0, 0.0)
                            for c, valid in coords_and_mask]
            result = self._transform(image=img_t.squeeze(0).numpy(), keypoints=valid_coords)
            img_t = torch.from_numpy(result["image"].copy()).unsqueeze(0)
            th, tw = self.resize
            kps = result["keypoints"]
            new_cam = []
            for i, (_, valid) in enumerate(coords_and_mask):
                if valid:
                    kx = max(0.0, min(float(kps[i][0]), tw - 1.0))
                    ky = max(0.0, min(float(kps[i][1]), th - 1.0))
                    new_cam.append(((kx, ky), True))
                else:
                    new_cam.append((None, False))
            coords_and_mask = new_cam

        hr, wr = self.resize
        heatmap, valid_mask = _make_heatmap_masked(coords_and_mask, (hr, wr), self.sigma)

        return {
            "image": img_t,
            "heatmap": heatmap,
            "valid_mask": valid_mask,
            "case_id": case_id,
        }

    # ------------------------------------------------------------------ #
    #  mode="stage1": YOLO形式 BBox ラベル
    # ------------------------------------------------------------------ #

    def _getitem_stage1(self, case_id: str, npy_path: str, json_path: str) -> Dict:
        img_np = self._load_image(npy_path)
        meta = self._load_meta(json_path)
        lm_dict = self._landmarks_to_dict(meta.get("landmarks_ijk", {}))

        img_np = _percentile_clip_norm(img_np, *self.percentile_clip)
        img_t = torch.from_numpy(img_np).unsqueeze(0)
        img_t, scale, pad_x, pad_y = _resize_with_padding(img_t, self.resize)

        H, W = img_np.shape
        bboxes = derive_bboxes(lm_dict, (H, W))

        # YOLO形式: [class_id, cx_norm, cy_norm, w_norm, h_norm]
        labels = []
        for region, box in bboxes.items():
            if box is None:
                continue
            if region not in BBOX_CLASS_TO_IDX:
                continue
            cls_id = BBOX_CLASS_TO_IDX[region]
            x1, y1, x2, y2 = box
            # 元画像座標 → resize後座標
            x1r = x1 * scale + pad_x
            y1r = y1 * scale + pad_y
            x2r = x2 * scale + pad_x
            y2r = y2 * scale + pad_y
            th, tw = self.resize
            cx = (x1r + x2r) / 2 / tw
            cy = (y1r + y2r) / 2 / th
            bw = (x2r - x1r) / tw
            bh = (y2r - y1r) / th
            labels.append([cls_id, cx, cy, bw, bh])

        labels_t = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))

        return {
            "image": img_t,
            "labels": labels_t,  # (N, 5): [cls, cx, cy, w, h]
            "case_id": case_id,
        }

    # ------------------------------------------------------------------ #
    #  mode="stage2": BBoxクロップ + 4ch heatmap
    # ------------------------------------------------------------------ #

    def _getitem_stage2(self, case_id: str, npy_path: str, json_path: str, region: str) -> Dict:
        img_np = self._load_image(npy_path)
        meta = self._load_meta(json_path)
        lm_dict = self._landmarks_to_dict(meta.get("landmarks_ijk", {}))

        H_orig, W_orig = img_np.shape
        bboxes = derive_bboxes(lm_dict, (H_orig, W_orig))
        box = bboxes.get(region)

        if box is None:
            # 全ランドマークが未アノテーション → 黒画像 + all-invalid heatmap
            crop_t = torch.zeros(1, *CROP_SIZE)
            heatmap = torch.zeros(4, *CROP_SIZE)
            valid_mask = torch.zeros(4, dtype=torch.bool)
            return {
                "image": crop_t,
                "heatmap": heatmap,
                "valid_mask": valid_mask,
                "case_id": case_id,
                "region": region,
                "bbox_valid": False,
            }

        x1, y1, x2, y2 = box
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)

        # crop
        crop_np = img_np[y1:y2+1, x1:x2+1]
        crop_np = _percentile_clip_norm(crop_np, *self.percentile_clip)
        crop_t = torch.from_numpy(crop_np).unsqueeze(0)  # (1, bH, bW)
        crop_t, scale_c, pad_cx, pad_cy = _resize_with_padding(crop_t, CROP_SIZE)

        # crop内の座標変換
        channels = REGION_CHANNELS[region]
        coords_and_mask = []
        for key in channels:
            coord = lm_dict.get(key) if key is not None else None
            if coord is not None and key is not None:
                lx = (coord[0] - x1) * scale_c + pad_cx
                ly = (coord[1] - y1) * scale_c + pad_cy
                coords_and_mask.append(((lx, ly), True))
            else:
                coords_and_mask.append((None, False))

        if self._transform is not None:
            valid_coords = [(c[0], c[1]) if (c is not None and v) else (0.0, 0.0)
                            for c, v in coords_and_mask]
            result = self._transform(image=crop_t.squeeze(0).numpy(), keypoints=valid_coords)
            crop_t = torch.from_numpy(result["image"].copy()).unsqueeze(0)
            ch, cw = CROP_SIZE
            kps = result["keypoints"]
            new_cam = []
            for i, (_, valid) in enumerate(coords_and_mask):
                if valid:
                    kx = max(0.0, min(float(kps[i][0]), cw - 1.0))
                    ky = max(0.0, min(float(kps[i][1]), ch - 1.0))
                    new_cam.append(((kx, ky), True))
                else:
                    new_cam.append((None, False))
            coords_and_mask = new_cam

        heatmap, valid_mask = _make_heatmap_masked(coords_and_mask, CROP_SIZE, self.sigma)

        return {
            "image": crop_t,
            "heatmap": heatmap,      # (4, 256, 256)
            "valid_mask": valid_mask, # (4,) bool
            "case_id": case_id,
            "region": region,
            "bbox_valid": True,
        }


# ---- YOLO形式テキストファイル書き出しユーティリティ ----

def write_yolo_labels(
    dataset: Phase2Dataset,
    out_dir: str,
    min_annotated_ratio: float = 0.5,
) -> None:
    """
    YOLO訓練用ラベルテキストを out_dir に書き出す。
    BBoxが少なすぎるサンプルはスキップする。
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    assert dataset.mode == "stage1", "mode='stage1' が必要です"

    for i in range(len(dataset)):
        item = dataset[i]
        case_id = item["case_id"]
        labels: torch.Tensor = item["labels"]
        if len(labels) == 0:
            continue
        out_path = os.path.join(out_dir, f"{case_id}.txt")
        with open(out_path, "w") as fp:
            for row in labels.tolist():
                cls_id = int(row[0])
                cx, cy, bw, bh = row[1], row[2], row[3], row[4]
                fp.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
