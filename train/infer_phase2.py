"""
Phase 2 二段階推論（Stage1 ONNX + Stage2 ONNX）と評価。

使用法:
  # 単一画像（JSON なしで推論のみ）
  uv run python train/infer_phase2.py \\
    --stage1 train/runs/detector.onnx \\
    --stage2 train/runs/phase2_best.onnx \\
    --image  train/dataset/phase2/K001_image.npy

  # ディレクトリ一括評価（GT JSON が必要）
  uv run python train/infer_phase2.py \\
    --stage1 train/runs/detector.onnx \\
    --stage2 train/runs/phase2_best.onnx \\
    --dir    train/dataset/phase2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.dataset import _percentile_clip_norm, _resize_with_padding
from train.landmark_scheme import (
    ALL_LANDMARK_KEYS,
    BBOX_CLASSES,
    BBOX_CLASS_TO_IDX,
    REGION_CHANNELS,
    derive_bboxes,
)

# Stage2 crop サイズ（dataset_phase2.py と一致させること）
_CROP_SIZE = (256, 256)

try:
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
except ImportError as e:
    sys.exit(f"依存が不足しています: {e}\nuv sync --extra ml を実行してください")


# ---- 前処理ユーティリティ ----

def _preprocess_image(img_np: np.ndarray, target_size: Tuple[int, int] = (512, 512)):
    """(H,W) float32 → (1,1,H,W) tensor + 変換パラメータ。"""
    import torch
    img_np = _percentile_clip_norm(img_np)
    img_t = torch.from_numpy(img_np).unsqueeze(0)  # (1,H,W)
    img_t, scale, pad_x, pad_y = _resize_with_padding(img_t, target_size)
    return img_t.unsqueeze(0).numpy(), scale, pad_x, pad_y  # (1,1,Ht,Wt)


def _preprocess_crop(crop_np: np.ndarray, target_size: Tuple[int, int] = _CROP_SIZE):
    """(H,W) float32 → (1,1,256,256) tensor + 変換パラメータ。"""
    import torch
    crop_np = _percentile_clip_norm(crop_np)
    crop_t = torch.from_numpy(crop_np).unsqueeze(0)
    crop_t, scale, pad_x, pad_y = _resize_with_padding(crop_t, target_size)
    return crop_t.unsqueeze(0).numpy(), scale, pad_x, pad_y


def _argmax2d(hmap: np.ndarray) -> Tuple[float, float, float]:
    """(H,W) heatmap から (x, y, confidence) を返す。"""
    idx = np.argmax(hmap)
    y, x = np.unravel_index(idx, hmap.shape)
    return float(x), float(y), float(hmap.flat[idx])


# ---- Stage 1: 椎体BBox検出 ----

class Stage1Detector:
    """YOLOv8 ONNX ラッパー。"""

    def __init__(self, onnx_path: str, conf_thresh: float = 0.3, iou_thresh: float = 0.5):
        self.sess = ort.InferenceSession(onnx_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(
        self, img_np: np.ndarray
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """
        (H,W) float32 画像から椎体BBoxを検出する。

        Returns:
            {region_name: (x1, y1, x2, y2)} 元画像座標
        """
        H, W = img_np.shape
        inp, scale, pad_x, pad_y = _preprocess_image(img_np, (640, 640))
        inp_3ch = np.repeat(inp, 3, axis=1)  # (1,3,640,640)

        outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: inp_3ch})[0]
        # YOLOv8 出力: (1, 4+num_classes, num_anchors) または (1, num_anchors, 4+num_classes)
        # ここでは後処理を簡易実装（ultralytics の出力形式に依存）
        boxes = self._parse_yolo_output(outputs, H, W, scale, pad_x, pad_y)
        return boxes

    def _parse_yolo_output(self, outputs, H, W, scale, pad_x, pad_y):
        """YOLOv8 ONNX 出力をパースして {region: (x1,y1,x2,y2)} を返す。"""
        # outputs shape: (1, 4+num_classes, num_anchors) [YOLOv8 export形式]
        pred = outputs[0]  # (4+num_classes, num_anchors)
        if pred.ndim == 3:
            pred = pred[0]  # remove batch dim → (4+nc, na)

        # xyxy形式: pred[:4] = [cx, cy, w, h] in resized space
        boxes_xywh = pred[:4].T    # (na, 4)
        scores     = pred[4:].T    # (na, nc)
        class_ids  = np.argmax(scores, axis=1)
        confs      = np.max(scores, axis=1)

        result = {}
        keep = confs >= self.conf_thresh
        for i in np.where(keep)[0]:
            cls_id = int(class_ids[i])
            if cls_id >= len(BBOX_CLASSES):
                continue
            region = BBOX_CLASSES[cls_id]
            cx, cy, bw, bh = boxes_xywh[i]
            # resized → 元画像座標
            x1 = max(0, int((cx - bw / 2 - pad_x) / scale))
            y1 = max(0, int((cy - bh / 2 - pad_y) / scale))
            x2 = min(W - 1, int((cx + bw / 2 - pad_x) / scale))
            y2 = min(H - 1, int((cy + bh / 2 - pad_y) / scale))
            result[region] = (x1, y1, x2, y2)
        return result


# ---- Stage 2: ランドマーク検出 ----

class Stage2Detector:
    """HRNet ONNX ラッパー。"""

    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path)

    def detect_region(
        self,
        img_np: np.ndarray,
        region: str,
        box: Tuple[int, int, int, int],
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        crop 内のランドマークを検出し、元画像座標で返す。

        Returns:
            {landmark_key: (i, j)} または None（信頼度低）
        """
        x1, y1, x2, y2 = box
        crop = img_np[y1:y2+1, x1:x2+1]
        if crop.size == 0:
            return {}

        inp, scale_c, pad_cx, pad_cy = _preprocess_crop(crop, _CROP_SIZE)
        heatmaps = self.sess.run(None, {self.sess.get_inputs()[0].name: inp})[0][0]  # (4, H, W)

        channels = REGION_CHANNELS.get(region, [None] * 4)
        result = {}
        for ch_idx, key in enumerate(channels):
            if key is None:
                continue
            hmap = heatmaps[ch_idx]
            lx, ly, conf = _argmax2d(hmap)
            # crop座標 → 元画像座標
            ix = (lx - pad_cx) / scale_c + x1
            iy = (ly - pad_cy) / scale_c + y1
            result[key] = (float(ix), float(iy)) if conf > 0.05 else None
        return result


# ---- 2段階パイプライン ----

def infer_two_stage(
    img_np: np.ndarray,
    stage1: Stage1Detector,
    stage2: Stage2Detector,
    gt_landmarks: Optional[Dict] = None,
    use_gt_boxes: bool = False,
) -> Dict:
    """
    2段階推論を実行する。

    Args:
        img_np: (H, W) float32 画像
        stage1: 椎体検出器（use_gt_boxes=True なら不要）
        stage2: ランドマーク検出器
        gt_landmarks: GT ランドマーク dict（{key: (i, j)}）評価用
        use_gt_boxes: True なら GT座標からBBoxを自動導出して Stage1 をスキップ

    Returns:
        {
            "landmarks": {key: (i, j) | None},
            "boxes":     {region: (x1,y1,x2,y2) | None},
            "mre_mm":    float | None,  # GT がある場合のみ
        }
    """
    H, W = img_np.shape

    if use_gt_boxes and gt_landmarks is not None:
        boxes = derive_bboxes(gt_landmarks, (H, W))
    else:
        boxes = stage1.detect(img_np)

    landmarks: Dict[str, Optional[Tuple[float, float]]] = {}
    for region, box in boxes.items():
        if box is None:
            continue
        region_lm = stage2.detect_region(img_np, region, box)
        landmarks.update(region_lm)

    result = {"landmarks": landmarks, "boxes": boxes}

    if gt_landmarks is not None:
        errors = []
        for key, pred in landmarks.items():
            gt = gt_landmarks.get(key)
            if pred is not None and gt is not None:
                di = pred[0] - gt[0]
                dj = pred[1] - gt[1]
                errors.append(np.hypot(di, dj))
        result["mre_px"] = float(np.mean(errors)) if errors else None
    return result


# ---- CLI ----

def _load_image(npy_path: str) -> np.ndarray:
    img = np.load(npy_path)
    if img.ndim == 3:
        img = img[0]
    return img.astype(np.float32)


def _load_gt(json_path: str) -> Optional[Dict[str, Optional[Tuple[float, float]]]]:
    with open(json_path) as f:
        meta = json.load(f)
    lm = meta.get("landmarks_ijk", {})
    result = {}
    for key in ALL_LANDMARK_KEYS:
        entry = lm.get(key)
        if entry and entry.get("i") is not None:
            result[key] = (float(entry["i"]), float(entry["j"]))
        else:
            result[key] = None
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase2 two-stage inference")
    parser.add_argument("--stage1", required=True, help="Stage1 YOLOv8 ONNX")
    parser.add_argument("--stage2", required=True, help="Stage2 HRNet ONNX")
    parser.add_argument("--image", help="単一 npy ファイル")
    parser.add_argument("--dir", help="ディレクトリ一括評価")
    parser.add_argument("--use-gt-boxes", action="store_true",
                        help="Stage1 をスキップして GT 座標から BBox を導出（評価用）")
    parser.add_argument("--spacing", type=float, default=0.429,
                        help="ピクセル間隔 (mm)。MRE を mm 単位で表示するために使用")
    args = parser.parse_args()

    stage1 = Stage1Detector(args.stage1)
    stage2 = Stage2Detector(args.stage2)
    spacing = args.spacing

    if args.image:
        img_np = _load_image(args.image)
        json_path = args.image.replace("_image.npy", "_landmarks.json")
        gt = _load_gt(json_path) if os.path.exists(json_path) else None
        res = infer_two_stage(img_np, stage1, stage2, gt_landmarks=gt, use_gt_boxes=args.use_gt_boxes)
        n = sum(1 for v in res["landmarks"].values() if v is not None)
        print(f"Detected {n} landmarks")
        if res.get("mre_px") is not None:
            print(f"MRE: {res['mre_px']:.2f} px ({res['mre_px'] * spacing:.2f} mm)")

    elif args.dir:
        npy_files = sorted(Path(args.dir).glob("*_image.npy"))
        all_errors = []
        for npy_path in npy_files:
            img_np = _load_image(str(npy_path))
            json_path = str(npy_path).replace("_image.npy", "_landmarks.json")
            gt = _load_gt(json_path) if os.path.exists(json_path) else None
            res = infer_two_stage(img_np, stage1, stage2, gt_landmarks=gt, use_gt_boxes=args.use_gt_boxes)
            if res.get("mre_px") is not None:
                all_errors.append(res["mre_px"])
                print(f"{npy_path.stem}: MRE={res['mre_px'] * spacing:.2f}mm")
        if all_errors:
            print(f"\n全体 MRE: {np.mean(all_errors) * spacing:.2f} ± {np.std(all_errors) * spacing:.2f} mm (n={len(all_errors)})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
