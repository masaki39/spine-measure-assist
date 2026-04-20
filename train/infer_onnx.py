"""
Simple ONNX inference example for exported heatmap model.
Usage:
  uv run python train/infer_onnx.py --model best.onnx --image sample_image.npy --json sample_landmarks.json
  uv run python train/infer_onnx.py --model best.onnx --dir dataset/
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from dataset import LANDMARK_ORDER, _percentile_clip_norm, _resize_with_padding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image", help=".npy image path")
    p.add_argument("--json", help="Optional landmarks json to compare")
    p.add_argument("--dir", help="Dataset directory for batch MRE evaluation")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    return p.parse_args()


def get_model_resize(sess, args_resize):
    if args_resize is not None:
        return args_resize
    shape = sess.get_inputs()[0].shape  # [batch, 1, H, W]
    return [shape[2], shape[3]]


def preprocess(img_np, resize):
    if img_np.ndim == 3:
        img_np = img_np[0]
    img_np = _percentile_clip_norm(img_np)
    t = torch.from_numpy(img_np).unsqueeze(0)  # (1,H,W)
    t, scale, pad_x, pad_y = _resize_with_padding(t, tuple(resize))
    return t.unsqueeze(0), scale, pad_x, pad_y  # (1,1,H,W), ...


def postprocess_heatmaps(hm: np.ndarray):
    # hm: (1, L, H, W)
    hm = hm[0]
    coords = []
    for c in hm:
        idx = np.argmax(c)
        y, x = np.unravel_index(idx, c.shape)
        coords.append((float(x), float(y)))
    return coords


def _load_gt_coords(json_path):
    """Load GT coords (i, j) and spacing from JSON. Returns (gt_coords, spacing_mm)."""
    with open(json_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    lm = meta["landmarks_ijk"]
    gt_coords = [(lm[k]["i"], lm[k]["j"]) for k in LANDMARK_ORDER]
    spacing = None
    if "metadata" in meta and "spacing" in meta["metadata"]:
        spacing = meta["metadata"]["spacing"][0]
    return gt_coords, spacing


def _compute_errors(pred_coords, gt_coords, scale, pad_x, pad_y, spacing):
    """Compute per-landmark radial errors in pixels and mm (original space)."""
    errors_px = []
    errors_mm = []
    for (px, py), (gx, gy) in zip(pred_coords, gt_coords):
        x_orig = (px - pad_x) / scale
        y_orig = (py - pad_y) / scale
        re_px = math.sqrt((x_orig - gx) ** 2 + (y_orig - gy) ** 2)
        errors_px.append(re_px)
        errors_mm.append(re_px * spacing if spacing is not None else None)
    return errors_px, errors_mm


def evaluate_dataset(args):
    ort_sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    resize = get_model_resize(ort_sess, args.resize)
    data_dir = args.dir

    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith("_image.npy"):
            continue
        base = fname.replace("_image.npy", "")
        json_path = os.path.join(data_dir, f"{base}_landmarks.json")
        npy_path = os.path.join(data_dir, fname)
        if os.path.exists(json_path):
            samples.append((base, npy_path, json_path))

    if not samples:
        print(f"No samples found in {data_dir}")
        return

    # per-landmark errors: list of lists
    all_errors_px = [[] for _ in LANDMARK_ORDER]
    all_errors_mm = [[] for _ in LANDMARK_ORDER]

    for case_id, npy_path, json_path in samples:
        img_np = np.load(npy_path)
        inp_t, scale, pad_x, pad_y = preprocess(img_np, resize)
        ort_out = ort_sess.run(None, {"image": inp_t.numpy()})
        pred_coords = postprocess_heatmaps(ort_out[0])

        gt_coords, spacing = _load_gt_coords(json_path)
        errors_px, errors_mm = _compute_errors(pred_coords, gt_coords, scale, pad_x, pad_y, spacing)

        for i, (epx, emm) in enumerate(zip(errors_px, errors_mm)):
            all_errors_px[i].append(epx)
            all_errors_mm[i].append(emm)

    n = len(samples)
    print(f"\n=== MRE Evaluation (N={n} samples) ===")
    header = f"{'Landmark':<12}  {'MRE(px)':>8}  {'MRE(mm)':>8}  {'SDR@2mm':>8}  {'SDR@4mm':>8}"
    print(header)
    print("-" * len(header))

    overall_px = []
    overall_mm = []
    for i, name in enumerate(LANDMARK_ORDER):
        epx = all_errors_px[i]
        emm = [v for v in all_errors_mm[i] if v is not None]
        mre_px = sum(epx) / len(epx)
        mre_mm = sum(emm) / len(emm) if emm else float("nan")
        sdr2 = sum(1 for v in emm if v <= 2.0) / len(emm) * 100 if emm else float("nan")
        sdr4 = sum(1 for v in emm if v <= 4.0) / len(emm) * 100 if emm else float("nan")
        print(f"{name:<12}  {mre_px:>8.2f}  {mre_mm:>8.2f}  {sdr2:>7.1f}%  {sdr4:>7.1f}%")
        overall_px.extend(epx)
        overall_mm.extend(emm)

    print("-" * len(header))
    mre_px = sum(overall_px) / len(overall_px)
    mre_mm = sum(overall_mm) / len(overall_mm) if overall_mm else float("nan")
    sdr2 = sum(1 for v in overall_mm if v <= 2.0) / len(overall_mm) * 100 if overall_mm else float("nan")
    sdr4 = sum(1 for v in overall_mm if v <= 4.0) / len(overall_mm) * 100 if overall_mm else float("nan")
    print(f"{'Overall':<12}  {mre_px:>8.2f}  {mre_mm:>8.2f}  {sdr2:>7.1f}%  {sdr4:>7.1f}%")


def main():
    args = parse_args()

    if args.dir:
        evaluate_dataset(args)
        return

    if not args.image:
        print("Error: --image or --dir is required")
        return

    ort_sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    resize = get_model_resize(ort_sess, args.resize)

    img_np = np.load(args.image)
    inp_t, scale, pad_x, pad_y = preprocess(img_np, resize)
    ort_out = ort_sess.run(None, {"image": inp_t.numpy()})
    coords = postprocess_heatmaps(ort_out[0])

    print("Predicted coords (x,y) in resized space:")
    for name, (x, y) in zip(LANDMARK_ORDER, coords):
        print(f"  {name}: ({x:.1f}, {y:.1f})")

    if args.json and os.path.exists(args.json):
        with open(args.json, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        gt_orig = [(meta["landmarks_ijk"][k]["i"], meta["landmarks_ijk"][k]["j"]) for k in LANDMARK_ORDER]
        gt_resized = [(x * scale + pad_x, y * scale + pad_y) for (x, y) in gt_orig]
        print("\nGround truth in resized space (IJK scaled):")
        for name, (x, y) in zip(LANDMARK_ORDER, gt_resized):
            print(f"  {name}: ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()
