"""
ONNX inference and evaluation for exported heatmap model.
Usage:
  uv run python train/infer_onnx.py --model best.onnx --image sample_image.npy --json sample_landmarks.json
  uv run python train/infer_onnx.py --model best.onnx --dir dataset/
  uv run python train/infer_onnx.py --model best.onnx --dir dataset/ --splits runs/splits.json --subset test
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


# ---------------------------------------------------------------------------
# Angle computation (mirrors SagittalMeasureAssist/lib/logic_angles.py)
# ---------------------------------------------------------------------------

def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _length(v):
    return math.hypot(v[0], v[1])


def _signed_slope(v):
    ang = math.degrees(math.atan2(v[1], v[0]))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return -ang


def _signed_vert(v):
    ang = math.degrees(math.atan2(v[0], -v[1]))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return ang


def _wrap(a):
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a


def compute_angles(coords, landmark_keys):
    """Compute PI/PT/SS/LL/L1PA from predicted (x,y) coords in any coordinate frame."""
    pts = {k: xy for k, xy in zip(landmark_keys, coords)}
    required = ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH"]
    if not all(k in pts for k in required):
        return None

    FH = pts["FH"]
    S1_ant, S1_post = pts["S1_ant"], pts["S1_post"]
    L1_ant, L1_post = pts["L1_ant"], pts["L1_post"]

    v_S1 = _vec(S1_ant, S1_post)
    v_L1 = _vec(L1_ant, L1_post)
    S1_mid = ((S1_ant[0] + S1_post[0]) / 2, (S1_ant[1] + S1_post[1]) / 2)
    v_pelvis = _vec(FH, S1_mid)

    l1, l2 = _length(v_pelvis), _length(v_S1)
    if l1 == 0 or l2 == 0:
        return None

    dot = v_pelvis[0] * v_S1[0] + v_pelvis[1] * v_S1[1]
    cos_t = max(min(dot / (l1 * l2), 1.0), -1.0)
    theta = math.degrees(math.acos(cos_t))

    result = {
        "PI": abs(90.0 - theta),
        "PT": _signed_vert(v_pelvis),
        "SS": _signed_slope(v_S1),
        "LL": _wrap(_signed_slope(v_S1) - _signed_slope(v_L1)),
    }
    if "L1_center" in pts:
        L1c = pts["L1_center"]
        v1 = _vec(FH, S1_mid)
        v2 = _vec(FH, L1c)
        result["L1PA"] = -math.degrees(math.atan2(
            v1[0] * v2[1] - v1[1] * v2[0],
            v1[0] * v2[0] + v1[1] * v2[1],
        ))
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _ci95(values):
    """95% CI of the mean via t-distribution (or ±1.96*SE for large n)."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    # use 1.96 as approximation (close enough for n>=30, conservative otherwise)
    se = sd / math.sqrt(n)
    return mean - 1.96 * se, mean + 1.96 * se


def _sd(values):
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def bland_altman_stats(ai_vals, gt_vals):
    """Return (bias, sd_diff, loa_lo, loa_hi) for Bland-Altman analysis."""
    diffs = [a - g for a, g in zip(ai_vals, gt_vals)]
    n = len(diffs)
    bias = sum(diffs) / n
    sd = _sd(diffs)
    return bias, sd, bias - 1.96 * sd, bias + 1.96 * sd


def icc_3_1(ai_vals, gt_vals):
    """ICC(3,1) consistency: two-way mixed effects, single measurement."""
    n = len(ai_vals)
    if n < 2:
        return float("nan")
    all_vals = ai_vals + gt_vals
    grand = sum(all_vals) / (2 * n)
    row_means = [(a + g) / 2 for a, g in zip(ai_vals, gt_vals)]
    ss_between = 2 * sum((r - grand) ** 2 for r in row_means)
    col_means = [sum(ai_vals) / n, sum(gt_vals) / n]
    ss_rater = n * sum((c - grand) ** 2 for c in col_means)
    ss_total = sum((v - grand) ** 2 for v in all_vals)
    ss_error = ss_total - ss_between - ss_rater
    ms_b = ss_between / (n - 1)
    ms_e = max(ss_error / (n - 1), 1e-12)  # k=2, df_error=(n-1)*(k-1)=n-1
    icc = (ms_b - ms_e) / (ms_b + ms_e)
    return icc


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image", help=".npy image path")
    p.add_argument("--json", help="Optional landmarks json to compare")
    p.add_argument("--dir", help="Dataset directory for batch MRE evaluation")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--splits", default=None, help="splits.json from train.py (for test-set evaluation)")
    p.add_argument("--subset", default="all", choices=["train", "val", "test", "all"],
                   help="Which split to evaluate (requires --splits)")
    return p.parse_args()


def get_model_resize(sess, args_resize):
    if args_resize is not None:
        return args_resize
    shape = sess.get_inputs()[0].shape
    return [shape[2], shape[3]]


def preprocess(img_np, resize):
    if img_np.ndim == 3:
        img_np = img_np[0]
    img_np = _percentile_clip_norm(img_np)
    t = torch.from_numpy(img_np).unsqueeze(0)
    t, scale, pad_x, pad_y = _resize_with_padding(t, tuple(resize))
    return t.unsqueeze(0), scale, pad_x, pad_y


def postprocess_heatmaps(hm: np.ndarray):
    """Returns (coords, confidences). coords: list of (x,y). confidences: list of peak values."""
    hm = hm[0]
    coords, confs = [], []
    for c in hm:
        idx = np.argmax(c)
        y, x = np.unravel_index(idx, c.shape)
        coords.append((float(x), float(y)))
        confs.append(float(c[y, x]))
    return coords, confs


def _load_gt(json_path):
    """Load GT coords (i, j), spacing, and GT angles from JSON."""
    with open(json_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    lm = meta["landmarks_ijk"]
    gt_coords = [(lm[k]["i"], lm[k]["j"]) for k in LANDMARK_ORDER if k in lm]
    gt_keys = [k for k in LANDMARK_ORDER if k in lm]
    spacing = None
    if "metadata" in meta and "spacing" in meta["metadata"]:
        spacing = meta["metadata"]["spacing"][0]
    gt_angles = meta.get("angles_deg", {})
    return gt_coords, gt_keys, spacing, gt_angles


def _compute_errors(pred_coords, gt_coords, scale, pad_x, pad_y, spacing):
    errors_px, errors_mm = [], []
    for (px, py), (gx, gy) in zip(pred_coords, gt_coords):
        x_orig = (px - pad_x) / scale
        y_orig = (py - pad_y) / scale
        re_px = math.sqrt((x_orig - gx) ** 2 + (y_orig - gy) ** 2)
        errors_px.append(re_px)
        errors_mm.append(re_px * spacing if spacing is not None else None)
    return errors_px, errors_mm


def _back_project(coords, scale, pad_x, pad_y):
    return [((x - pad_x) / scale, (y - pad_y) / scale) for x, y in coords]


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

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

    # Filter by split subset if requested
    if args.splits and args.subset != "all":
        with open(args.splits, "r") as fp:
            splits = json.load(fp)
        allowed = set(splits[args.subset])
        samples = [(b, n, j) for b, n, j in samples if b in allowed]
        print(f"Evaluating subset='{args.subset}' ({len(samples)} cases)")

    n = len(samples)
    all_lm_keys = LANDMARK_ORDER

    # per-landmark storage
    all_errors_px = {k: [] for k in all_lm_keys}
    all_errors_mm = {k: [] for k in all_lm_keys}
    all_confs = {k: [] for k in all_lm_keys}

    # per-angle storage: ai_val, gt_val
    angle_names = ["PI", "PT", "SS", "LL", "L1PA"]
    ai_angles = {a: [] for a in angle_names}
    gt_angles_store = {a: [] for a in angle_names}

    low_conf_cases = []
    outlier_cases = []

    for case_id, npy_path, json_path in samples:
        img_np = np.load(npy_path)
        inp_t, scale, pad_x, pad_y = preprocess(img_np, resize)
        ort_out = ort_sess.run(None, {"image": inp_t.numpy()})
        pred_coords, confs = postprocess_heatmaps(ort_out[0])

        gt_coords, gt_keys, spacing, gt_ang = _load_gt(json_path)
        errors_px, errors_mm = _compute_errors(pred_coords, gt_coords, scale, pad_x, pad_y, spacing)

        for i, k in enumerate(all_lm_keys):
            if i < len(errors_px):
                all_errors_px[k].append(errors_px[i])
                if errors_mm[i] is not None:
                    all_errors_mm[k].append(errors_mm[i])
                all_confs[k].append(confs[i])

        # Confidence check
        if any(c < 0.05 for c in confs):
            low_conf_cases.append((case_id, min(confs)))

        # Outlier check
        valid_mm = [e for e in errors_mm if e is not None]
        if valid_mm and max(valid_mm) > 10.0:
            outlier_cases.append((case_id, max(valid_mm)))

        # Angle evaluation
        pred_orig = _back_project(pred_coords, scale, pad_x, pad_y)
        ai_ang = compute_angles(pred_orig, all_lm_keys)
        if ai_ang and gt_ang:
            for ang in angle_names:
                if ang in ai_ang and ang in gt_ang:
                    ai_angles[ang].append(ai_ang[ang])
                    gt_angles_store[ang].append(gt_ang[ang])

    # -----------------------------------------------------------------------
    # Print: Landmark MRE
    # -----------------------------------------------------------------------
    sep = "-" * 72
    print(f"\n=== Landmark Detection (N={n}) ===")
    hdr = f"{'Landmark':<12}  {'MRE(mm)':>8}  {'SD(mm)':>7}  {'95%CI':>14}  {'SDR@2':>6}  {'SDR@4':>6}  {'Conf':>5}"
    print(hdr)
    print(sep)

    overall_mm, overall_conf = [], []
    for k in all_lm_keys:
        emm = all_errors_mm[k]
        cvals = all_confs[k]
        if not emm:
            continue
        mean_mm = sum(emm) / len(emm)
        sd_mm = _sd(emm)
        ci_lo, ci_hi = _ci95(emm)
        sdr2 = sum(1 for v in emm if v <= 2.0) / len(emm) * 100
        sdr4 = sum(1 for v in emm if v <= 4.0) / len(emm) * 100
        mean_conf = sum(cvals) / len(cvals)
        print(f"{k:<12}  {mean_mm:>8.2f}  {sd_mm:>7.2f}  [{ci_lo:>5.2f},{ci_hi:>5.2f}]  {sdr2:>5.1f}%  {sdr4:>5.1f}%  {mean_conf:>5.3f}")
        overall_mm.extend(emm)
        overall_conf.extend(cvals)

    print(sep)
    if overall_mm:
        m = sum(overall_mm) / len(overall_mm)
        s = _sd(overall_mm)
        ci_lo, ci_hi = _ci95(overall_mm)
        sdr2 = sum(1 for v in overall_mm if v <= 2.0) / len(overall_mm) * 100
        sdr4 = sum(1 for v in overall_mm if v <= 4.0) / len(overall_mm) * 100
        mc = sum(overall_conf) / len(overall_conf)
        print(f"{'Overall':<12}  {m:>8.2f}  {s:>7.2f}  [{ci_lo:>5.2f},{ci_hi:>5.2f}]  {sdr2:>5.1f}%  {sdr4:>5.1f}%  {mc:>5.3f}")

    if outlier_cases:
        print(f"\nOutliers (any landmark >10mm): {len(outlier_cases)}/{n} ({100*len(outlier_cases)/n:.1f}%)")
        for cid, worst in sorted(outlier_cases, key=lambda x: -x[1])[:5]:
            print(f"  {cid}: worst={worst:.1f}mm")
    else:
        print(f"\nOutliers (any landmark >10mm): 0/{n}")

    # -----------------------------------------------------------------------
    # Print: Angle evaluation
    # -----------------------------------------------------------------------
    print(f"\n=== Angle Accuracy — AI vs GT (Bland-Altman) ===")
    hdr2 = f"{'Angle':<6}  {'N':>4}  {'MAE(°)':>7}  {'SD(°)':>6}  {'Bias(°)':>8}  {'LoA_lo':>7}  {'LoA_hi':>7}  {'ICC':>5}"
    print(hdr2)
    print(sep)

    all_ang_ae = []
    for ang in angle_names:
        ai_v = ai_angles[ang]
        gt_v = gt_angles_store[ang]
        if len(ai_v) < 2:
            print(f"{ang:<6}  {'n/a':>4}")
            continue
        abs_errs = [abs(a - g) for a, g in zip(ai_v, gt_v)]
        mae = sum(abs_errs) / len(abs_errs)
        sd = _sd(abs_errs)
        bias, sd_diff, loa_lo, loa_hi = bland_altman_stats(ai_v, gt_v)
        icc = icc_3_1(ai_v, gt_v)
        print(f"{ang:<6}  {len(ai_v):>4}  {mae:>7.2f}  {sd:>6.2f}  {bias:>8.2f}  {loa_lo:>7.2f}  {loa_hi:>7.2f}  {icc:>5.3f}")
        all_ang_ae.extend(abs_errs)

    print(sep)
    if all_ang_ae:
        print(f"{'Overall':<6}  {len(all_ang_ae):>4}  {sum(all_ang_ae)/len(all_ang_ae):>7.2f}  {_sd(all_ang_ae):>6.2f}")

    # -----------------------------------------------------------------------
    # Print: Confidence summary
    # -----------------------------------------------------------------------
    if low_conf_cases:
        print(f"\n=== Low-confidence cases (peak < 0.05): {len(low_conf_cases)}/{n} ===")
        for cid, min_c in sorted(low_conf_cases, key=lambda x: x[1])[:10]:
            print(f"  {cid}: min_conf={min_c:.4f}")
    else:
        print(f"\nAll {n} cases had heatmap confidence >= 0.05")


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

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
    coords, confs = postprocess_heatmaps(ort_out[0])

    print("Predicted coords (x,y) in resized space:")
    for name, (x, y), c in zip(LANDMARK_ORDER, coords, confs):
        print(f"  {name}: ({x:.1f}, {y:.1f})  conf={c:.4f}")

    # Compute and show angles
    orig_coords = _back_project(coords, scale, 0, 0)  # no padding info for single image
    angles = compute_angles(orig_coords, LANDMARK_ORDER)
    if angles:
        print("\nPredicted angles:")
        for ang, val in angles.items():
            print(f"  {ang}: {val:.2f}°")

    if args.json and os.path.exists(args.json):
        with open(args.json, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        gt_orig = [(meta["landmarks_ijk"][k]["i"], meta["landmarks_ijk"][k]["j"]) for k in LANDMARK_ORDER]
        gt_resized = [(x * scale + 0, y * scale + 0) for (x, y) in gt_orig]
        print("\nGround truth coords (IJK scaled to resized space):")
        for name, (x, y) in zip(LANDMARK_ORDER, gt_resized):
            print(f"  {name}: ({x:.1f}, {y:.1f})")

        gt_ang = meta.get("angles_deg", {})
        if gt_ang and angles:
            print("\nAngle comparison (AI vs GT):")
            for ang in ["PI", "PT", "SS", "LL", "L1PA"]:
                if ang in angles and ang in gt_ang:
                    print(f"  {ang}: AI={angles[ang]:.2f}°  GT={gt_ang[ang]:.2f}°  diff={angles[ang]-gt_ang[ang]:.2f}°")


if __name__ == "__main__":
    main()
