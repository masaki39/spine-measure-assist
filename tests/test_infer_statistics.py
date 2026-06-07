"""
Tests for statistical functions and angle computation in train/eval_lumbar.py.
These guard the clinical evaluation pipeline against unintended regressions.
"""
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.eval_lumbar import (
    bland_altman_stats,
    compute_angles,
    icc_3_1,
    postprocess_heatmaps,
)

import numpy as np


# ---------------------------------------------------------------------------
# bland_altman_stats
# ---------------------------------------------------------------------------

def test_bland_altman_zero_bias():
    ai = [10.0, 20.0, 30.0]
    gt = [10.0, 20.0, 30.0]
    bias, sd, lo, hi = bland_altman_stats(ai, gt)
    assert math.isclose(bias, 0.0, abs_tol=1e-9)
    assert math.isclose(sd, 0.0, abs_tol=1e-9)
    assert math.isclose(lo, 0.0, abs_tol=1e-9)
    assert math.isclose(hi, 0.0, abs_tol=1e-9)


def test_bland_altman_constant_offset():
    # AI always 2 degrees higher → bias=2, LoA=[2-0, 2+0]
    ai = [12.0, 22.0, 32.0]
    gt = [10.0, 20.0, 30.0]
    bias, sd, lo, hi = bland_altman_stats(ai, gt)
    assert math.isclose(bias, 2.0, abs_tol=1e-9)
    assert math.isclose(sd, 0.0, abs_tol=1e-9)


def test_bland_altman_loa_width():
    # Differences: [0, 2, -2] → sd ≈ 2.0
    ai = [10.0, 22.0, 28.0]
    gt = [10.0, 20.0, 30.0]
    bias, sd, lo, hi = bland_altman_stats(ai, gt)
    assert math.isclose(bias, 0.0, abs_tol=1e-9)
    assert hi - lo == pytest.approx(2 * 1.96 * sd, rel=1e-6)


# ---------------------------------------------------------------------------
# icc_3_1
# ---------------------------------------------------------------------------

import pytest


def test_icc_perfect_agreement():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    icc = icc_3_1(vals, vals)
    assert math.isclose(icc, 1.0, abs_tol=1e-9)


def test_icc_range():
    ai = [10.0, 20.0, 30.0, 40.0]
    gt = [12.0, 18.0, 32.0, 38.0]
    icc = icc_3_1(ai, gt)
    assert 0.0 <= icc <= 1.0


def test_icc_single_case_returns_nan():
    icc = icc_3_1([10.0], [12.0])
    assert math.isnan(icc)


# ---------------------------------------------------------------------------
# compute_angles
# ---------------------------------------------------------------------------

# Reference values from logic_angles.py tests and known geometry
KEYS_6 = ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH", "L1_center"]
KEYS_5 = ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH"]


def _make_coords(pts_dict, keys):
    return [pts_dict[k] for k in keys]


def test_compute_angles_ss_horizontal_plate():
    # S1 plate is perfectly horizontal → SS=0
    pts = {
        "L1_ant": (100, 200), "L1_post": (150, 200),
        "S1_ant": (100, 400), "S1_post": (150, 400),  # horizontal
        "FH": (125, 600),
    }
    angles = compute_angles(_make_coords(pts, KEYS_5), KEYS_5)
    assert angles is not None
    assert math.isclose(angles["SS"], 0.0, abs_tol=1e-6)


def test_compute_angles_returns_none_missing_required():
    # FH is missing → should return None
    pts = {"L1_ant": (0, 0), "L1_post": (1, 0), "S1_ant": (0, 5), "S1_post": (1, 5)}
    coords = [pts.get(k, (0, 0)) for k in ["L1_ant", "L1_post", "S1_ant", "S1_post"]]
    keys = ["L1_ant", "L1_post", "S1_ant", "S1_post"]
    result = compute_angles(coords, keys)
    assert result is None


def test_compute_angles_pi_ss_pt_relationship():
    # PI = SS + PT must hold (fundamental pelvic geometry)
    pts = {
        "L1_ant": (100, 200), "L1_post": (160, 210),
        "S1_ant": (90, 500), "S1_post": (140, 480),
        "FH": (80, 700),
    }
    angles = compute_angles(_make_coords(pts, KEYS_5), KEYS_5)
    assert angles is not None
    pi, pt, ss = angles["PI"], angles["PT"], angles["SS"]
    assert math.isclose(pi, ss + pt, abs_tol=0.1)


def test_compute_angles_l1pa_present_with_l1center():
    pts = {
        "L1_ant": (100, 200), "L1_post": (160, 210),
        "S1_ant": (90, 500), "S1_post": (140, 480),
        "FH": (80, 700), "L1_center": (130, 250),
    }
    angles = compute_angles(_make_coords(pts, KEYS_6), KEYS_6)
    assert angles is not None
    assert "L1PA" in angles


def test_compute_angles_l1pa_absent_without_l1center():
    pts = {
        "L1_ant": (100, 200), "L1_post": (160, 210),
        "S1_ant": (90, 500), "S1_post": (140, 480),
        "FH": (80, 700),
    }
    angles = compute_angles(_make_coords(pts, KEYS_5), KEYS_5)
    assert "L1PA" not in angles


def test_compute_angles_matches_logic_angles():
    """compute_angles must produce identical results to logic_angles.compute_angles_from_points."""
    import logic_angles as ref

    pts_dict = {
        "L1_ant": (100.0, 200.0), "L1_post": (160.0, 210.0),
        "S1_ant": (90.0, 500.0), "S1_post": (140.0, 480.0),
        "FH": (80.0, 700.0), "L1_center": (130.0, 250.0),
    }
    ref_angles = ref.compute_angles_from_points(pts_dict)
    our_angles = compute_angles(_make_coords(pts_dict, KEYS_6), KEYS_6)

    for key in ["PI", "PT", "SS", "LL", "L1PA"]:
        assert math.isclose(ref_angles[key], our_angles[key], abs_tol=1e-6), \
            f"{key}: ref={ref_angles[key]}, ours={our_angles[key]}"


# ---------------------------------------------------------------------------
# postprocess_heatmaps — confidence return
# ---------------------------------------------------------------------------

def test_postprocess_returns_confidence():
    hm = np.zeros((1, 2, 32, 32), dtype=np.float32)
    hm[0, 0, 10, 20] = 0.8
    hm[0, 1, 5, 15] = 0.6
    coords, confs = postprocess_heatmaps(hm)
    assert math.isclose(confs[0], 0.8, abs_tol=1e-6)
    assert math.isclose(confs[1], 0.6, abs_tol=1e-6)


def test_postprocess_flat_heatmap_low_confidence():
    # All zeros → peak value = 0 → low confidence
    hm = np.zeros((1, 1, 32, 32), dtype=np.float32)
    _, confs = postprocess_heatmaps(hm)
    assert confs[0] == 0.0
