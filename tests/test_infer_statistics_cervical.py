"""
Guards that compute_cervical_angles() in infer_onnx_cervical.py is identical
to compute_cervical_measurements() in logic_angles_cervical.py.
"""

import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.infer_onnx_cervical import compute_cervical_angles
import logic_angles_cervical as ref_module

KEYS = [
    "C2_ant", "C2_post", "C2_center",
    "C7_inf_ant", "C7_inf_post", "C7_sup_post",
    "T1_ant", "T1_post",
]


def _make_coords(pts_dict, keys):
    return [pts_dict[k] for k in keys]


def _sample_points():
    return {
        "C2_ant":      (100.0, 50.0),
        "C2_post":     (160.0, 55.0),
        "C2_center":   (130.0, 52.0),
        "C7_inf_ant":  (95.0,  300.0),
        "C7_inf_post": (155.0, 310.0),
        "C7_sup_post": (155.0, 280.0),
        "T1_ant":      (90.0,  350.0),
        "T1_post":     (150.0, 360.0),
    }


def test_compute_cervical_angles_matches_logic_angles_cervical():
    """infer_onnx_cervical and logic_angles_cervical must produce identical results."""
    pts = _sample_points()
    coords = _make_coords(pts, KEYS)

    ref = ref_module.compute_cervical_measurements(pts, pixel_spacing_mm=1.0)
    ours = compute_cervical_angles(coords, KEYS, spacing_mm=1.0)

    for key in ("C2C7_angle", "T1S", "C2C7_SVA"):
        assert math.isclose(ref[key], ours[key], abs_tol=1e-6), \
            f"{key}: ref={ref[key]}, ours={ours[key]}"


def test_compute_cervical_sva_spacing_applied():
    """spacing_mm=0.5 halves the SVA vs spacing_mm=1.0."""
    pts = _sample_points()
    coords = _make_coords(pts, KEYS)

    r1 = compute_cervical_angles(coords, KEYS, spacing_mm=1.0)
    r05 = compute_cervical_angles(coords, KEYS, spacing_mm=0.5)
    assert math.isclose(r1["C2C7_SVA"], 2 * r05["C2C7_SVA"], abs_tol=1e-9)


def test_compute_cervical_returns_none_if_missing_required():
    """Returns None when a required key is missing."""
    pts = _sample_points()
    pts.pop("T1_ant")
    coords = _make_coords(pts, [k for k in KEYS if k != "T1_ant"])
    keys_without_t1 = [k for k in KEYS if k != "T1_ant"]
    result = compute_cervical_angles(coords, keys_without_t1, spacing_mm=1.0)
    assert result is None
