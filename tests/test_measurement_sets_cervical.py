"""Tests for CervicalMeasureAssist measurement set definition."""

import importlib.util
import os
import sys

import pytest


def _load_cervical_measurement_sets():
    """Load measurement_sets.py from CervicalMeasureAssist/lib explicitly."""
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "CervicalMeasureAssist", "lib", "cervical_measurement_sets.py")
    )
    spec = importlib.util.spec_from_file_location("measurement_sets_cervical", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_ms = _load_cervical_measurement_sets()
CERVICAL_SET = _ms.CERVICAL_SET


def test_cervical_set_has_eight_labels():
    assert len(CERVICAL_SET.point_labels) == 8


def test_cervical_set_labels_unique():
    assert len(set(CERVICAL_SET.point_labels)) == len(CERVICAL_SET.point_labels)


def test_cervical_set_has_three_angle_names():
    assert set(CERVICAL_SET.angle_names) == {"C2C7_angle", "T1S", "C2C7_SVA"}


def test_cervical_set_sva_unit_is_mm():
    assert CERVICAL_SET.value_units.get("C2C7_SVA") == "mm"


def test_cervical_set_angle_units_default_to_degree():
    for name in ("C2C7_angle", "T1S"):
        assert name not in CERVICAL_SET.value_units or CERVICAL_SET.value_units[name] == "°"


def test_cervical_set_vector_endpoints_valid():
    all_labels = set(CERVICAL_SET.point_labels) | set(CERVICAL_SET.midpoint_definitions.keys())
    for name, (p1, p2) in CERVICAL_SET.vector_definitions.items():
        assert p1 in all_labels, f"Vector {name}: endpoint '{p1}' not in labels"
        assert p2 in all_labels, f"Vector {name}: endpoint '{p2}' not in labels"


def test_cervical_set_vector_modes_valid():
    for name, mode in CERVICAL_SET.vector_modes.items():
        assert mode in ("Line", "Segment"), f"Invalid mode '{mode}' for vector {name}"


def test_cervical_set_colors_in_range():
    for name, rgb in CERVICAL_SET.vector_colors.items():
        assert len(rgb) == 3
        for c in rgb:
            assert 0.0 <= c <= 1.0, f"Color component out of range for vector {name}: {rgb}"


def test_cervical_set_compute_fn_returns_dict():
    pts = {
        "C2_ant":      (0.0, 0.0),
        "C2_post":     (1.0, 0.0),
        "C2_center":   (0.5, 0.0),
        "C7_inf_ant":  (0.0, 5.0),
        "C7_inf_post": (1.0, 5.0),
        "C7_sup_post": (1.0, 4.5),
        "T1_ant":      (0.0, 6.0),
        "T1_post":     (1.0, 6.0),
    }
    result = CERVICAL_SET.compute_fn(pts)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(CERVICAL_SET.angle_names)


def test_cervical_set_no_midpoints_needed():
    assert CERVICAL_SET.midpoint_definitions == {}
