"""
Cervical spinal parameter angle and SVA computations.
Self-contained: vector math primitives are inlined here so this module has no
cross-module dependencies and works in both 3D Slicer and offline tests.
"""

import math

# ---------------------------------------------------------------------------
# Shared vector math (same implementations as LumbarMeasureAssist/lib/logic_angles.py)
# ---------------------------------------------------------------------------

def vector_from_points(a, b):
    return (b[0] - a[0], b[1] - a[1])


def vector_length(v):
    return math.hypot(v[0], v[1])


def signed_slope_angle_deg(v):
    """Signed angle to horizontal in [-90, 90]; y increases downward, so sign flipped."""
    if vector_length(v) == 0:
        raise ValueError("Cannot compute slope for zero-length vector.")
    ang = math.degrees(math.atan2(v[1], v[0]))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return -ang


def wrap_signed_angle(angle):
    """Wrap to [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def lumbosacral_lordosis_deg(v_top, v_bottom):
    """Signed Cobb: slope(bottom) - slope(top), wrapped to [-180, 180]."""
    return wrap_signed_angle(signed_slope_angle_deg(v_bottom) - signed_slope_angle_deg(v_top))


# ---------------------------------------------------------------------------
# Cervical-specific calculations
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    "C2_ant",
    "C2_post",
    "C2_center",
    "C7_inf_ant",
    "C7_inf_post",
    "C7_sup_post",
    "T1_ant",
    "T1_post",
]


def c2c7_angle_deg(C2_ant, C2_post, C7_inf_ant, C7_inf_post):
    """
    C2-C7 lordosis Cobb angle: slope(C7_inf) - slope(C2_inf), wrapped to [-180, 180].
    Same formula as lumbosacral_lordosis_deg.
    Negative values indicate cervical lordosis in standard image coordinates (y-down).
    """
    v_C2 = vector_from_points(C2_ant, C2_post)
    v_C7i = vector_from_points(C7_inf_ant, C7_inf_post)
    return lumbosacral_lordosis_deg(v_C2, v_C7i)


def t1_slope_deg(T1_ant, T1_post):
    """
    T1 slope: signed angle of T1 superior endplate to horizontal [-90, 90].
    Positive when the endplate tilts anterior-downward.
    """
    v_T1 = vector_from_points(T1_ant, T1_post)
    return signed_slope_angle_deg(v_T1)


def c2c7_sva_mm(C2_center, C7_sup_post, pixel_spacing_mm=1.0):
    """
    C2-C7 SVA: signed horizontal distance from C7_sup_post to the vertical line
    through C2_center, in mm.

    Positive = C2 is anterior (larger x) relative to C7 (forward head posture).

    Args:
        C2_center: (x, y) coordinates of C2 vertebral body center.
        C7_sup_post: (x, y) coordinates of C7 superior posterior corner.
        pixel_spacing_mm: mm per pixel (from DICOM PixelSpacing).
            Use 1.0 when coordinates are already in mm (e.g., Slicer RAS space).
            For pixel-space coordinates from training evaluation, supply the actual spacing.
    """
    dx_px = C2_center[0] - C7_sup_post[0]
    return dx_px * pixel_spacing_mm


def compute_cervical_measurements(points, pixel_spacing_mm=1.0):
    """
    Compute C2C7_angle, T1S, and C2C7_SVA from landmark points.

    Args:
        points: dict mapping landmark names to (x, y) coordinates.
        pixel_spacing_mm: mm per pixel for SVA conversion.
            Defaults to 1.0 (mm-space, for use inside 3D Slicer with RAS coordinates).
    Returns:
        dict with keys "C2C7_angle" (deg), "T1S" (deg), "C2C7_SVA" (mm).
    Raises:
        ValueError if any required key is missing.
    """
    missing = [k for k in REQUIRED_KEYS if k not in points]
    if missing:
        raise ValueError(f"Missing points: {', '.join(missing)}")

    return {
        "C2C7_angle": c2c7_angle_deg(
            points["C2_ant"], points["C2_post"],
            points["C7_inf_ant"], points["C7_inf_post"],
        ),
        "T1S": t1_slope_deg(points["T1_ant"], points["T1_post"]),
        "C2C7_SVA": c2c7_sva_mm(points["C2_center"], points["C7_sup_post"], pixel_spacing_mm),
    }
