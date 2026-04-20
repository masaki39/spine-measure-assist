from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from logic_angles import compute_angles_from_points


@dataclass
class MeasurementSetDef:
    name: str
    point_labels: List[str]
    point_instructions: List[str]
    angle_names: List[str]
    compute_fn: Callable[[Dict[str, Tuple[float, float]]], Dict[str, float]]
    vector_definitions: Dict[str, Tuple[str, str]]
    vector_modes: Dict[str, str]
    vector_colors: Dict[str, Tuple[float, float, float]]
    midpoint_definitions: Dict[str, Tuple[str, str]]


PELVIC_SET = MeasurementSetDef(
    name="Pelvic Parameters",
    point_labels=["L1_ant", "L1_post", "S1_ant", "S1_post", "FH"],
    point_instructions=[
        "L1_ant  — L1 superior endplate, anterior edge",
        "L1_post — L1 superior endplate, posterior edge",
        "S1_ant  — S1 superior endplate, anterior edge",
        "S1_post — S1 superior endplate, posterior edge",
        "FH      — femoral head center (bilateral average)",
    ],
    angle_names=["PI", "PT", "SS", "LL"],
    compute_fn=compute_angles_from_points,
    vector_definitions={
        "L1":     ("L1_ant",  "L1_post"),
        "S1":     ("S1_ant",  "S1_post"),
        "pelvis": ("FH",      "_S1_mid"),
    },
    vector_modes={
        "L1":     "Line",
        "S1":     "Line",
        "pelvis": "Segment",
    },
    vector_colors={
        "L1":     (0.2, 0.8, 1.0),
        "S1":     (1.0, 0.6, 0.1),
        "pelvis": (0.4, 1.0, 0.4),
    },
    midpoint_definitions={
        "_S1_mid": ("S1_ant", "S1_post"),
    },
)

_ALL_SETS: Dict[str, MeasurementSetDef] = {s.name: s for s in [PELVIC_SET]}


def set_names() -> List[str]:
    return list(_ALL_SETS.keys())


def get_set(name: str) -> MeasurementSetDef:
    return _ALL_SETS[name]
