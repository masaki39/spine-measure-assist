from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from logic_angles_cervical import compute_cervical_measurements


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
    value_units: Dict[str, str] = field(default_factory=dict)


CERVICAL_SET = MeasurementSetDef(
    name="Cervical Parameters",
    point_labels=[
        "C2_center", "C2_ant", "C2_post",
        "C7_sup_post", "C7_inf_ant", "C7_inf_post",
        "T1_ant", "T1_post",
    ],
    point_instructions=[
        "C2_center  — C2 vertebral body center (for SVA)",
        "C2_ant     — C2 inferior endplate, anterior edge",
        "C2_post    — C2 inferior endplate, posterior edge",
        "C7_sup_post— C7 superior posterior corner (SVA plumb reference)",
        "C7_inf_ant — C7 inferior endplate, anterior edge (C2-C7 Cobb angle)",
        "C7_inf_post— C7 inferior endplate, posterior edge (C2-C7 Cobb angle)",
        "T1_ant     — T1 superior endplate, anterior edge (T1 slope)",
        "T1_post    — T1 superior endplate, posterior edge (T1 slope)",
    ],
    angle_names=["C2C7_angle", "T1S", "C2C7_SVA"],
    compute_fn=compute_cervical_measurements,
    vector_definitions={
        "C2":    ("C2_ant",      "C2_post"),
        "C7inf": ("C7_inf_ant",  "C7_inf_post"),
        "T1":    ("T1_ant",      "T1_post"),
        "SVA":   ("C7_sup_post", "C2_center"),
    },
    vector_modes={
        "C2":    "Line",
        "C7inf": "Line",
        "T1":    "Line",
        "SVA":   "Segment",
    },
    vector_colors={
        "C2":    (0.2, 0.8, 1.0),
        "C7inf": (1.0, 0.6, 0.1),
        "T1":    (0.4, 1.0, 0.4),
        "SVA":   (1.0, 0.4, 0.8),
    },
    midpoint_definitions={},
    value_units={"C2C7_SVA": "mm"},
)

_ALL_SETS: Dict[str, MeasurementSetDef] = {s.name: s for s in [CERVICAL_SET]}


def set_names() -> List[str]:
    return list(_ALL_SETS.keys())


def get_set(name: str) -> MeasurementSetDef:
    return _ALL_SETS[name]
