"""
Cervical spine landmark dataset.
Reuses HeatmapDataset from dataset.py with cervical-specific LANDMARK_ORDER.
"""

import sys as _sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from dataset import HeatmapDataset

LANDMARK_ORDER = [
    "C2_center",
    "C2_ant",
    "C2_post",
    "C7_sup_post",
    "C7_inf_ant",
    "C7_inf_post",
    "T1_ant",
    "T1_post",
]


def make_cervical_dataset(data_dir, **kwargs):
    """
    Create a HeatmapDataset configured for the 8 cervical landmarks.
    All keyword arguments are forwarded to HeatmapDataset.__init__.
    """
    kwargs.setdefault("landmark_keys", LANDMARK_ORDER)
    return HeatmapDataset(data_dir, **kwargs)
