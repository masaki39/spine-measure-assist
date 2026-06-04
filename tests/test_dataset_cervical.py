"""Basic structural tests for the cervical dataset module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.dataset_cervical import LANDMARK_ORDER

EXPECTED_KEYS = [
    "C2_ant", "C2_post", "C2_center",
    "C7_inf_ant", "C7_inf_post", "C7_sup_post",
    "T1_ant", "T1_post",
]


def test_landmark_order_length():
    assert len(LANDMARK_ORDER) == 8


def test_landmark_order_contains_all_required_keys():
    assert set(LANDMARK_ORDER) == set(EXPECTED_KEYS)


def test_landmark_order_no_duplicates():
    assert len(set(LANDMARK_ORDER)) == len(LANDMARK_ORDER)
