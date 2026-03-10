"""
Unit tests for src/validation.py and src/pairs.py validation utilities.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.validation import (
    validate_pair_list,
    validate_config,
    validate_scores,
    validate_metric,
    validate_feature_mode,
    check_no_split_leakage,
)
from src.pairs import validate_pairs


# ---------------------------------------------------------------------------
# validate_pair_list
# ---------------------------------------------------------------------------

def _make_pairs(n_pos=5, n_neg=5):
    pairs = []
    for i in range(n_pos):
        pairs.append({"left_idx": i, "right_idx": i + 100,
                      "label": 1, "left_name": "Alice", "right_name": "Alice"})
    for i in range(n_neg):
        pairs.append({"left_idx": i + 200, "right_idx": i + 300,
                      "label": 0, "left_name": "Alice", "right_name": "Bob"})
    return pairs


def test_validate_pair_list_ok():
    validate_pair_list(_make_pairs())  # should not raise


def test_validate_pair_list_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_pair_list([])


def test_validate_pair_list_bad_label():
    pairs = _make_pairs()
    pairs[0]["label"] = 2
    with pytest.raises(ValueError, match="label"):
        validate_pair_list(pairs)


def test_validate_pair_list_same_index():
    pairs = _make_pairs()
    pairs[0]["right_idx"] = pairs[0]["left_idx"]
    with pytest.raises(ValueError, match="identical"):
        validate_pair_list(pairs)


def test_validate_pair_list_all_positives():
    pairs = _make_pairs(n_pos=10, n_neg=0)
    with pytest.raises(ValueError, match="No negative"):
        validate_pair_list(pairs)


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

def _base_config():
    return {
        "seed": 42,
        "data": {"source": "sklearn_lfw", "resize": 0.5,
                 "min_faces_baseline": 2, "min_faces_filtered": 5},
        "split_policy": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "pair_policy": {"positives_per_identity": 5, "negatives_per_identity": 5,
                        "max_pairs_per_identity": 5},
        "features": {"hog": {"orientations": 8, "pixels_per_cell": [8, 8],
                             "cells_per_block": [2, 2]},
                     "pixel": {"flatten_size": [32, 32]}},
        "threshold_sweep": {"n_points_coarse": 10, "n_points_fine": 20,
                            "selection_rule": "max_balanced_accuracy"},
        "tracking": {"runs_log": "outputs/runs_log.csv",
                     "plots_dir": "outputs/plots"},
    }


def test_validate_config_ok():
    validate_config(_base_config())  # should not raise


def test_validate_config_missing_key():
    cfg = _base_config()
    del cfg["tracking"]
    with pytest.raises(ValueError, match="missing"):
        validate_config(cfg)


def test_validate_config_bad_ratios():
    cfg = _base_config()
    cfg["split_policy"]["train_ratio"] = 0.5  # sum = 0.8, not 1.0
    with pytest.raises(ValueError, match="sum"):
        validate_config(cfg)


def test_validate_config_zero_positives():
    cfg = _base_config()
    cfg["pair_policy"]["positives_per_identity"] = 0
    with pytest.raises(ValueError):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# validate_scores
# ---------------------------------------------------------------------------

def test_validate_scores_ok():
    validate_scores(np.array([0.1, 0.9, 0.5]), np.array([0, 1, 0]))


def test_validate_scores_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        validate_scores(np.array([0.1, 0.2]), np.array([0, 1, 0]))


def test_validate_scores_nan():
    with pytest.raises(ValueError, match="NaN"):
        validate_scores(np.array([0.1, float("nan")]), np.array([0, 1]))


def test_validate_scores_bad_label():
    with pytest.raises(ValueError, match="labels"):
        validate_scores(np.array([0.1, 0.9]), np.array([0, 2]))


# ---------------------------------------------------------------------------
# validate_metric / validate_feature_mode
# ---------------------------------------------------------------------------

def test_validate_metric_valid():
    validate_metric("cosine")
    validate_metric("euclidean")


def test_validate_metric_invalid():
    with pytest.raises(ValueError):
        validate_metric("l1")


def test_validate_feature_mode_valid():
    validate_feature_mode("hog")
    validate_feature_mode("pixel")


def test_validate_feature_mode_invalid():
    with pytest.raises(ValueError):
        validate_feature_mode("deep")


# ---------------------------------------------------------------------------
# check_no_split_leakage
# ---------------------------------------------------------------------------

def test_no_leakage_ok():
    check_no_split_leakage(
        {"targets": np.array([0, 1, 2])},
        {"targets": np.array([3, 4])},
        {"targets": np.array([5, 6])},
    )


def test_leakage_detected():
    with pytest.raises(ValueError, match="leakage"):
        check_no_split_leakage(
            {"targets": np.array([0, 1, 2])},
            {"targets": np.array([2, 3])},  # 2 appears in both train and val
            {"targets": np.array([4, 5])},
        )


# ---------------------------------------------------------------------------
# pairs.validate_pairs
# ---------------------------------------------------------------------------

def test_validate_pairs_positive_name_mismatch():
    # Include a negative pair so the balance check passes; the name-mismatch check fires next
    pairs = [
        {"left_idx": 0, "right_idx": 1, "label": 1,
         "left_name": "Alice", "right_name": "Bob"},   # positive with wrong names
        {"left_idx": 2, "right_idx": 3, "label": 0,
         "left_name": "Alice", "right_name": "Bob"},   # valid negative
    ]
    with pytest.raises(ValueError, match="Positive pair"):
        validate_pairs(pairs)


def test_validate_pairs_negative_same_name():
    pairs = [{"left_idx": 0, "right_idx": 1, "label": 0,
               "left_name": "Alice", "right_name": "Alice"},
             {"left_idx": 2, "right_idx": 3, "label": 1,
               "left_name": "Alice", "right_name": "Alice"}]
    with pytest.raises(ValueError, match="Negative pair"):
        validate_pairs(pairs)
