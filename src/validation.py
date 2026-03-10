"""
Pipeline validation checks — fail fast before wasting compute.
"""
import os
from typing import List, Dict


REQUIRED_PAIR_KEYS = {"left_idx", "right_idx", "label", "left_name", "right_name"}
VALID_SPLITS = {"train", "val", "test"}
VALID_METRICS = {"cosine", "euclidean"}
VALID_FEATURE_MODES = {"hog", "pixel"}


def validate_pair_list(pairs: List[Dict]) -> None:
    """Validate a list of pair dicts produced by src.pairs.build_pairs."""
    if not pairs:
        raise ValueError("Pair list is empty.")
    for i, p in enumerate(pairs):
        missing = REQUIRED_PAIR_KEYS - set(p.keys())
        if missing:
            raise ValueError(f"Pair {i} missing keys: {missing}")
        if p["label"] not in (0, 1):
            raise ValueError(f"Pair {i} has invalid label {p['label']!r} (must be 0 or 1).")
        if p["left_idx"] == p["right_idx"]:
            raise ValueError(f"Pair {i} has identical left/right indices ({p['left_idx']}).")
    labels = [p["label"] for p in pairs]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("No positive pairs found.")
    if n_neg == 0:
        raise ValueError("No negative pairs found.")


def validate_config(config: dict) -> None:
    """Validate required config fields and value ranges."""
    required_top = {"seed", "data", "split_policy", "pair_policy",
                    "features", "threshold_sweep", "tracking"}
    missing = required_top - set(config.keys())
    if missing:
        raise ValueError(f"Config missing top-level keys: {missing}")

    ratios = config["split_policy"]
    total = ratios["train_ratio"] + ratios["val_ratio"] + ratios["test_ratio"]
    if not (0.99 < total < 1.01):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    pos = config["pair_policy"]["positives_per_identity"]
    neg = config["pair_policy"]["negatives_per_identity"]
    if pos < 1 or neg < 1:
        raise ValueError("positives/negatives_per_identity must be >= 1")

    n = config["threshold_sweep"]["n_points_coarse"]
    if n < 2:
        raise ValueError("threshold_sweep.n_points_coarse must be >= 2")


def validate_scores(scores, labels) -> None:
    """Check that scores array aligns with labels and contains finite values."""
    import numpy as np
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores shape {scores.shape} != labels shape {labels.shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or Inf values.")
    if not set(labels).issubset({0, 1}):
        raise ValueError("labels must only contain 0 and 1.")


def validate_metric(metric: str) -> None:
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")


def validate_feature_mode(mode: str) -> None:
    if mode not in VALID_FEATURE_MODES:
        raise ValueError(
            f"feature mode must be one of {VALID_FEATURE_MODES}, got {mode!r}"
        )


def check_no_split_leakage(train_data: dict, val_data: dict, test_data: dict) -> None:
    """Confirm no identity appears in more than one split."""
    import numpy as np
    train_ids = set(np.unique(train_data["targets"]))
    val_ids = set(np.unique(val_data["targets"]))
    test_ids = set(np.unique(test_data["targets"]))
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    if tv or tt or vt:
        raise ValueError(
            f"Split leakage detected — train∩val={tv}, train∩test={tt}, val∩test={vt}"
        )
