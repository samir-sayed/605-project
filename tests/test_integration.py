"""
Small integration test: synthetic LFW-shaped data → full pipeline → logged run.

This test does NOT download LFW. It creates a tiny synthetic dataset with the
same image shape (62, 47) and runs the full pipeline path:
  split → pairs → feature extraction → scoring → threshold calibration → metrics → run log
"""
import sys
import os
import csv
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.pairs import build_pairs, validate_pairs
from src.features import extract_features
from src.similarity import cosine_similarity_vectorized
from src.evaluation import calibrate_threshold, compute_metrics, compute_roc
from src.tracking import log_run, load_runs
from src.validation import validate_scores, check_no_split_leakage


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset():
    """
    50 images, 10 identities, 5 images each.
    Images for each identity share a distinct mean pixel value so that
    cosine similarity correctly discriminates same vs different identity.
    Image shape matches real LFW: (62, 47) float32 in [0, 1].
    """
    rng = np.random.default_rng(0)
    n_ids = 10
    imgs_per_id = 5
    H, W = 62, 47

    all_images = []
    all_targets = []
    all_names = []

    for identity_id in range(n_ids):
        # Each identity has a bright horizontal stripe at a distinct row position.
        # This creates identity-specific HOG gradient edges that are easily discriminated.
        base = np.zeros((H, W), dtype=np.float32)
        band_start = (identity_id * H) // n_ids
        band_end = ((identity_id + 1) * H) // n_ids
        base[band_start:band_end, :] = 1.0
        for _ in range(imgs_per_id):
            img = np.clip(base + rng.uniform(-0.05, 0.05, (H, W)).astype(np.float32), 0, 1)
            all_images.append(img)
            all_targets.append(identity_id)
            all_names.append(f"person_{identity_id:02d}")

    all_images = np.stack(all_images)   # (50, 62, 47)
    all_targets = np.array(all_targets)
    all_names = np.array(all_names)

    # Identity-based splits (manually assign to avoid importing full ingestion)
    train_ids = set(range(6))   # identities 0-5
    val_ids = {6, 7}            # identities 6-7
    test_ids = {8, 9}           # identities 8-9

    def _subset(id_set):
        mask = np.array([t in id_set for t in all_targets])
        return {
            "images": all_images[mask],
            "targets": all_targets[mask],
            "names": all_names[mask],
        }

    return {
        "train": _subset(train_ids),
        "val": _subset(val_ids),
        "test": _subset(test_ids),
    }


BASE_CONFIG = {
    "seed": 42,
    "data": {"min_faces_filtered": 5},
    "pair_policy": {
        "positives_per_identity": 3,
        "negatives_per_identity": 3,
        "max_pairs_per_identity": 3,
    },
    "features": {
        "hog": {"orientations": 8, "pixels_per_cell": [8, 8], "cells_per_block": [2, 2]},
        "pixel": {"flatten_size": [32, 32]},
    },
}


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_full_pipeline_val_and_test(synthetic_dataset, tmp_path):
    ds = synthetic_dataset

    # 1. Verify no split leakage
    check_no_split_leakage(ds["train"], ds["val"], ds["test"])

    # 2. Build pairs for val and test
    val_pairs = build_pairs(
        ds["val"]["images"], ds["val"]["targets"], ds["val"]["names"],
        config=BASE_CONFIG, mode="baseline", seed=42,
    )
    test_pairs = build_pairs(
        ds["test"]["images"], ds["test"]["targets"], ds["test"]["names"],
        config=BASE_CONFIG, mode="baseline", seed=42,
    )
    validate_pairs(val_pairs)
    validate_pairs(test_pairs)

    assert len(val_pairs) > 0
    assert len(test_pairs) > 0

    # 3. Extract HOG features
    val_features = extract_features(ds["val"]["images"], BASE_CONFIG, mode="hog")
    test_features = extract_features(ds["test"]["images"], BASE_CONFIG, mode="hog")

    assert val_features.shape[0] == len(ds["val"]["images"])
    assert test_features.shape[1] == val_features.shape[1]  # same feature dim

    # 4. Score pairs
    def _score(pairs, features):
        left = np.stack([features[p["left_idx"]] for p in pairs])
        right = np.stack([features[p["right_idx"]] for p in pairs])
        return cosine_similarity_vectorized(left, right)

    val_scores = _score(val_pairs, val_features)
    test_scores = _score(test_pairs, test_features)

    val_labels = np.array([p["label"] for p in val_pairs])
    test_labels = np.array([p["label"] for p in test_pairs])

    validate_scores(val_scores, val_labels)
    validate_scores(test_scores, test_labels)

    # 5. Calibrate threshold on val
    thresh, sweep = calibrate_threshold(val_scores, val_labels, n_points=5, metric="cosine")
    assert isinstance(thresh, float)
    assert len(sweep) == 5

    # 6. Evaluate on test
    metrics = compute_metrics(test_scores, test_labels, thresh, metric="cosine")
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # 7. ROC on test
    fpr, tpr, _, auc = compute_roc(test_scores, test_labels, metric="cosine")
    assert 0.0 <= auc <= 1.0

    # 8. Log run and verify CSV is written
    log_path = str(tmp_path / "test_runs_log.csv")
    log_run({
        "run_id": "integration_test",
        "features": "hog",
        "metric": "cosine",
        "pair_mode": "baseline",
        "n_thresh": 5,
        "val_auc": round(auc, 4),
        "test_f1": round(metrics["f1"], 4),
        "best_threshold": round(thresh, 4),
        "note": "synthetic integration test",
    }, log_path)

    runs = load_runs(log_path)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "integration_test"

    # 9. Synthetic data should be well above random (same-identity pairs look alike)
    assert metrics["accuracy"] > 0.5, (
        f"Expected accuracy > 0.5 on synthetic data, got {metrics['accuracy']:.3f}"
    )
