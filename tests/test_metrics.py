"""
Unit tests for similarity functions and evaluation metrics.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.similarity import cosine_similarity_vectorized, euclidean_distance_vectorized
from src.evaluation import compute_metrics, calibrate_threshold, compute_roc


# ---------------------------------------------------------------------------
# Similarity function tests
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    result = cosine_similarity_vectorized(a, b)
    assert abs(result[0] - 1.0) < 1e-6


def test_cosine_opposite_vectors():
    a = np.array([[1.0, 0.0]])
    b = np.array([[-1.0, 0.0]])
    result = cosine_similarity_vectorized(a, b)
    assert abs(result[0] - (-1.0)) < 1e-6


def test_cosine_orthogonal_vectors():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    result = cosine_similarity_vectorized(a, b)
    assert abs(result[0] - 0.0) < 1e-6


def test_euclidean_zero_for_equal():
    a = np.array([[3.0, 4.0]])
    b = np.array([[3.0, 4.0]])
    result = euclidean_distance_vectorized(a, b)
    assert abs(result[0]) < 1e-6


def test_euclidean_known_distance():
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    result = euclidean_distance_vectorized(a, b)
    assert abs(result[0] - 5.0) < 1e-6


def test_cosine_vectorized_matches_loop():
    from src.similarity import cosine_similarity_loop
    rng = np.random.default_rng(0)
    a = rng.standard_normal((50, 32)).astype(np.float32)
    b = rng.standard_normal((50, 32)).astype(np.float32)
    vec = cosine_similarity_vectorized(a, b)
    loop = cosine_similarity_loop(a, b)
    assert np.max(np.abs(vec - loop)) < 1e-5


def test_euclidean_vectorized_matches_loop():
    from src.similarity import euclidean_distance_loop
    rng = np.random.default_rng(1)
    a = rng.standard_normal((50, 32)).astype(np.float32)
    b = rng.standard_normal((50, 32)).astype(np.float32)
    vec = euclidean_distance_vectorized(a, b)
    loop = euclidean_distance_loop(a, b)
    assert np.max(np.abs(vec - loop)) < 1e-5


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect_cosine():
    # Scores perfectly separated: same-pairs score 1.0, diff-pairs score 0.0
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    labels = np.array([1, 1, 0, 0])
    m = compute_metrics(scores, labels, threshold=0.5, metric="cosine")
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["tpr"] == 1.0
    assert m["fpr"] == 0.0


def test_compute_metrics_all_wrong_cosine():
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    labels = np.array([0, 0, 1, 1])  # inverted labels
    m = compute_metrics(scores, labels, threshold=0.5, metric="cosine")
    assert m["accuracy"] == 0.0
    assert m["tpr"] == 0.0


def test_compute_metrics_perfect_euclidean():
    # For euclidean: predict same when score <= threshold
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([1, 1, 0, 0])
    m = compute_metrics(scores, labels, threshold=0.5, metric="euclidean")
    assert m["accuracy"] == 1.0
    assert m["tpr"] == 1.0
    assert m["fpr"] == 0.0


def test_balanced_accuracy_range():
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, 100)
    labels = (scores > 0.5).astype(int)
    m = compute_metrics(scores, labels, threshold=0.5, metric="cosine")
    assert 0.0 <= m["balanced_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Threshold calibration tests
# ---------------------------------------------------------------------------

def test_calibrate_returns_float():
    rng = np.random.default_rng(7)
    scores = rng.uniform(0, 1, 100)
    labels = (scores > 0.5).astype(int)
    thresh, sweep = calibrate_threshold(scores, labels, n_points=5, metric="cosine")
    assert isinstance(thresh, float)
    assert len(sweep) == 5


def test_calibrate_threshold_selects_best_balanced_acc():
    # Constructed so that threshold=0.5 is clearly best
    scores = np.concatenate([np.ones(40) * 0.8, np.zeros(40) * 0.2])
    labels = np.array([1] * 40 + [0] * 40)
    thresh, _ = calibrate_threshold(scores, labels, n_points=10, metric="cosine")
    m = compute_metrics(scores, labels, thresh, metric="cosine")
    assert m["balanced_accuracy"] >= 0.9


# ---------------------------------------------------------------------------
# ROC tests
# ---------------------------------------------------------------------------

def test_roc_auc_above_chance():
    rng = np.random.default_rng(99)
    # scores positively correlated with labels → AUC should be > 0.5
    labels = rng.integers(0, 2, 100)
    scores = labels.astype(float) + rng.uniform(-0.3, 0.3, 100)
    _, _, _, auc = compute_roc(scores, labels, metric="cosine")
    assert auc > 0.5


def test_roc_arrays_same_length():
    rng = np.random.default_rng(5)
    scores = rng.uniform(0, 1, 60)
    labels = rng.integers(0, 2, 60)
    fpr, tpr, thresholds, auc = compute_roc(scores, labels, metric="cosine")
    assert len(fpr) == len(tpr) == len(thresholds)
