"""
Run a single tracked experiment end-to-end:
  load dataset → extract features → compute similarity scores →
  calibrate threshold on val → evaluate on test → log run → save plots

Usage examples:
    python scripts/run_experiment.py --config configs/m2.yaml \
        --run-id run01 --features hog --metric cosine --pair-mode baseline --n-thresh 10

    python scripts/run_experiment.py --config configs/m2.yaml \
        --run-id run04 --features hog --metric cosine --pair-mode filtered --n-thresh 10
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv

import numpy as np

from src.ingestion import load_config, load_lfw_dataset
from src.features import extract_features
from src.similarity import cosine_similarity_vectorized, euclidean_distance_vectorized
from src.evaluation import (
    calibrate_threshold, compute_metrics, compute_roc,
    plot_roc, plot_confusion_matrix, plot_error_slices,
)
from src.tracking import log_run
from src.validation import (
    validate_config, validate_scores, validate_metric, validate_feature_mode,
    check_no_split_leakage,
)


def load_pairs_csv(path: str):
    """Load a pairs CSV into a list of dicts (with int-cast indices)."""
    pairs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pairs.append({
                "left_idx": int(row["left_idx"]),
                "right_idx": int(row["right_idx"]),
                "label": int(row["label"]),
                "left_name": row["left_name"],
                "right_name": row["right_name"],
            })
    return pairs


def score_pairs(pairs, features: np.ndarray, metric: str) -> np.ndarray:
    """Compute pairwise similarity scores using pre-extracted features."""
    left = np.stack([features[p["left_idx"]] for p in pairs])
    right = np.stack([features[p["right_idx"]] for p in pairs])
    if metric == "cosine":
        return cosine_similarity_vectorized(left, right)
    else:
        return euclidean_distance_vectorized(left, right)


def main(config_path, run_id, feature_mode, metric, pair_mode, n_thresh, note):
    config = load_config(config_path)
    validate_config(config)
    validate_metric(metric)
    validate_feature_mode(feature_mode)

    plots_dir = config["tracking"]["plots_dir"]
    log_path = config["tracking"]["runs_log"]
    pairs_dir = "outputs/pairs"

    # Load dataset with the correct min_faces for the pair mode
    min_faces = (
        config["data"]["min_faces_filtered"]
        if pair_mode == "filtered"
        else config["data"]["min_faces_baseline"]
    )
    print(f"[{run_id}] Loading LFW (min_faces={min_faces}, pair_mode={pair_mode})...")
    dataset = load_lfw_dataset(config, min_faces_override=min_faces)
    check_no_split_leakage(dataset["train"], dataset["val"], dataset["test"])

    # Extract features for every image in val and test splits
    print(f"[{run_id}] Extracting {feature_mode} features for val split...")
    val_features = extract_features(dataset["val"]["images"], config, feature_mode)
    print(f"[{run_id}] Extracting {feature_mode} features for test split...")
    test_features = extract_features(dataset["test"]["images"], config, feature_mode)

    # Load pre-generated pairs
    val_csv = os.path.join(pairs_dir, f"val_{pair_mode}.csv")
    test_csv = os.path.join(pairs_dir, f"test_{pair_mode}.csv")
    if not os.path.isfile(val_csv):
        raise FileNotFoundError(
            f"{val_csv} not found. Run make_pairs.py --mode {pair_mode} first."
        )
    val_pairs = load_pairs_csv(val_csv)
    test_pairs = load_pairs_csv(test_csv)

    val_labels = np.array([p["label"] for p in val_pairs])
    test_labels = np.array([p["label"] for p in test_pairs])

    # Score pairs
    print(f"[{run_id}] Scoring {len(val_pairs)} val pairs with {metric}...")
    val_scores = score_pairs(val_pairs, val_features, metric)
    print(f"[{run_id}] Scoring {len(test_pairs)} test pairs with {metric}...")
    test_scores = score_pairs(test_pairs, test_features, metric)

    validate_scores(val_scores, val_labels)
    validate_scores(test_scores, test_labels)

    # Calibrate threshold on val (rule: max balanced accuracy — stated in configs/m2.yaml)
    print(f"[{run_id}] Calibrating threshold on val split ({n_thresh} points)...")
    best_thresh, sweep = calibrate_threshold(val_scores, val_labels, n_thresh, metric)
    val_metrics = compute_metrics(val_scores, val_labels, best_thresh, metric)
    val_fpr, val_tpr, _, val_auc = compute_roc(val_scores, val_labels, metric)

    print(f"[{run_id}] Selected threshold = {best_thresh:.4f} "
          f"(val bal_acc={val_metrics['balanced_accuracy']:.3f}, AUC={val_auc:.3f})")

    # Evaluate on test with the locked threshold
    test_metrics = compute_metrics(test_scores, test_labels, best_thresh, metric)
    test_fpr, test_tpr, _, test_auc = compute_roc(test_scores, test_labels, metric)

    print(f"[{run_id}] Test  accuracy={test_metrics['accuracy']:.3f}, "
          f"bal_acc={test_metrics['balanced_accuracy']:.3f}, "
          f"F1={test_metrics['f1']:.3f}, AUC={test_auc:.3f}")

    # Save plots
    roc_path = plot_roc(test_fpr, test_tpr, test_auc, run_id, plots_dir)
    cm_path = plot_confusion_matrix(test_scores, test_labels, best_thresh, metric,
                                    run_id, plots_dir)
    fp_path, fn_path = plot_error_slices(
        test_pairs, dataset["test"]["images"], test_scores, test_labels,
        best_thresh, metric, run_id, plots_dir,
    )
    print(f"[{run_id}] Plots saved: {roc_path}, {cm_path}, {fp_path}, {fn_path}")

    # Log the run
    import subprocess
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit_hash = "unknown"

    log_run({
        "run_id": run_id,
        "features": feature_mode,
        "metric": metric,
        "pair_mode": pair_mode,
        "n_thresh": n_thresh,
        "val_balanced_acc": round(val_metrics["balanced_accuracy"], 4),
        "val_auc": round(val_auc, 4),
        "val_f1": round(val_metrics["f1"], 4),
        "best_threshold": round(best_thresh, 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_balanced_acc": round(test_metrics["balanced_accuracy"], 4),
        "test_auc": round(test_auc, 4),
        "test_f1": round(test_metrics["f1"], 4),
        "note": note or f"commit={commit_hash}",
    }, log_path)
    print(f"[{run_id}] Logged to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--features", default="hog", choices=["hog", "pixel"])
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--pair-mode", default="baseline", choices=["baseline", "filtered"])
    parser.add_argument("--n-thresh", type=int, default=10)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    main(args.config, args.run_id, args.features, args.metric,
         args.pair_mode, args.n_thresh, args.note)
