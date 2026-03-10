"""
Evaluation utilities: threshold calibration, metrics, ROC, confusion matrix,
and error-slice visualisation.
"""
import os
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _predictions(scores: np.ndarray, threshold: float, metric: str) -> np.ndarray:
    """Apply threshold to produce binary predictions.
    For cosine similarity  : predict 1 when score >= threshold
    For euclidean distance : predict 1 when score <= threshold  (inverted)
    """
    if metric == "cosine":
        return (scores >= threshold).astype(int)
    elif metric == "euclidean":
        return (scores <= threshold).astype(int)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")


def compute_metrics(scores: np.ndarray, labels: np.ndarray,
                    threshold: float, metric: str = "cosine") -> Dict:
    """Return a dict of classification metrics at a given threshold."""
    preds = _predictions(scores, threshold, metric)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    accuracy = (tp + tn) / len(labels)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * tpr / (precision + tpr)
          if (precision + tpr) > 0 else 0.0)
    balanced_acc = (tpr + (1 - fpr)) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "precision": precision,
        "recall": tpr,
        "tpr": tpr,
        "fpr": fpr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(scores: np.ndarray, labels: np.ndarray,
                         n_points: int, metric: str = "cosine") -> Tuple[float, List[Dict]]:
    """
    Sweep n_points evenly-spaced thresholds and return:
      - best_threshold : float chosen by maximising balanced accuracy (Youden-J)
      - sweep_records  : list of dicts with threshold + metrics at each step

    The selection rule is stated in configs/m2.yaml (max_balanced_accuracy).
    """
    lo, hi = float(scores.min()), float(scores.max())
    thresholds = np.linspace(lo, hi, n_points)

    sweep_records = []
    best_thresh = float(thresholds[len(thresholds) // 2])
    best_ba = -1.0

    for t in thresholds:
        m = compute_metrics(scores, labels, float(t), metric)
        m["threshold"] = float(t)
        sweep_records.append(m)
        if m["balanced_accuracy"] > best_ba:
            best_ba = m["balanced_accuracy"]
            best_thresh = float(t)

    return best_thresh, sweep_records


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

def compute_roc(scores: np.ndarray, labels: np.ndarray,
                metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (fpr, tpr, thresholds, auc).
    For euclidean distance the scores are negated so that sklearn roc_curve
    always treats higher values as more-similar (positive class).
    """
    signed = scores if metric == "cosine" else -scores
    fpr, tpr, thresholds = roc_curve(labels, signed)
    auc = float(roc_auc_score(labels, signed))
    return fpr, tpr, thresholds, auc


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float,
             run_id: str, plots_dir: str) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {run_id}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    path = os.path.join(plots_dir, f"roc_{run_id}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(scores: np.ndarray, labels: np.ndarray,
                           threshold: float, metric: str,
                           run_id: str, plots_dir: str) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    m = compute_metrics(scores, labels, threshold, metric)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im)
    classes = ["Different (0)", "Same (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix @ t={threshold:.3f}\n{run_id}")
    thresh_val = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh_val else "black")
    path = os.path.join(plots_dir, f"cm_{run_id}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_error_slices(pairs: List[Dict], images: np.ndarray,
                      scores: np.ndarray, labels: np.ndarray,
                      threshold: float, metric: str,
                      run_id: str, plots_dir: str,
                      n_examples: int = 4) -> Tuple[str, str]:
    """
    Save image grids for two error slices:
      fp_slice : False Positives (predicted same, actually different)
      fn_slice : False Negatives (predicted different, actually same)
    """
    os.makedirs(plots_dir, exist_ok=True)
    preds = _predictions(scores, threshold, metric)

    fp_indices = [i for i, (p, l) in enumerate(zip(preds, labels))
                  if p == 1 and l == 0]
    fn_indices = [i for i, (p, l) in enumerate(zip(preds, labels))
                  if p == 0 and l == 1]

    def _save_slice(indices, title_prefix, fname_prefix):
        chosen = indices[:n_examples]
        if not chosen:
            return None
        fig, axes = plt.subplots(len(chosen), 2,
                                 figsize=(4, 2 * len(chosen)))
        if len(chosen) == 1:
            axes = [axes]
        for row, idx in enumerate(chosen):
            p = pairs[idx]
            l_img = images[p["left_idx"]]
            r_img = images[p["right_idx"]]
            axes[row][0].imshow(l_img, cmap="gray")
            axes[row][0].axis("off")
            axes[row][0].set_title(p["left_name"][:20], fontsize=7)
            axes[row][1].imshow(r_img, cmap="gray")
            axes[row][1].axis("off")
            axes[row][1].set_title(
                f"{p['right_name'][:20]}\nscore={scores[idx]:.3f}", fontsize=7)
        total = len(indices)
        fig.suptitle(f"{title_prefix} ({total} total)\n{run_id}", fontsize=9)
        path = os.path.join(plots_dir, f"{fname_prefix}_{run_id}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    fp_path = _save_slice(fp_indices, "False Positives (predicted Same, actually Different)",
                          "fp_slice")
    fn_path = _save_slice(fn_indices, "False Negatives (predicted Different, actually Same)",
                          "fn_slice")
    return fp_path, fn_path
