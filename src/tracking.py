"""
Lightweight CSV-based run tracker.
Each call to log_run() appends one row to the runs log file.
"""
import csv
import os
from datetime import datetime, timezone
from typing import Dict


FIELDS = [
    "run_id", "timestamp", "features", "metric", "pair_mode",
    "n_thresh", "val_balanced_acc", "val_auc", "val_f1",
    "best_threshold", "test_accuracy", "test_balanced_acc",
    "test_auc", "test_f1", "note",
]


def log_run(record: Dict, log_path: str) -> None:
    """Append a run record to the CSV log, writing header if needed."""
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        row = {k: record.get(k, "") for k in FIELDS}
        if not row.get("timestamp"):
            row["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        writer.writerow(row)


def load_runs(log_path: str) -> list:
    """Load all tracked runs from the CSV log as a list of dicts."""
    if not os.path.isfile(log_path):
        return []
    with open(log_path, newline="") as f:
        return list(csv.DictReader(f))
