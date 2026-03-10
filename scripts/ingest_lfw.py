"""
Ingest the LFW dataset via scikit-learn, create identity-based splits,
and write a real manifest to outputs/manifest.json.

Usage:
    python scripts/ingest_lfw.py --config configs/m2.yaml
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np

from src.ingestion import load_config, load_lfw_dataset, save_manifest
from src.validation import validate_config, check_no_split_leakage


def main(config_path: str) -> None:
    config = load_config(config_path)
    validate_config(config)

    print("Loading LFW dataset (downloading on first run)...")
    dataset = load_lfw_dataset(config)

    check_no_split_leakage(dataset["train"], dataset["val"], dataset["test"])

    for split in ("train", "val", "test"):
        d = dataset[split]
        n_ids = len(np.unique(d["targets"]))
        print(f"  {split}: {len(d['images'])} images, {n_ids} identities")

    save_manifest(dataset, config)
    print("Manifest written to outputs/manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
