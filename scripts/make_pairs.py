"""
Generate face pair CSVs for train/val/test splits.

Usage:
    python scripts/make_pairs.py --config configs/m2.yaml --mode baseline
    python scripts/make_pairs.py --config configs/m2.yaml --mode filtered
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv

from src.ingestion import load_config, load_lfw_dataset
from src.pairs import build_pairs, validate_pairs
from src.validation import validate_config


def save_pairs_csv(pairs, split_name: str, mode: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split_name}_{mode}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["left_idx", "right_idx", "label", "left_name", "right_name", "split"]
        )
        writer.writeheader()
        for p in pairs:
            writer.writerow({**p, "split": split_name})
    return path


def main(config_path: str, mode: str) -> None:
    config = load_config(config_path)
    validate_config(config)

    min_faces = (
        config["data"]["min_faces_filtered"]
        if mode == "filtered"
        else config["data"]["min_faces_baseline"]
    )

    print(f"Loading LFW (min_faces={min_faces}, mode={mode})...")
    dataset = load_lfw_dataset(config, min_faces_override=min_faces)

    out_dir = "outputs/pairs"
    seed = config["seed"]

    for split in ("train", "val", "test"):
        d = dataset[split]
        pairs = build_pairs(
            images=d["images"],
            targets=d["targets"],
            names=d["names"],
            config=config,
            mode=mode,
            seed=seed,
        )
        validate_pairs(pairs)
        n_pos = sum(p["label"] for p in pairs)
        n_neg = len(pairs) - n_pos
        path = save_pairs_csv(pairs, split, mode, out_dir)
        print(f"  {split}: {len(pairs)} pairs ({n_pos} pos, {n_neg} neg) → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="baseline", choices=["baseline", "filtered"])
    args = parser.parse_args()
    main(args.config, args.mode)
