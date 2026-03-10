"""
LFW dataset ingestion and identity-based train/val/test splitting.
"""
import json
import os

import numpy as np
import yaml
from sklearn.datasets import fetch_lfw_people


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_lfw_dataset(config: dict, min_faces_override: int = None) -> dict:
    """
    Download/load LFW via scikit-learn, split by identity (no leakage).

    Returns a dict with keys: train, val, test, each containing:
        images   : np.ndarray (N, H, W) float32 in [0, 1]
        targets  : np.ndarray (N,) int  — identity index
        names    : np.ndarray (N,) str  — identity name per image
    Plus top-level keys: target_names, all_images, all_targets, image_shape.
    """
    seed = config["seed"]
    resize = config["data"]["resize"]
    min_faces = min_faces_override if min_faces_override is not None \
        else config["data"]["min_faces_baseline"]

    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=resize,
        color=False,
    )

    all_images = lfw.images          # (N, H, W) float32
    all_targets = lfw.target         # (N,) int
    target_names = lfw.target_names  # (n_classes,) str

    # Identity-based split — shuffle unique identities with fixed seed
    unique_ids = np.unique(all_targets)
    rng = np.random.default_rng(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_ratio = config["split_policy"]["train_ratio"]
    val_ratio = config["split_policy"]["val_ratio"]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train: n_train + n_val])
    test_ids = set(shuffled[n_train + n_val:])

    # Validate no leakage
    assert len(train_ids & val_ids) == 0, "Train/val identity overlap"
    assert len(train_ids & test_ids) == 0, "Train/test identity overlap"
    assert len(val_ids & test_ids) == 0, "Val/test identity overlap"

    def _subset(id_set):
        mask = np.array([t in id_set for t in all_targets])
        imgs = all_images[mask]
        tgts = all_targets[mask]
        nms = np.array([target_names[t] for t in tgts])
        return {"images": imgs, "targets": tgts, "names": nms}

    return {
        "train": _subset(train_ids),
        "val": _subset(val_ids),
        "test": _subset(test_ids),
        "all_images": all_images,
        "all_targets": all_targets,
        "target_names": target_names,
        "image_shape": all_images.shape[1:],
    }


def save_manifest(dataset: dict, config: dict, out_path: str = "outputs/manifest.json") -> None:
    """Write a real manifest with actual split counts."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _counts(split_data):
        return {
            "images": int(len(split_data["images"])),
            "identities": int(len(np.unique(split_data["targets"]))),
        }

    manifest = {
        "seed": config["seed"],
        "data_source": "sklearn_lfw (resize={}, min_faces={})".format(
            config["data"]["resize"],
            config["data"]["min_faces_baseline"],
        ),
        "split_policy": "identity-based {:.0f}/{:.0f}/{:.0f}".format(
            config["split_policy"]["train_ratio"] * 100,
            config["split_policy"]["val_ratio"] * 100,
            config["split_policy"]["test_ratio"] * 100,
        ),
        "image_shape": list(dataset["image_shape"]),
        "counts": {
            "train": _counts(dataset["train"]),
            "val": _counts(dataset["val"]),
            "test": _counts(dataset["test"]),
        },
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
