"""
Pair generation for face verification: positive (same identity) and
negative (different identity) pairs, in two modes:
  baseline : all identities with >= min_faces_baseline images
  filtered : only identities with >= min_faces_filtered images,
             with a per-identity cap on negatives (data-centric improvement)
"""
import itertools
import random
from typing import List, Dict


def build_pairs(
    images,
    targets,
    names,
    config: dict,
    mode: str = "baseline",
    seed: int = 42,
) -> List[Dict]:
    """
    Build positive and negative pairs from a split's image/target arrays.

    Parameters
    ----------
    images  : np.ndarray (N, H, W)
    targets : np.ndarray (N,) int — identity index
    names   : np.ndarray (N,) str — identity name per image
    config  : loaded m2.yaml config dict
    mode    : "baseline" or "filtered"
    seed    : random seed for reproducibility

    Returns
    -------
    List of dicts with keys: left_idx, right_idx, label, left_name, right_name
    (indices are into the supplied arrays, not the global dataset)
    """
    rng = random.Random(seed)

    pos_per_id = config["pair_policy"]["positives_per_identity"]
    neg_per_id = config["pair_policy"]["negatives_per_identity"]
    max_neg = config["pair_policy"].get("max_pairs_per_identity", neg_per_id)

    min_faces_filtered = config["data"].get("min_faces_filtered", 5)

    # Build index map: identity -> list of local indices
    from collections import defaultdict
    id_to_indices = defaultdict(list)
    for local_idx, t in enumerate(targets):
        id_to_indices[int(t)].append(local_idx)

    # For filtered mode: restrict to identities with >= min_faces_filtered images
    if mode == "filtered":
        id_to_indices = {
            k: v for k, v in id_to_indices.items()
            if len(v) >= min_faces_filtered
        }

    identity_list = sorted(id_to_indices.keys())

    pairs = []

    # Positive pairs
    for identity in identity_list:
        indices = id_to_indices[identity]
        all_pos = list(itertools.combinations(indices, 2))
        rng.shuffle(all_pos)
        selected = all_pos[:pos_per_id]
        for l, r in selected:
            pairs.append({
                "left_idx": l,
                "right_idx": r,
                "label": 1,
                "left_name": names[l],
                "right_name": names[r],
            })

    # Negative pairs
    other_identities = identity_list.copy()
    cap = max_neg if mode == "filtered" else neg_per_id
    for identity in identity_list:
        indices = id_to_indices[identity]
        # Pool of indices from other identities
        other_pool = []
        for other_id in other_identities:
            if other_id != identity:
                other_pool.extend(id_to_indices[other_id])
        rng.shuffle(other_pool)
        neg_right = other_pool[:cap]
        left_img = rng.choice(indices)
        for r in neg_right:
            pairs.append({
                "left_idx": left_img,
                "right_idx": r,
                "label": 0,
                "left_name": names[left_img],
                "right_name": names[r],
            })

    # Deduplicate by frozenset of indices (also removes symmetric duplicates)
    seen = set()
    deduped = []
    for p in pairs:
        key = frozenset([p["left_idx"], p["right_idx"]])
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped


def validate_pairs(pairs: List[Dict]) -> None:
    """Raise ValueError if pairs list fails basic sanity checks."""
    if not pairs:
        raise ValueError("Pair list is empty")

    labels = [p["label"] for p in pairs]
    if not all(l in (0, 1) for l in labels):
        raise ValueError("All labels must be 0 or 1")

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    ratio = n_pos / len(labels)
    if ratio < 0.1 or ratio > 0.9:
        raise ValueError(
            f"Pair set is severely imbalanced: {n_pos} pos / {n_neg} neg "
            f"(ratio={ratio:.2f}). Reconsider pair policy."
        )

    # Check no duplicate (unordered) pairs
    seen = set()
    for p in pairs:
        key = frozenset([p["left_idx"], p["right_idx"]])
        if key in seen:
            raise ValueError(
                f"Duplicate pair found: indices {p['left_idx']}, {p['right_idx']}"
            )
        seen.add(key)

    # Positive pairs must have same name
    for p in pairs:
        if p["label"] == 1 and p["left_name"] != p["right_name"]:
            raise ValueError(
                f"Positive pair has mismatched names: "
                f"{p['left_name']} vs {p['right_name']}"
            )
    # Negative pairs must have different names
    for p in pairs:
        if p["label"] == 0 and p["left_name"] == p["right_name"]:
            raise ValueError(
                f"Negative pair has identical names: {p['left_name']}"
            )
