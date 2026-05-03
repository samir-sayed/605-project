"""
Milestone 4 — Hardware-aware profiling script.

Measures per-stage latency (preprocessing, feature extraction, scoring)
and batch-size sensitivity on CPU.

Usage:
    python scripts/profile_system.py --config configs/m4.yaml
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import json
import platform
import time

import numpy as np

from src.ingestion import load_config, load_lfw_dataset
from src.features import extract_features
from src.similarity import cosine_similarity_vectorized


def get_cpu_info():
    """Gather CPU and platform information for the profiling report."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    # Try to get more detailed CPU info on macOS
    if platform.system() == "Darwin":
        try:
            import subprocess
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["cpu_brand"] = brand
            cores = subprocess.check_output(
                ["sysctl", "-n", "hw.ncpu"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["cpu_cores"] = int(cores)
        except Exception:
            pass
    return info


def time_preprocessing(images, n_pairs, warmup, runs):
    """Time the preprocessing stage: loading image pairs into arrays."""
    rng = np.random.default_rng(42)
    indices_a = rng.integers(0, len(images), size=n_pairs)
    indices_b = rng.integers(0, len(images), size=n_pairs)

    # Warmup
    for _ in range(warmup):
        left = images[indices_a]
        right = images[indices_b]

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        left = images[indices_a]
        right = images[indices_b]
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return timings, left, right


def time_feature_extraction(images, config, mode, warmup, runs):
    """Time the feature extraction stage."""
    # Warmup
    for _ in range(warmup):
        extract_features(images, config, mode)

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        extract_features(images, config, mode)
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    features = extract_features(images, config, mode)
    return timings, features


def time_scoring(feat_a, feat_b, warmup, runs):
    """Time the similarity scoring stage."""
    # Warmup
    for _ in range(warmup):
        cosine_similarity_vectorized(feat_a, feat_b)

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        cosine_similarity_vectorized(feat_a, feat_b)
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    return timings


def profile_batch_size(images, config, batch_sizes, warmup, runs):
    """Measure end-to-end latency across different batch sizes."""
    rng = np.random.default_rng(42)
    mode = config["features"]["mode"]
    results = []

    for bs in batch_sizes:
        indices_a = rng.integers(0, len(images), size=bs)
        indices_b = rng.integers(0, len(images), size=bs)

        # Warmup
        for _ in range(warmup):
            left = images[indices_a]
            right = images[indices_b]
            feat_l = extract_features(left, config, mode)
            feat_r = extract_features(right, config, mode)
            cosine_similarity_vectorized(feat_l, feat_r)

        timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            left = images[indices_a]
            right = images[indices_b]
            feat_l = extract_features(left, config, mode)
            feat_r = extract_features(right, config, mode)
            cosine_similarity_vectorized(feat_l, feat_r)
            t1 = time.perf_counter()
            timings.append(t1 - t0)

        mean_ms = np.mean(timings) * 1000
        std_ms = np.std(timings) * 1000
        per_pair_ms = mean_ms / bs
        results.append({
            "batch_size": bs,
            "mean_total_ms": round(mean_ms, 3),
            "std_total_ms": round(std_ms, 3),
            "per_pair_ms": round(per_pair_ms, 3),
            "throughput_pairs_per_sec": round(bs / (np.mean(timings)), 1),
        })
        print(f"  batch_size={bs:>4d}  total={mean_ms:>8.2f} ms  "
              f"per_pair={per_pair_ms:>7.3f} ms  "
              f"throughput={results[-1]['throughput_pairs_per_sec']:>8.1f} pairs/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile the face verification pipeline")
    parser.add_argument("--config", default="configs/m4.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    profiling_cfg = config.get("profiling", {})
    batch_sizes = profiling_cfg.get("batch_sizes", [1, 2, 4, 8, 16, 32, 64])
    warmup = profiling_cfg.get("warmup_runs", 2)
    runs = profiling_cfg.get("timed_runs", 5)
    output_dir = profiling_cfg.get("output_dir", "outputs/profiling")
    os.makedirs(output_dir, exist_ok=True)

    mode = config["features"]["mode"]

    # --- Gather system info ---
    cpu_info = get_cpu_info()
    print("=" * 60)
    print("HARDWARE-AWARE PROFILING — CPU BASELINE")
    print("=" * 60)
    for k, v in cpu_info.items():
        print(f"  {k}: {v}")
    print()

    # --- Load dataset ---
    print("[1/4] Loading LFW dataset (filtered, min_faces=5)...")
    min_faces = config["data"]["min_faces_filtered"]
    dataset = load_lfw_dataset(config, min_faces_override=min_faces)
    test_images = dataset["test"]["images"]
    print(f"  Test split: {len(test_images)} images, shape={test_images.shape[1:]}")
    print()

    # --- Per-stage latency (fixed batch of 32 pairs) ---
    n_pairs = 32
    print(f"[2/4] Per-stage latency breakdown (n_pairs={n_pairs}, {runs} runs, {warmup} warmup)...")

    # Stage 1: Preprocessing
    preprocess_times, left_imgs, right_imgs = time_preprocessing(
        test_images, n_pairs, warmup, runs
    )
    preprocess_mean = np.mean(preprocess_times) * 1000
    preprocess_std = np.std(preprocess_times) * 1000

    # Stage 2: Feature extraction (both sides)
    feat_times_l, feat_l = time_feature_extraction(left_imgs, config, mode, warmup, runs)
    feat_times_r, feat_r = time_feature_extraction(right_imgs, config, mode, warmup, runs)
    feat_mean = (np.mean(feat_times_l) + np.mean(feat_times_r)) * 1000
    feat_std = np.sqrt(np.std(feat_times_l)**2 + np.std(feat_times_r)**2) * 1000

    # Stage 3: Scoring
    score_times = time_scoring(feat_l, feat_r, warmup, runs)
    score_mean = np.mean(score_times) * 1000
    score_std = np.std(score_times) * 1000

    total_mean = preprocess_mean + feat_mean + score_mean

    print(f"  Preprocessing (image indexing):  {preprocess_mean:>8.3f} ms  +/- {preprocess_std:.3f} ms")
    print(f"  Feature extraction (HOG x2):     {feat_mean:>8.3f} ms  +/- {feat_std:.3f} ms")
    print(f"  Similarity scoring (cosine):     {score_mean:>8.3f} ms  +/- {score_std:.3f} ms")
    print(f"  End-to-end total:                {total_mean:>8.3f} ms")
    print(f"  Feature extraction dominance:    {feat_mean / total_mean * 100:.1f}%")
    print()

    latency_breakdown = {
        "n_pairs": n_pairs,
        "warmup_runs": warmup,
        "timed_runs": runs,
        "feature_mode": mode,
        "metric": "cosine",
        "preprocessing_ms": {"mean": round(preprocess_mean, 3), "std": round(preprocess_std, 3)},
        "feature_extraction_ms": {"mean": round(feat_mean, 3), "std": round(feat_std, 3)},
        "scoring_ms": {"mean": round(score_mean, 3), "std": round(score_std, 3)},
        "end_to_end_ms": round(total_mean, 3),
        "feature_extraction_pct": round(feat_mean / total_mean * 100, 1),
    }

    # --- Batch-size sensitivity ---
    print(f"[3/4] Batch-size sensitivity ({runs} runs, {warmup} warmup)...")
    batch_results = profile_batch_size(test_images, config, batch_sizes, warmup, runs)
    print()

    # --- Save results ---
    print("[4/4] Saving profiling results...")
    results = {
        "cpu_info": cpu_info,
        "latency_breakdown": latency_breakdown,
        "batch_size_sensitivity": batch_results,
    }
    json_path = os.path.join(output_dir, "profiling_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON: {json_path}")

    csv_path = os.path.join(output_dir, "batch_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=batch_results[0].keys())
        writer.writeheader()
        writer.writerows(batch_results)
    print(f"  CSV:  {csv_path}")

    print()
    print("Profiling complete.")


if __name__ == "__main__":
    main()
