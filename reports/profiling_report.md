# Profiling Report — LFW Face Verification Pipeline

**Version:** v1.0-final
**Date:** 2026-04-22
**Profile type:** CPU Baseline

---

## 1. Measurement Environment

| Property | Value |
|----------|-------|
| CPU | Apple M2 |
| Cores | 8 |
| Architecture | arm64 |
| OS | macOS 15.0 (Darwin) |
| Python | 3.12.2 |
| NumPy | 1.26.4 |
| scikit-image | >= 0.23.0 |

**Methodology:**
- Each measurement uses 2 warmup runs followed by 5 timed runs.
- Timing is done with `time.perf_counter()` for sub-millisecond precision.
- The test split (690 images, 62x47 px, filtered mode with min_faces=5) is used for all measurements.
- No GPU was used. All results are CPU-only.

---

## 2. Per-Stage Latency Breakdown (32 pairs)

| Stage | Mean (ms) | Std (ms) | % of Total |
|-------|-----------|----------|------------|
| Preprocessing (image indexing) | 0.115 | 0.060 | 0.3% |
| Feature extraction (HOG x2) | 37.289 | 1.964 | 99.4% |
| Similarity scoring (cosine) | 0.106 | 0.011 | 0.3% |
| **End-to-end total** | **37.509** | — | **100%** |

**Key finding:** Feature extraction (HOG computation) overwhelmingly dominates the pipeline at 99.4% of total latency. Preprocessing (array indexing) and similarity scoring (vectorized cosine) are both sub-millisecond and negligible.

This is expected because HOG computation involves per-image gradient computation, histogram binning, and block normalization — all CPU-bound operations applied independently to each image. In contrast, cosine similarity is a single vectorized dot-product and norm computation across all pairs simultaneously.

---

## 3. Batch-Size Sensitivity

| Batch Size | Total Time (ms) | Per-Pair (ms) | Throughput (pairs/s) |
|------------|-----------------|---------------|----------------------|
| 1 | 1.359 | 1.359 | 735.9 |
| 2 | 2.428 | 1.214 | 823.6 |
| 4 | 4.745 | 1.186 | 843.0 |
| 8 | 9.639 | 1.205 | 830.0 |
| 16 | 18.584 | 1.161 | 861.0 |
| 32 | 34.354 | 1.074 | 931.5 |
| 64 | 67.922 | 1.061 | 942.3 |

**Observations:**

1. **Total latency scales linearly** with batch size. This is expected because HOG extraction is applied per-image in a Python loop — there is no batch-level parallelism in the current implementation.

2. **Per-pair latency decreases modestly** from 1.36 ms (batch=1) to 1.06 ms (batch=64), a ~22% reduction. The improvement comes from amortizing the fixed overhead of array allocation and the vectorized cosine scoring step across more pairs.

3. **Throughput improves** from ~736 pairs/s (batch=1) to ~942 pairs/s (batch=64). The gains flatten beyond batch size 32, suggesting diminishing returns from overhead amortization.

4. **Bottleneck:** Since HOG extraction runs per-image in a sequential loop, batch-size improvements are limited. A vectorized or compiled feature extractor (e.g., using OpenCV's HOG or a CNN-based embedding) would show stronger batch scaling.

---

## 4. Interpretation and Tradeoffs

- **For single-pair verification** (e.g., CLI usage): Latency is ~1.4 ms per pair, which is well within interactive response requirements.
- **For batch evaluation** (e.g., running experiments): Processing 1,000 pairs takes approximately 1.1 seconds, which is fast enough for the scale of this project.
- **Scaling limitation:** The sequential HOG loop is the primary bottleneck. If throughput were critical, the main optimization targets would be:
  1. Using a compiled HOG implementation (e.g., OpenCV).
  2. Replacing HOG with a pre-trained embedding model that supports batch GPU inference.
  3. Parallelizing feature extraction across CPU cores.

---

## 5. How to Reproduce

```bash
cd lfw-verification
source .venv/bin/activate
python scripts/profile_system.py --config configs/m4.yaml
```

Results are saved to:
- `outputs/profiling/profiling_results.json` — Full JSON results
- `outputs/profiling/batch_sensitivity.csv` — Batch-size sensitivity table
