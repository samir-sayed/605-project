# System Card — LFW Face Verification Pipeline

**Version:** v1.0-final
**Date:** 2026-04-22
**Final Run:** run05 (HOG + cosine + filtered pairs)

---

## 1. System Overview

This system is a face verification pipeline that determines whether two face images depict the same person or different persons. It accepts two grayscale face images as input and produces a cosine similarity score and a binary same/different decision based on a calibrated threshold.

**Pipeline stages:**

1. **Preprocessing** — Input images are converted to grayscale and resized to 62x47 pixels to match the LFW training format.
2. **Feature extraction** — Histogram of Oriented Gradients (HOG) descriptors are computed (8 orientations, 8x8 pixels per cell, 2x2 cells per block), producing a 768-dimensional feature vector per image.
3. **Similarity scoring** — Cosine similarity is computed between the two HOG feature vectors.
4. **Decision** — If the cosine similarity score meets or exceeds the operating threshold (0.8045), the pair is classified as "same person"; otherwise, "different person."

The system is implemented in Python using NumPy, scikit-image, and scikit-learn. A Dockerized CLI is provided for deployment.

---

## 2. Intended Use

**Supported use cases:**

- Academic evaluation of face verification approaches on the LFW benchmark.
- Educational demonstration of a complete ML pipeline (ingestion, feature extraction, scoring, threshold calibration, evaluation).
- Baseline comparison for more advanced face verification systems.

**Explicitly out-of-scope uses:**

- Production identity verification or access control.
- Law enforcement or surveillance applications.
- Any safety-critical decision-making involving personal identity.
- Verification on non-face images, occluded faces, or images significantly different from the LFW distribution.

This system is an academic project and is not suitable for deployment in real-world identity verification scenarios.

---

## 3. Data Summary

**Dataset:** Labeled Faces in the Wild (LFW), accessed via `sklearn.datasets.fetch_lfw_people`.

- **Total images:** 9,164 (with min_faces_per_person=5 for the final system's filtered mode)
- **Image dimensions:** 62x47 pixels, grayscale
- **Identity-based split:** 70% train / 15% val / 15% test (no identity leakage across splits)
- **Pair generation (filtered mode):** Restricted to identities with >= 5 images, 5 positive pairs and 5 negative pairs per identity, per-identity negative pairs capped at 5.

**Important data limitations:**

- LFW is heavily skewed toward public figures, predominantly male, and predominantly light-skinned. It does not represent the global population.
- Images are web-scraped celebrity photos with generally cooperative pose, adequate lighting, and moderate quality. This is not representative of real-world verification conditions.
- The filtered mode (min_faces >= 5) further reduces the dataset to a subset of well-represented identities, which may inflate perceived performance compared to a broader population.

---

## 4. Operating Threshold and Key Metrics

**Threshold selection rule:** Maximum balanced accuracy (Youden-J statistic: TPR - FPR) on the validation split. The threshold is selected before any test-set inspection and applied consistently across all runs.

**Final system (run05):**

| Metric | Value |
|--------|-------|
| Operating threshold | 0.8045 |
| Validation AUC | 0.6202 |
| Test F1 | 0.4632 |
| Feature type | HOG (768-d) |
| Similarity metric | Cosine |
| Pair mode | Filtered (min_faces=5) |
| Threshold sweep points | 20 |

**Tracked runs comparison:**

| Run | Features | Metric | Pair Mode | Val AUC | Test F1 | Threshold |
|-----|----------|--------|-----------|---------|---------|-----------|
| run01 | HOG | cosine | baseline | 0.6436 | 0.4294 | 0.8085 |
| run02 | HOG | euclidean | baseline | 0.6436 | 0.4501 | 3.0682 |
| run03 | pixel | cosine | baseline | 0.6188 | 0.5159 | 0.9413 |
| run04 | HOG | cosine | filtered | 0.6202 | 0.4028 | 0.8165 |
| run05 | HOG | cosine | filtered | 0.6202 | 0.4632 | 0.8045 |

---

## 5. Failure Modes and Limitations

**Known failure modes:**

1. **Pose variation:** HOG features are not pose-invariant. Large differences in head orientation between paired images lead to dissimilar feature vectors even for the same person, producing false negatives.

2. **Lighting and contrast differences:** Images taken under very different lighting conditions produce different gradient distributions, causing same-person pairs to score below the threshold.

3. **Visually similar different-identity pairs:** Individuals with similar facial structure, hair, and skin tone may produce high cosine similarity scores, causing false positives. Error-slice analysis (fp_slice plots) confirms this pattern.

4. **Low image quality:** Blurry, low-resolution, or heavily compressed images lose the gradient detail that HOG depends on, degrading feature quality.

5. **Occlusion:** Glasses, hats, masks, or hand occlusion changes the gradient pattern and can shift scores unpredictably.

6. **Out-of-distribution inputs:** The system assumes cropped, roughly aligned face images matching the LFW preprocessing pipeline. Arbitrary images, full-body shots, or non-face images will produce meaningless results.

**Performance limitations:**

- Test F1 of 0.4632 indicates the system frequently misclassifies pairs. This is insufficient for any real-world verification task.
- The HOG representation captures only local gradient statistics and lacks the discriminative power of learned embeddings (e.g., deep neural networks).
- The threshold was calibrated on a relatively small validation set (380 filtered pairs), which limits the precision of the calibrated threshold.

---

## 6. Fairness-Related Risks

**This system has not been evaluated for demographic fairness.** The LFW dataset does not include reliable demographic metadata, so subgroup performance breakdowns are not available.

However, the following fairness-related risks are known to exist in face verification systems generally, and are likely present in this system:

1. **Population bias in training data:** LFW is heavily skewed toward lighter-skinned public figures. The system has seen far more images of some demographic groups than others, which may cause uneven performance across populations.

2. **Uneven performance across skin tones:** HOG features depend on gradient contrast, which varies with skin tone and lighting interaction. Darker skin tones under low-contrast lighting may produce weaker gradient signals, potentially increasing false negative rates for those groups.

3. **Gender imbalance:** LFW contains significantly more male faces than female faces. The system may perform differently across genders due to this imbalance.

4. **Age representation:** LFW primarily contains adult faces. Performance on children, elderly individuals, or across large age gaps is unknown and likely degraded.

5. **Misuse risk:** Even with its limited accuracy, the system could be misused for unauthorized surveillance or discriminatory screening. The low accuracy makes such misuse both unethical and unreliable.

**Mitigation:** This system should not be used for any decision-making that affects individuals. It is an academic exercise only.

---

## 7. Operational Constraints

| Constraint | Details |
|------------|---------|
| Input format | Grayscale face images, any common format (JPEG, PNG). Automatically resized to 62x47 px. |
| Hardware | CPU-only. Tested on Apple M2 (8 cores). No GPU required. |
| Latency | ~1.1 ms per pair end-to-end on CPU (batch=64). Feature extraction dominates at 99.4% of runtime. |
| Throughput | ~940 pairs/second at batch size 64 on Apple M2. |
| Dependencies | Python 3.12, NumPy, scikit-learn, scikit-image, matplotlib, Pillow, PyYAML. See `requirements.txt`. |
| Docker | Multi-stage Dockerfile provided. Image size ~500 MB. |
| Storage | LFW dataset (~200 MB) downloaded on first run to ~/scikit_learn_data/ (cached). |

---

## 8. Reproducibility

- **Final Git tag:** `v1.0-final`
- **Config:** `configs/m4.yaml`
- **Reproducibility checklist:** `reports/reproducibility_checklist.md`
- **README:** Contains full environment setup, CLI, Docker, and test commands.
- **Profiling report:** `reports/profiling_report.md`
- **Seed:** 42 (fixed for all random operations including dataset splitting and pair generation).

All evaluation results can be reproduced from a fresh clone by following the reproducibility checklist.
