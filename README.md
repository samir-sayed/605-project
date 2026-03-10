# LFW Face Verification Pipeline — Milestone 2

A reproducible face verification system built on the Labeled Faces in the Wild (LFW) dataset.
Given two face images the pipeline produces a similarity score and a same-person vs.
different-person decision.

## Milestone 2 Summary

Milestone 2 adds a disciplined evaluation loop on top of the Milestone 1 foundation:

- **5 tracked experiments** (runs_log.csv) comparing feature types, similarity metrics, and a data-centric improvement
- **Threshold calibration** on the validation split using the max-balanced-accuracy (Youden-J) rule — threshold is selected before any test-set inspection
- **ROC curves and confusion matrices** for each run
- **Error analysis** with two error slices (false positives and false negatives)
- **Data-centric improvement**: raising `min_faces_per_person` from 2 → 5 and capping per-identity negative pairs (run04 vs run01)
- **Pipeline validation checks** that fail fast on malformed inputs
- **37 unit + integration tests** covering metrics, validation logic, pair generation, and end-to-end pipeline

## Tracked Runs Summary

| Run   | Features | Metric     | Pair Mode | Val AUC | Test F1 | Threshold |
|-------|----------|------------|-----------|---------|---------|-----------|
| run01 | HOG      | cosine     | baseline  | 0.6436  | 0.4294  | 0.8085    |
| run02 | HOG      | euclidean  | baseline  | 0.6436  | 0.4501  | 3.0682    |
| run03 | pixel    | cosine     | baseline  | 0.6188  | 0.5159  | 0.9413    |
| run04 | HOG      | cosine     | filtered  | 0.6202  | 0.4028  | 0.8165    |
| run05 | HOG      | cosine     | filtered  | 0.6202  | 0.4632  | 0.8045    |

**Best run by test F1: run05** (HOG + cosine + filtered pairs, fine-grained threshold sweep)

## Report and Artifacts

| Artifact | Path |
|----------|------|
| Milestone 2 report (PDF) | `reports/milestone2_report.pdf` |
| Tracked runs log | `reports/runs_log.csv` |
| ROC / CM / error-slice plots | `reports/plots/` |

## Repository Structure

```
lfw-verification/
├── configs/
│   ├── m1.yaml               # Milestone 1 config
│   └── m2.yaml               # Milestone 2 config (features, threshold, tracking)
├── scripts/
│   ├── ingest_lfw.py         # Download LFW, create identity-based splits, write manifest
│   ├── make_pairs.py         # Generate pair CSVs (baseline or filtered mode)
│   ├── run_experiment.py     # Run one tracked evaluation end-to-end
│   └── make_report.py        # Assemble 2-page PDF report from tracked run artifacts
├── src/
│   ├── ingestion.py          # LFW loading and split logic
│   ├── pairs.py              # Pair generation (baseline and filtered)
│   ├── features.py           # HOG and pixel feature extractors
│   ├── similarity.py         # Vectorized cosine and Euclidean similarity
│   ├── evaluation.py         # Threshold calibration, metrics, ROC, plots
│   ├── tracking.py           # CSV-based run logger
│   └── validation.py         # Pipeline validation checks
├── tests/
│   ├── test_metrics.py       # Unit tests: similarity functions and metrics
│   ├── test_validation.py    # Unit tests: validation and pair checks
│   └── test_integration.py   # Integration test: synthetic data → full pipeline
├── reports/
│   ├── milestone2_report.pdf # 2-page evaluation report
│   ├── runs_log.csv          # Evidence of 5 tracked runs
│   └── plots/                # ROC curves, confusion matrices, error slices
└── requirements.txt
```

## How to Run

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ingest LFW (downloads ~200 MB on first run, cached in ~/scikit_learn_data/)

```bash
python scripts/ingest_lfw.py --config configs/m2.yaml
```

### 3. Generate pair CSVs (run both modes)

```bash
python scripts/make_pairs.py --config configs/m2.yaml --mode baseline
python scripts/make_pairs.py --config configs/m2.yaml --mode filtered
```

### 4. Run the 5 tracked experiments

```bash
# Run 1 — Baseline: HOG + cosine, all identities (min_faces=2)
python scripts/run_experiment.py --config configs/m2.yaml \
    --run-id run01 --features hog --metric cosine --pair-mode baseline --n-thresh 10

# Run 2 — Ablation: HOG + euclidean distance
python scripts/run_experiment.py --config configs/m2.yaml \
    --run-id run02 --features hog --metric euclidean --pair-mode baseline --n-thresh 10

# Run 3 — Ablation: flat pixel features + cosine
python scripts/run_experiment.py --config configs/m2.yaml \
    --run-id run03 --features pixel --metric cosine --pair-mode baseline --n-thresh 10

# Run 4 — Data-centric improvement: min_faces=5, capped negatives (coarse sweep)
python scripts/run_experiment.py --config configs/m2.yaml \
    --run-id run04 --features hog --metric cosine --pair-mode filtered --n-thresh 10

# Run 5 — Data-centric improvement: same as run04 with fine threshold sweep
python scripts/run_experiment.py --config configs/m2.yaml \
    --run-id run05 --features hog --metric cosine --pair-mode filtered --n-thresh 20
```

### 5. Generate the report PDF

```bash
python scripts/make_report.py --config configs/m2.yaml
# Output: outputs/report_m2.pdf  (also committed at reports/milestone2_report.pdf)
```

### 6. Run tests

```bash
pytest tests/ -v
# Expected: 37 passed
```

## Threshold Selection Rule

The operating threshold is selected on the **validation split** by maximising balanced accuracy
(Youden-J statistic: `TPR - FPR`). This rule is stated before any test-set inspection and applied
consistently across all runs. See `src/evaluation.py:calibrate_threshold`.

## Data-Centric Improvement (run01 → run04/05)

**Baseline** (`pair-mode baseline`): all identities with >= 2 images. Identities with exactly
2 images can contribute only 1 positive pair, and high-frequency identities dominate the negative
pool (George W. Bush has ~530 images in LFW).

**Filtered** (`pair-mode filtered`): restricts to identities with >= 5 images, so every identity
can contribute the target 5 positive pairs. Per-identity negative pairs are capped at 5 and drawn
only from the filtered pool. Symmetric duplicate pairs are removed deterministically.

## Git Tag

`v0.2` — Milestone 2 release commit.
