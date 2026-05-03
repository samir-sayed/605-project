# LFW Face Verification Pipeline — Final Release

A reproducible face verification system built on the Labeled Faces in the Wild (LFW) dataset.
Given two face images the pipeline produces a similarity score and a same-person vs.
different-person decision.

## Final System Summary

The final system uses **HOG features** (768-d) with **cosine similarity** scoring and a threshold of **0.8045**, calibrated on the validation split using the max-balanced-accuracy (Youden-J) rule. The system operates on grayscale 62x47 images and is packaged as a Dockerized CLI.

| Property | Value |
|----------|-------|
| Final run | run05 |
| Features | HOG (8 orientations, 8x8 px/cell, 2x2 cells/block) |
| Metric | Cosine similarity |
| Threshold | 0.8045 |
| Val AUC | 0.6202 |
| Test F1 | 0.4632 |
| Pair mode | Filtered (min_faces >= 5) |

## Milestone Summary

| Milestone | Contribution |
|-----------|-------------|
| M1 | Deterministic ingestion, saved pairs, reproducible structure, vectorized scoring |
| M2 | Threshold calibration, 5 tracked runs, error analysis, data-centric iteration, 37 tests |
| M4 | System Card, profiling report, reproducibility checklist, final release alignment |

## Key Artifacts

| Artifact | Path |
|----------|------|
| System Card | [`reports/system_card.md`](reports/system_card.md) |
| Profiling Report | [`reports/profiling_report.md`](reports/profiling_report.md) |
| Reproducibility Checklist | [`reports/reproducibility_checklist.md`](reports/reproducibility_checklist.md) |
| Milestone 2 Report (PDF) | [`reports/milestone2_report.pdf`](reports/milestone2_report.pdf) |
| Tracked Runs Log | [`reports/runs_log.csv`](reports/runs_log.csv) |
| ROC / CM / Error Plots | [`reports/plots/`](reports/plots/) |
| Final Config | [`configs/m4.yaml`](configs/m4.yaml) |

## Repository Structure

```
lfw-verification/
├── configs/
│   ├── m1.yaml               # Milestone 1 config
│   ├── m2.yaml               # Milestone 2 config
│   └── m4.yaml               # Milestone 4 final config
├── scripts/
│   ├── ingest_lfw.py         # Download LFW, create identity-based splits
│   ├── make_pairs.py         # Generate pair CSVs (baseline or filtered)
│   ├── run_experiment.py     # Run one tracked evaluation end-to-end
│   ├── make_report.py        # Assemble PDF report from tracked runs
│   ├── bench_similarity.py   # Benchmark vectorized vs loop similarity
│   ├── profile_system.py     # Hardware-aware profiling (M4)
│   └── verify_pair.py        # CLI face verification entrypoint (M4)
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
│   ├── system_card.md        # Final System Card (M4)
│   ├── profiling_report.md   # Profiling report with latency breakdown (M4)
│   ├── reproducibility_checklist.md  # Reproducibility checklist (M4)
│   ├── milestone2_report.pdf # 2-page evaluation report (M2)
│   ├── runs_log.csv          # Evidence of 5 tracked runs
│   └── plots/                # ROC curves, confusion matrices, error slices
├── Dockerfile                # Docker packaging for CLI inference
└── requirements.txt
```

## How to Run

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ingest LFW (downloads ~200 MB on first run)

```bash
python scripts/ingest_lfw.py --config configs/m4.yaml
```

### 3. Generate Pair CSVs

```bash
python scripts/make_pairs.py --config configs/m4.yaml --mode baseline
python scripts/make_pairs.py --config configs/m4.yaml --mode filtered
```

### 4. Reproduce the Final Experiment (run05)

```bash
python scripts/run_experiment.py --config configs/m4.yaml \
    --run-id run05 --features hog --metric cosine --pair-mode filtered --n-thresh 20
```

### 5. Run the CLI Verifier

```bash
python scripts/verify_pair.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg
python scripts/verify_pair.py --img1 face1.jpg --img2 face2.jpg --json
```

### 6. Run Profiling

```bash
python scripts/profile_system.py --config configs/m4.yaml
```

### 7. Docker

```bash
docker build -t lfw-verify .
docker run --rm -v $(pwd)/samples:/data lfw-verify \
    --img1 /data/face1.jpg --img2 /data/face2.jpg --json
```

### 8. Run Tests

```bash
pytest tests/ -v
# Expected: 37 passed
```

## Tracked Runs

| Run | Features | Metric | Pair Mode | Val AUC | Test F1 | Threshold |
|-----|----------|--------|-----------|---------|---------|-----------|
| run01 | HOG | cosine | baseline | 0.6436 | 0.4294 | 0.8085 |
| run02 | HOG | euclidean | baseline | 0.6436 | 0.4501 | 3.0682 |
| run03 | pixel | cosine | baseline | 0.6188 | 0.5159 | 0.9413 |
| run04 | HOG | cosine | filtered | 0.6202 | 0.4028 | 0.8165 |
| run05 | HOG | cosine | filtered | 0.6202 | 0.4632 | 0.8045 |

**Best run by test F1: run05** (HOG + cosine + filtered pairs, fine-grained threshold sweep)

## Threshold Selection Rule

The operating threshold is selected on the **validation split** by maximising balanced accuracy
(Youden-J statistic: `TPR - FPR`). This rule is stated before any test-set inspection and applied
consistently across all runs. See `src/evaluation.py:calibrate_threshold`.

## Final Git Tag

```bash
git tag v1.0-final
```
