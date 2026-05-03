# Reproducibility Checklist — LFW Face Verification Pipeline

**Final Tag:** `v1.0-final`
**Config:** `configs/m4.yaml`

---

## 1. Environment Setup

```bash
git clone <repo-url>
cd lfw-verification
git checkout v1.0-final

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Ingest LFW Dataset

Downloads ~200 MB on first run (cached in `~/scikit_learn_data/`).

```bash
python scripts/ingest_lfw.py --config configs/m4.yaml
```

**Expected output:** `outputs/manifest.json` with train/val/test split counts.

## 3. Generate Pair CSVs

```bash
python scripts/make_pairs.py --config configs/m4.yaml --mode baseline
python scripts/make_pairs.py --config configs/m4.yaml --mode filtered
```

**Expected output:** CSV files in `outputs/pairs/` (train, val, test for each mode).

## 4. Reproduce Final Experiment (run05)

```bash
python scripts/run_experiment.py --config configs/m4.yaml \
    --run-id run05 --features hog --metric cosine --pair-mode filtered --n-thresh 20
```

**Expected output:**
- Threshold: ~0.8045
- Test F1: ~0.4632
- Plots saved to `outputs/plots/`
- Run logged to `outputs/runs_log.csv`

## 5. Run Profiling

```bash
python scripts/profile_system.py --config configs/m4.yaml
```

**Expected output:**
- `outputs/profiling/profiling_results.json`
- `outputs/profiling/batch_sensitivity.csv`
- Console output with per-stage latency and batch-size sensitivity

## 6. Run CLI Verification

```bash
# On two sample LFW images (after ingestion)
python scripts/verify_pair.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg --json
```

## 7. Docker Build and Run

```bash
docker build -t lfw-verify .
docker run --rm -v $(pwd)/samples:/data lfw-verify \
    --img1 /data/face1.jpg --img2 /data/face2.jpg --json
```

## 8. Run Tests

```bash
pytest tests/ -v
```

**Expected:** 37 tests passed.

## 9. Key Artifact Locations

| Artifact | Path |
|----------|------|
| Final config | `configs/m4.yaml` |
| System Card | `reports/system_card.md` |
| Profiling report | `reports/profiling_report.md` |
| Reproducibility checklist | `reports/reproducibility_checklist.md` |
| Milestone 2 report (PDF) | `reports/milestone2_report.pdf` |
| Tracked runs log | `reports/runs_log.csv` |
| ROC / CM / error-slice plots | `reports/plots/` |
| Profiling results (JSON) | `outputs/profiling/profiling_results.json` |
| Batch sensitivity (CSV) | `outputs/profiling/batch_sensitivity.csv` |
| Dockerfile | `Dockerfile` |
| CLI entrypoint | `scripts/verify_pair.py` |

## 10. Final Git Tag

```bash
git tag v1.0-final
git push origin v1.0-final
```
