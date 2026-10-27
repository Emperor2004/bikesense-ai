# Implementation Plan: Drift-Aware Bike-Share Demand Forecasting

8 weeks · 3 members · USD 0 infrastructure

---

## 0. Corrections to settle in Week 1 (gaps found in the proposal)

| # | Gap | Decision to record |
|---|---|---|
| 1 | "Latest holdout month" for the promotion gate is undefined. If the challenger trains on M1-M10, M10 is no longer held out. | Gate is evaluated on a month **excluded from the challenger's training**. For v2: train M1-M9, validate M10, then refit on M1-M10 only after the gate decision. Owner: Member B. |
| 2 | Real drift between chosen months may be weak, so the demo would show nothing. | Pick M1-M14 in Week 1 by **measuring** drift (Evidently on candidate months), not by assumption. Keep a labelled injected-shift test as fallback. Owner: Member B (EDA) + Member C (drift check). |
| 3 | Citi Bike file schemas and availability vary by period. | Confirm all 14 files download and pass schema validation in Week 1. Owner: Member A. |
| 4 | Member A carries ingestion + features + DVC + replay + CI, which is as heavy as Member C. | Replay driver is built early (Week 3-4, not Week 4 only); Member B takes EDA, station selection and the demo script. |
| 5 | Alert threshold (1.25x MAE, 7-day window) may be noisy on 30 stations. | Calibrate threshold on v1 replay in Week 5; store in `configs/monitoring.yaml`. Owner: Member C. |
| 6 | Weather features are reanalysis values (optimistic bias). | Document as a limitation in README and model card. Owner: Member B. |

---

## 1. Repository structure (final target)

```
.
├── README.md
├── requirements.txt
├── Makefile
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── .github/
│   ├── CODEOWNERS
│   └── workflows/ (ci.yml, retrain.yml)
├── configs/ (features.yaml, model.yaml, gate.yaml, monitoring.yaml)
├── data/ (raw/, external/, processed/  as .dvc pointers)
├── notebooks/
├── src/
│   ├── data/ (ingest.py, validate.py, weather.py, holidays.py)
│   ├── features/ (build_features.py)
│   ├── replay/ (replay_driver.py)
│   ├── models/ (baseline.py, train.py, evaluate.py, metrics.py)
│   ├── registry/ (register.py, promote.py, rollback.py)
│   └── monitoring/ (join_actuals.py, rolling_error.py, drift_report.py, alerts.py)
├── service/ (app/, Dockerfile)
├── monitoring/ (prometheus.yml, grafana/)
├── docker-compose.yml
├── tests/ (data/, features/, models/, registry/, service/, monitoring/)
├── scripts/ (fresh_clone_test.sh, registry_audit.py, run_demo.sh)
├── reports/ (evaluation/, monitoring/)
└── docs/ (architecture.md, data_card.md, lineage_report.md, model_card.md, monitoring_guide.md, slides/)
```

---

## 2. GitHub push ownership (what each member must push)

**Rule: every file and folder has exactly one owner. Owners push only their own paths.** This avoids merge conflicts and makes each member's contribution unambiguous in the commit history.

### Member A: Data and automation lead

| Path | Content |
|---|---|
| `requirements.txt`, `Makefile`, `.gitignore` | Repo scaffold (Week 1) |
| `dvc.yaml`, `dvc.lock`, `params.yaml`, `.dvc/` | Pipeline definition and pinned hashes |
| `data/**/*.dvc` | DVC pointers for raw, weather, processed data |
| `configs/features.yaml` | Feature configuration |
| `src/data/` | `ingest.py`, `validate.py`, `weather.py`, `holidays.py` |
| `src/features/` | `build_features.py` |
| `src/replay/` | `replay_driver.py` |
| `tests/data/`, `tests/features/` | Schema, leakage and feature tests |
| `.github/workflows/` | `ci.yml`, `retrain.yml` |
| `.github/CODEOWNERS` | Ownership rules from this section |
| `scripts/fresh_clone_test.sh` | Reproducibility test |
| `docs/data_card.md`, `docs/lineage_report.md` | Data and lineage documentation |
| `docs/slides/01_data_versioning_and_lineage.pdf` | Presentation section |

### Member B: Modeling and registry lead

| Path | Content |
|---|---|
| `README.md` | Final README (all members supply a section; B merges) |
| `notebooks/` | EDA, station selection, month selection and drift check |
| `configs/model.yaml`, `configs/gate.yaml` | Model hyperparameters, promotion margin |
| `src/models/` | `baseline.py`, `train.py`, `evaluate.py`, `metrics.py` |
| `src/registry/` | `register.py`, `promote.py`, `rollback.py` |
| `tests/models/`, `tests/registry/` | Metric, gate and rollback tests |
| `scripts/registry_audit.py` | Verifies 100% lineage |
| `scripts/run_demo.sh` | End-to-end demo runner |
| `reports/evaluation/` | Evaluation tables (MAE, RMSE, shortfall) per window and model |
| `docs/architecture.md`, `docs/model_card.md` | Architecture and model documentation |
| `docs/slides/02_models_and_results.pdf` | Presentation section |

### Member C: Serving and monitoring lead

| Path | Content |
|---|---|
| `service/` | FastAPI app (`main.py`, `schemas.py`, `model_loader.py`, `prediction_log.py`), `Dockerfile` |
| `docker-compose.yml` | Service, MLflow, monitoring stack |
| `configs/monitoring.yaml` | Drift and error thresholds |
| `src/monitoring/` | `join_actuals.py`, `rolling_error.py`, `drift_report.py`, `alerts.py` |
| `monitoring/` | `prometheus.yml`, Grafana dashboards (optional) |
| `tests/service/`, `tests/monitoring/` | API and monitoring tests |
| `reports/monitoring/` | Evidently HTML reports for v1, v2, v3 replays |
| `docs/monitoring_guide.md` | How to read reports and alerts |
| `docs/slides/03_serving_and_monitoring.pdf` | Presentation section |

### Push rules (so every member's work is visible)

1. **One branch per member per task**, named `a/<task>`, `b/<task>`, `c/<task>`. Merge to `main` only through a pull request.
2. **Cross-review:** A's PRs reviewed by B, B's by C, C's by A.
3. **Commit under your own GitHub account** (check `git config user.name` and `user.email` before the first commit).
4. **Minimum cadence:** at least 2 merged commits or PRs per member per week from Week 1.
5. **Do not edit another member's paths.** If a change is needed, open an issue or request the owner. `CODEOWNERS` enforces review by the owner.
6. **Big files stay out of Git.** Raw data, weather cache and model artifacts go through DVC or MLflow; only pointers and configs are pushed.
7. **Final tag:** `v1.0` created after Week 7 dry run; the final tree must match section 1.

### Weekly push checkpoints

| Week | Member A must have pushed | Member B must have pushed | Member C must have pushed |
|---|---|---|---|
| 1 | Scaffold, `.gitignore`, `requirements.txt`, DVC init, `CODEOWNERS`, feature schema contract (`docs/data_card.md` draft) | `configs/model.yaml`, metric definitions in `src/models/metrics.py`, MLflow setup notes, alias contract in `docs/architecture.md` | `docker-compose.yml` skeleton, prediction log schema, service stub in `service/` |
| 2 | `src/data/*`, `src/features/*`, `dvc.yaml`, dataset v1 `.dvc` files | `notebooks/` (EDA, station and month selection), `baseline.py`, Ridge in `train.py` | `service/app/main.py` endpoint with dummy model, prediction logging |
| 3 | `tests/data/`, `tests/features/`, finalised v1, `ci.yml` (lint + tests) | LightGBM runs, `register.py`, first champion, `reports/evaluation/` v1 | `model_loader.py` loading champion alias, `Dockerfile` |
| 4 | `src/replay/replay_driver.py` | Per-window evaluation, shortfall metric in `evaluate.py` | `join_actuals.py`, `rolling_error.py` |
| 5 | Dataset v2 release (`dvc.lock` update), tests for replay | `promote.py`, `configs/gate.yaml`, `tests/registry/` | `drift_report.py`, `alerts.py`, `configs/monitoring.yaml` |
| 6 | Dataset v3, `retrain.yml` | `rollback.py`, gate tests, challenger runs on v2/v3 | `reports/monitoring/` for v2 and v3, optional `monitoring/` Grafana |
| 7 | `docs/lineage_report.md`, `scripts/fresh_clone_test.sh` | `scripts/registry_audit.py`, `docs/model_card.md`, final results tables, `README.md` | Monitoring report export, `docs/monitoring_guide.md`, `tests/service/`, `tests/monitoring/` |
| 8 | `docs/slides/01_*.pdf` | `scripts/run_demo.sh`, `docs/slides/02_*.pdf` | `docs/slides/03_*.pdf` |

---

## 3. Week-by-week plan

### Week 1: Framing, audit, scaffold
- **All:** agree on interface contracts (feature table schema: A; model name and alias convention: B; prediction log schema: C).
- **A:** repo scaffold, DVC init (local remote), confirm all 14 monthly files download and parse.
- **B:** evaluation protocol, metric definitions, MLflow setup (SQLite backend), alias contract.
- **C:** Docker Compose skeleton, prediction log schema, service stub.
- **Exit:** repo with agreed contracts; month candidates shortlisted.

### Week 2: Data pipeline and baselines
- **A:** ingestion, weather caching (one call per month-chunk), holiday flags, feature pipeline, dataset v1 in DVC.
- **B:** EDA, fix 30 busiest stations from v1, baseline and Ridge on a sample; select M1-M14 by measured drift.
- **C:** FastAPI endpoint and prediction logging with dummy model.
- **Exit:** dataset v1 tracked; baseline MAE recorded.

### Week 3: First champion
- **A:** schema tests, leakage test (every feature uses only information available at 00:00 issue time), CI lint and tests.
- **B:** LightGBM hyperparameter runs, register best as challenger, set first champion.
- **C:** service loads champion alias; Dockerfile working.
- **Exit:** first registered champion, serving forecasts.

### Week 4: Replay and actuals join
- **A:** replay driver feeding daily windows.
- **B:** per-window evaluation with MAE, RMSE, shortfall.
- **C:** join predictions to actuals after 24 h; rolling error computation.
- **Exit:** predictions logged and joined with actuals for the v1 replay window.

### Week 5: Monitoring and gate
- **A:** release dataset v2.
- **B:** promotion gate logic (margin from `configs/gate.yaml`), challenger runs.
- **C:** Evidently reports, thresholds calibrated on v1 replay, alerts.
- **Exit:** monitoring detects drift on replay; gate decision reproducible.

### Week 6: Closed loop
- **A:** dataset v3; retrain workflow (manual or alert-triggered; human approves alias change).
- **B:** rollback via aliases, gate tests.
- **C:** monitoring over v2 and v3 replay; optional Grafana.
- **Exit:** detect, retrain, compare, promote, rollback runs end to end.

### Week 7: Integration
- **A:** lineage report, fresh-clone reproducibility test.
- **B:** registry audit script, results tables, model card, README.
- **C:** monitoring report export, guide, remaining tests.
- **All:** end-to-end dry run; tag `v1.0`.

### Week 8: Presentation
- Slides (each member their own pillar), demo rehearsal, buffer for fixes.

---

## 4. Gantt chart

| Phase | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 |
|---|---|---|---|---|---|---|---|---|
| Framing, audit, repo and DVC setup | ■ | | | | | | | |
| Ingestion, weather, features, dataset v1 | | ■ | | | | | | |
| Training, MLflow, first champion | | | ■ | | | | | |
| Serving, prediction log, actuals join | | | | ■ | | | | |
| Monitoring and thresholds | | | | | ■ | | | |
| v2/v3, retrain, gate, rollback, CI | | | | | | ■ | | |
| Lineage, docs, dry run | | | | | | | ■ | |
| Slides, rehearsal, buffer | | | | | | | | ■ |

---

## 5. Definition of done

| Criterion | Target | Verified by | Owner |
|---|---|---|---|
| Forecast quality | Champion beats seasonal-naive MAE on each replay window | `reports/evaluation/` | B |
| Business proxy | Shortfall reported next to MAE for every window and model | `evaluate.py` output | B |
| Lineage | 100% of registered versions carry Git commit and dataset version | `scripts/registry_audit.py` | B |
| Reproducibility | Clean copy + `dvc repro` reproduces metrics within tolerance | `scripts/fresh_clone_test.sh` | A |
| Detection | Alert within one 7-day window of visible degradation | `reports/monitoring/` | C |
| Rollback | Alias revert restores service in under 5 minutes, no redeploy | Timed demo step | B + C |
| Contribution visible | Each member has merged commits in own paths every week | Commit history on `main` | All |

---

## 6. Risks and mitigations

| Risk | Mitigation | Owner |
|---|---|---|
| Months show little real drift | Measure in Week 1-2; fallback labelled injected shift | B, C |
| Citi Bike schema or availability differs | Validate all 14 files in Week 1; schema tests | A |
| Actuals join is complex | Build in Week 4 before dashboards | C |
| Uneven load | Interface contracts, dummy model and sample data as stand-ins, weekly 15-minute integration run; B takes Evidently reports if C falls behind | All |
| Merge conflicts | Single-owner paths, `CODEOWNERS`, branch per task | A |
| Open-Meteo limits or terms change | Cached DVC files, CC BY 4.0 attribution | A |
| Scope creep (Kubernetes, feature store) | Out of scope until closed loop works | All |
