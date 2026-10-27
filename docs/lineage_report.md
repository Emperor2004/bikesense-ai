# Lineage Report: BikeSense AI

## 1. Lineage Strategy
Lineage tracking guarantees that for any forecast model in production:
1. The exact raw and processed data can be reproduced via **DVC hashes** pinned in `dvc.lock`.
2. The exact code state is identified via **Git commit hash**.
3. Training parameters, metrics, and models are linked via **MLflow run ID**.

```
[Git Commit] ───────┐
[DVC Hash]   ───────┼──► [MLflow Run] ──► [Model Registry: champion/challenger]
[params.yaml] ──────┘
```

## 2. Pinned Artifacts
- **Ingestion Hash:** Hash of raw Citi Bike data + validation schemas.
- **Feature Hash:** Hash of transformed feature parquets in `data/processed/`.
- **Model Artifact:** Stored with full input schema, parameters, and evaluated metrics.

## 3. Auditing Lineage
Lineage compliance is audited via `scripts/registry_audit.py` (owned by Member B), ensuring 100% of registered versions have valid `git_commit` and `dvc_dataset_hash` tags.
