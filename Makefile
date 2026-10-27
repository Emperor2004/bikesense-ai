.PHONY: data weather holidays features train register serve replay monitor test demo clean

# ─── Data pipeline ───
data: weather holidays
	python -m src.data.ingest

weather:
	python -m src.data.weather

holidays:
	python -m src.data.holidays

features: data
	python -m src.features.build_features

# ─── Model pipeline (Member B) ───
train:
	python -m src.models.train

register:
	python -m src.registry.register

# ─── Serving (Member C) ───
serve:
	docker compose up service -d

# ─── Replay & monitoring ───
replay:
	python -m src.replay.replay_driver

monitor:
	python -m src.monitoring.drift_report

# ─── Quality ───
test:
	ruff check src/ tests/
	pytest tests/ -v --tb=short

# ─── Demo ───
demo:
	bash scripts/run_demo.sh

# ─── Cleanup ───
clean:
	@echo "Removing generated data files..."
	rm -rf data/raw/*.csv data/raw/*.parquet
	rm -rf data/external/weather/*.parquet data/external/holidays.parquet
	rm -rf data/processed/*.parquet
	rm -rf data/predictions/
