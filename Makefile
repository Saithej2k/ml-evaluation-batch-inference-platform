.PHONY: install lint test run init-db demo gate docker-up docker-test

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest

run:
	uvicorn ml_eval_platform.main:app --reload

init-db:
	python -m ml_eval_platform.cli init-db

demo:
	python -m ml_eval_platform.cli run-demo --records-per-dataset 3200 --output artifacts/evaluation-results.json

gate:
	python -m ml_eval_platform.cli check-gate --baseline benchmarks/baseline_metrics.json --current artifacts/evaluation-results.json

docker-up:
	docker compose up --build api

docker-test:
	docker compose --profile test up --build --abort-on-container-exit test

