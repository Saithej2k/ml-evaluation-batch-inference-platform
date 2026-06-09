# ML Evaluation and Batch-Inference Platform

FastAPI platform for batch NLP text-classification evaluation. It evaluates versioned PyTorch
models against versioned datasets, stores accuracy, p95 latency, and error-rate metrics in
PostgreSQL, and exposes side-by-side comparisons plus release gates for regression checks.

## Features

- FastAPI catalog endpoints for dataset versions, model versions, evaluation runs, comparisons,
  and release-gate checks.
- PyTorch text classifier profiles for six model versions with deterministic latency and
  failure behavior for reproducible batch runs.
- Batch evaluator with Pydantic input-schema validation, retry handling for transient inference
  failures, and structured failure logs persisted in PostgreSQL.
- Demo workload covering 4 versioned datasets x 6 model versions. The default run processes
  76,800 predictions from 3,200 records per dataset.
- Docker Compose environment for API, PostgreSQL, and test execution.
- GitHub Actions workflow that runs lint/tests, executes an evaluation smoke workload, and fails
  when accuracy drops by more than 2 percentage points or p95 latency regresses by more than 15%.

## Quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Start PostgreSQL and the API:

```bash
docker compose up --build api
```

Open the API at [http://localhost:8000/docs](http://localhost:8000/docs).

Run the demo workload:

```bash
python -m ml_eval_platform.cli init-db
python -m ml_eval_platform.cli run-demo --records-per-dataset 3200 --output artifacts/evaluation-results.json
python -m ml_eval_platform.cli check-gate --baseline benchmarks/baseline_metrics.json --current artifacts/evaluation-results.json
```

## API Overview

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health |
| `POST` | `/datasets` | Register a dataset version |
| `GET` | `/datasets` | List dataset versions |
| `POST` | `/models` | Register a model version |
| `GET` | `/models` | List model versions |
| `POST` | `/evaluations` | Run a batch evaluation |
| `GET` | `/evaluations` | List recent evaluation runs |
| `GET` | `/evaluations/compare` | Compare stored metrics side by side |
| `POST` | `/release-gates/check` | Compare a candidate run against a baseline run |
| `POST` | `/demo/run` | Execute the built-in 4 x 6 demo workload |

Example batch evaluation request:

```json
{
  "dataset_version_id": 1,
  "model_version_id": 1,
  "records": [
    {
      "external_id": "sample-001",
      "text": "The checkout flow is fast, accurate, and reliable.",
      "label": "positive"
    }
  ],
  "run_config": {
    "source": "manual"
  }
}
```

## Release Gates

The release gate compares a candidate result with baseline metrics and fails when either threshold
is exceeded:

- accuracy decrease greater than `0.02`
- p95 latency increase greater than `0.15`

The same logic is available through the API and CLI:

```bash
python -m ml_eval_platform.cli check-gate \
  --baseline benchmarks/baseline_metrics.json \
  --current artifacts/evaluation-results.json
```

## Development

```bash
make install
make lint
make test
make demo
```

Docker-based tests:

```bash
docker compose --profile test up --build --abort-on-container-exit test
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://ml_eval:ml_eval@localhost:5432/ml_eval` | SQLAlchemy database URL |
| `AUTO_CREATE_TABLES` | `true` | Create tables on API startup |
| `BATCH_RETRY_ATTEMPTS` | `3` | Retry attempts for transient inference errors |
| `BATCH_RETRY_BACKOFF_SECONDS` | `0.05` | Initial exponential backoff interval |
| `STORE_PREDICTIONS` | `false` | Persist per-record prediction rows |
| `LOG_LEVEL` | `INFO` | Application log level |

