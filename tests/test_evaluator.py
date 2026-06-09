from sqlalchemy import select

from ml_eval_platform.config import Settings
from ml_eval_platform.models import DatasetVersion, FailureLog, ModelVersion
from ml_eval_platform.schemas import TextRecord
from ml_eval_platform.services.evaluator import evaluate_records


def test_evaluator_retries_transient_failures(db_session):
    dataset, model = _catalog(db_session, transient_error_rate=1.0, permanent_error_rate=0.0)
    settings = Settings(
        BATCH_RETRY_ATTEMPTS=2,
        BATCH_RETRY_BACKOFF_SECONDS=0,
        STORE_PREDICTIONS=True,
    )

    result = evaluate_records(
        session=db_session,
        dataset_version=dataset,
        model_version=model,
        records=[
            TextRecord(
                external_id="retry-me",
                text="The workflow is fast, accurate, and reliable.",
                label="positive",
            )
        ],
        settings=settings,
    )

    assert result.run.status == "completed"
    assert result.run.record_count == 1
    assert result.run.failure_count == 0
    assert result.run.error_rate == 0
    assert result.run.p95_latency_ms > 0


def test_evaluator_persists_failure_logs(db_session):
    dataset, model = _catalog(db_session, transient_error_rate=0.0, permanent_error_rate=1.0)
    settings = Settings(
        BATCH_RETRY_ATTEMPTS=2,
        BATCH_RETRY_BACKOFF_SECONDS=0,
        STORE_PREDICTIONS=False,
    )

    result = evaluate_records(
        session=db_session,
        dataset_version=dataset,
        model_version=model,
        records=[
            TextRecord(
                external_id="bad-1", text="The result is broken and slow.", label="negative"
            ),
            TextRecord(
                external_id="bad-2", text="The result is fast and useful.", label="positive"
            ),
        ],
        settings=settings,
    )

    assert result.run.failure_count == 2
    assert result.run.error_rate == 1
    logs = db_session.scalars(
        select(FailureLog).where(FailureLog.evaluation_run_id == result.run.id)
    ).all()
    assert len(logs) == 2
    assert {log.stage for log in logs} == {"inference"}


def _catalog(
    db_session,
    transient_error_rate: float,
    permanent_error_rate: float,
) -> tuple[DatasetVersion, ModelVersion]:
    dataset = DatasetVersion(
        name="product-reviews",
        version="test",
        labels=["negative", "neutral", "positive"],
        size=1,
    )
    model = ModelVersion(
        name="sentiment-classifier",
        version="v6",
        labels=["negative", "neutral", "positive"],
        seed=23,
        metadata_json={
            "quality": 0.95,
            "latency_ms": 20,
            "latency_jitter_ms": 2,
            "transient_error_rate": transient_error_rate,
            "permanent_error_rate": permanent_error_rate,
        },
    )
    db_session.add_all([dataset, model])
    db_session.commit()
    db_session.refresh(dataset)
    db_session.refresh(model)
    return dataset, model
