import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ml_eval_platform.config import Settings, get_settings
from ml_eval_platform.models import (
    DatasetVersion,
    EvaluationRun,
    FailureLog,
    ModelVersion,
    PredictionResult,
    utcnow,
)
from ml_eval_platform.schemas import TextRecord
from ml_eval_platform.services.inference import (
    PermanentInferenceError,
    Prediction,
    TorchTextClassifier,
    TransientInferenceError,
)
from ml_eval_platform.services.metrics import p95_latency, rate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchEvaluationResult:
    run: EvaluationRun
    correct_count: int
    successful_count: int


def evaluate_records(
    session: Session,
    dataset_version: DatasetVersion,
    model_version: ModelVersion,
    records: Sequence[TextRecord],
    run_config: dict[str, object] | None = None,
    settings: Settings | None = None,
) -> BatchEvaluationResult:
    if not records:
        raise ValueError("evaluation requires at least one record")

    settings = settings or get_settings()
    _validate_records(dataset_version, model_version, records)

    run = EvaluationRun(
        run_id=uuid.uuid4().hex,
        dataset_version_id=dataset_version.id,
        model_version_id=model_version.id,
        status="running",
        record_count=len(records),
        run_config=run_config or {},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    classifier = TorchTextClassifier(model_version)
    latencies: list[float] = []
    predictions: list[PredictionResult] = []
    failures: list[FailureLog] = []
    correct_count = 0
    successful_count = 0

    for record in records:
        try:
            prediction = _predict_with_retry(classifier, record, settings)
        except (PermanentInferenceError, TransientInferenceError) as exc:
            failures.append(_failure_log(run.id, record, "inference", exc))
            if settings.store_predictions:
                predictions.append(_failed_prediction(run.id, record, exc))
            logger.warning(
                "batch_prediction_failed",
                extra={
                    "run_id": run.run_id,
                    "external_id": record.external_id,
                    "stage": "inference",
                    "error_type": type(exc).__name__,
                },
            )
            continue

        successful_count += 1
        latencies.append(prediction.latency_ms)
        if prediction.label == record.label:
            correct_count += 1
        if settings.store_predictions:
            predictions.append(_prediction_result(run.id, record, prediction))

    if predictions:
        session.add_all(predictions)
    if failures:
        session.add_all(failures)

    run.status = "completed"
    run.record_count = len(records)
    run.accuracy = rate(correct_count, len(records))
    run.p95_latency_ms = p95_latency(latencies)
    run.error_rate = rate(len(failures), len(records))
    run.failure_count = len(failures)
    run.completed_at = utcnow()
    session.add(run)
    session.commit()
    session.refresh(run)

    logger.info(
        "evaluation_completed",
        extra={
            "run_id": run.run_id,
            "external_id": None,
            "stage": "evaluation",
            "record_count": len(records),
            "accuracy": run.accuracy,
            "p95_latency_ms": run.p95_latency_ms,
            "error_rate": run.error_rate,
        },
    )
    return BatchEvaluationResult(run=run, correct_count=correct_count, successful_count=successful_count)


def _predict_with_retry(
    classifier: TorchTextClassifier,
    record: TextRecord,
    settings: Settings,
) -> Prediction:
    retrying = Retrying(
        stop=stop_after_attempt(settings.batch_retry_attempts),
        wait=wait_exponential(multiplier=settings.batch_retry_backoff_seconds, min=0, max=1),
        retry=retry_if_exception_type(TransientInferenceError),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            return classifier.predict(record, attempt=attempt.retry_state.attempt_number)
    raise RuntimeError("retry loop exited without a prediction")


def _validate_records(
    dataset_version: DatasetVersion,
    model_version: ModelVersion,
    records: Sequence[TextRecord],
) -> None:
    dataset_labels = set(dataset_version.labels)
    model_labels = set(model_version.labels)
    missing_from_dataset = {record.label for record in records if record.label not in dataset_labels}
    missing_from_model = {record.label for record in records if record.label not in model_labels}
    if missing_from_dataset:
        labels = ", ".join(sorted(missing_from_dataset))
        raise ValueError(f"records contain labels not present in dataset version: {labels}")
    if missing_from_model:
        labels = ", ".join(sorted(missing_from_model))
        raise ValueError(f"records contain labels not present in model version: {labels}")


def _prediction_result(
    run_id: int,
    record: TextRecord,
    prediction: Prediction,
) -> PredictionResult:
    return PredictionResult(
        evaluation_run_id=run_id,
        external_id=record.external_id,
        expected_label=record.label,
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        latency_ms=prediction.latency_ms,
        succeeded=True,
    )


def _failed_prediction(
    run_id: int,
    record: TextRecord,
    exc: Exception,
) -> PredictionResult:
    return PredictionResult(
        evaluation_run_id=run_id,
        external_id=record.external_id,
        expected_label=record.label,
        predicted_label=None,
        confidence=None,
        latency_ms=None,
        succeeded=False,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def _failure_log(
    run_id: int,
    record: TextRecord,
    stage: str,
    exc: Exception,
) -> FailureLog:
    return FailureLog(
        evaluation_run_id=run_id,
        external_id=record.external_id,
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc),
        payload={"label": record.label, "text_length": len(record.text)},
    )

