from ml_eval_platform.models import EvaluationRun
from ml_eval_platform.schemas import ComparisonRow, EvaluationSummary


def run_to_summary(run: EvaluationRun) -> EvaluationSummary:
    return EvaluationSummary(
        run_id=run.run_id,
        status=run.status,
        dataset_name=run.dataset_version.name,
        dataset_version=run.dataset_version.version,
        model_name=run.model_version.name,
        model_version=run.model_version.version,
        record_count=run.record_count,
        accuracy=run.accuracy,
        p95_latency_ms=run.p95_latency_ms,
        error_rate=run.error_rate,
        failure_count=run.failure_count,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )


def run_to_comparison_row(run: EvaluationRun) -> ComparisonRow:
    return ComparisonRow(
        dataset_name=run.dataset_version.name,
        dataset_version=run.dataset_version.version,
        model_name=run.model_version.name,
        model_version=run.model_version.version,
        run_id=run.run_id,
        accuracy=run.accuracy,
        p95_latency_ms=run.p95_latency_ms,
        error_rate=run.error_rate,
        failure_count=run.failure_count,
        record_count=run.record_count,
        completed_at=run.completed_at,
    )


def run_to_metric_row(run: EvaluationRun) -> dict[str, object]:
    row = run_to_comparison_row(run)
    return row.model_dump(mode="json")
