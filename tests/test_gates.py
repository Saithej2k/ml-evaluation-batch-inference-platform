from ml_eval_platform.models import EvaluationRun
from ml_eval_platform.services.gates import check_metric_rows, compare_runs


def test_compare_runs_passes_within_thresholds():
    baseline = EvaluationRun(run_id="base", accuracy=0.91, p95_latency_ms=100)
    candidate = EvaluationRun(run_id="candidate", accuracy=0.895, p95_latency_ms=113)

    outcome = compare_runs(baseline, candidate)

    assert outcome.passed is True
    assert outcome.reasons == []


def test_compare_runs_fails_on_accuracy_and_latency_regression():
    baseline = EvaluationRun(run_id="base", accuracy=0.91, p95_latency_ms=100)
    candidate = EvaluationRun(run_id="candidate", accuracy=0.87, p95_latency_ms=130)

    outcome = compare_runs(baseline, candidate)

    assert outcome.passed is False
    assert len(outcome.reasons) == 2


def test_check_metric_rows_matches_baseline_keys():
    baseline = [
        {
            "dataset_name": "product-reviews",
            "dataset_version": "2024-01",
            "model_name": "sentiment-classifier",
            "model_version": "v6",
            "accuracy": 0.9,
            "p95_latency_ms": 30,
        }
    ]
    current = [
        {
            "dataset_name": "product-reviews",
            "dataset_version": "2024-01",
            "model_name": "sentiment-classifier",
            "model_version": "v6",
            "accuracy": 0.91,
            "p95_latency_ms": 31,
        }
    ]

    outcome = check_metric_rows(baseline, current)

    assert outcome["passed"] is True
