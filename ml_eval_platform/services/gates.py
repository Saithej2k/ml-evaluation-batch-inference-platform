from dataclasses import dataclass
from typing import Any

from ml_eval_platform.models import EvaluationRun


@dataclass(frozen=True)
class RegressionGateOutcome:
    passed: bool
    accuracy_delta: float
    latency_delta_pct: float
    reasons: list[str]


def compare_runs(
    baseline: EvaluationRun,
    candidate: EvaluationRun,
    max_accuracy_drop: float = 0.02,
    max_latency_increase: float = 0.15,
) -> RegressionGateOutcome:
    accuracy_delta = round(candidate.accuracy - baseline.accuracy, 6)
    latency_delta_pct = _latency_delta_pct(baseline.p95_latency_ms, candidate.p95_latency_ms)
    reasons: list[str] = []

    if accuracy_delta < -max_accuracy_drop:
        reasons.append(
            "accuracy decreased by "
            f"{abs(accuracy_delta):.2%}, above the {max_accuracy_drop:.2%} limit"
        )
    if latency_delta_pct > max_latency_increase:
        reasons.append(
            "p95 latency increased by "
            f"{latency_delta_pct:.2%}, above the {max_latency_increase:.2%} limit"
        )

    return RegressionGateOutcome(
        passed=not reasons,
        accuracy_delta=accuracy_delta,
        latency_delta_pct=latency_delta_pct,
        reasons=reasons,
    )


def check_metric_rows(
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    max_accuracy_drop: float = 0.02,
    max_latency_increase: float = 0.15,
) -> dict[str, Any]:
    current_by_key = {_metric_key(row): row for row in current_rows}
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for baseline in baseline_rows:
        key = _metric_key(baseline)
        current = current_by_key.get(key)
        label = " / ".join(key)
        if current is None:
            message = f"missing current metrics for {label}"
            failures.append(message)
            results.append({"key": key, "passed": False, "reasons": [message]})
            continue

        baseline_accuracy = float(baseline["accuracy"])
        current_accuracy = float(current["accuracy"])
        baseline_latency = float(baseline["p95_latency_ms"])
        current_latency = float(current["p95_latency_ms"])
        accuracy_delta = round(current_accuracy - baseline_accuracy, 6)
        latency_delta_pct = _latency_delta_pct(baseline_latency, current_latency)
        reasons: list[str] = []

        if accuracy_delta < -max_accuracy_drop:
            reasons.append(
                "accuracy decreased by "
                f"{abs(accuracy_delta):.2%}, above the {max_accuracy_drop:.2%} limit"
            )
        if latency_delta_pct > max_latency_increase:
            reasons.append(
                "p95 latency increased by "
                f"{latency_delta_pct:.2%}, above the {max_latency_increase:.2%} limit"
            )

        if reasons:
            failures.extend(f"{label}: {reason}" for reason in reasons)
        results.append(
            {
                "key": key,
                "passed": not reasons,
                "accuracy_delta": accuracy_delta,
                "latency_delta_pct": latency_delta_pct,
                "reasons": reasons,
            }
        )

    return {"passed": not failures, "failures": failures, "results": results}


def _latency_delta_pct(baseline_latency: float, candidate_latency: float) -> float:
    if baseline_latency <= 0:
        return 0.0 if candidate_latency <= 0 else 1.0
    return round((candidate_latency - baseline_latency) / baseline_latency, 6)


def _metric_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["dataset_name"]),
        str(row["dataset_version"]),
        str(row["model_name"]),
        str(row["model_version"]),
    )
