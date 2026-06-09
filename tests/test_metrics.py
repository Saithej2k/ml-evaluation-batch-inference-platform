from ml_eval_platform.services.metrics import p95_latency, rate


def test_p95_latency_uses_nearest_rank():
    assert p95_latency([1, 2, 3, 4, 5, 100]) == 100
    assert p95_latency([1, 2, 3, 4, 5]) == 5
    assert p95_latency([]) == 0


def test_rate_handles_empty_denominator():
    assert rate(3, 10) == 0.3
    assert rate(1, 0) == 0
