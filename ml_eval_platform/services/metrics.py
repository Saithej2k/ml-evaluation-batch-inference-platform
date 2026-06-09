from math import ceil


def p95_latency(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, ceil(len(ordered) * 0.95) - 1)
    return round(float(ordered[index]), 4)


def rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)

