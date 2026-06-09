def test_api_runs_evaluation_and_release_gate(client):
    dataset_response = client.post(
        "/datasets",
        json={
            "name": "product-reviews",
            "version": "test",
            "labels": ["negative", "neutral", "positive"],
            "size": 3,
        },
    )
    assert dataset_response.status_code == 201

    model_response = client.post(
        "/models",
        json={
            "name": "sentiment-classifier",
            "version": "v6",
            "labels": ["negative", "neutral", "positive"],
            "metadata": {
                "quality": 0.95,
                "latency_ms": 20,
                "latency_jitter_ms": 2,
                "transient_error_rate": 0,
                "permanent_error_rate": 0,
            },
        },
    )
    assert model_response.status_code == 201

    evaluation_response = client.post(
        "/evaluations",
        json={
            "dataset_version_id": dataset_response.json()["id"],
            "model_version_id": model_response.json()["id"],
            "records": [
                {
                    "external_id": "r-1",
                    "text": "The checkout flow is fast, accurate, and reliable.",
                    "label": "positive",
                },
                {
                    "external_id": "r-2",
                    "text": "The update is normal with expected baseline behavior.",
                    "label": "neutral",
                },
                {
                    "external_id": "r-3",
                    "text": "The payment step is broken and slow with an error.",
                    "label": "negative",
                },
            ],
        },
    )
    assert evaluation_response.status_code == 200
    payload = evaluation_response.json()
    assert payload["record_count"] == 3
    assert payload["error_rate"] == 0
    assert payload["p95_latency_ms"] > 0

    compare_response = client.get("/evaluations/compare")
    assert compare_response.status_code == 200
    assert len(compare_response.json()) == 1

    gate_response = client.post(
        "/release-gates/check",
        json={"baseline_run_id": payload["run_id"], "candidate_run_id": payload["run_id"]},
    )
    assert gate_response.status_code == 200
    assert gate_response.json()["passed"] is True


def test_api_rejects_unknown_labels(client):
    dataset_response = client.post(
        "/datasets",
        json={"name": "support", "version": "test", "labels": ["negative", "positive"]},
    )
    model_response = client.post(
        "/models",
        json={"name": "sentiment-classifier", "version": "v6", "labels": ["negative", "positive"]},
    )

    response = client.post(
        "/evaluations",
        json={
            "dataset_version_id": dataset_response.json()["id"],
            "model_version_id": model_response.json()["id"],
            "records": [{"external_id": "r-1", "text": "average update", "label": "neutral"}],
        },
    )
    assert response.status_code == 422
