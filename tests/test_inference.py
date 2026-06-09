from ml_eval_platform.models import ModelVersion
from ml_eval_platform.schemas import TextRecord
from ml_eval_platform.services.inference import TorchTextClassifier, build_features


def test_torch_classifier_predicts_expected_label_set():
    model = ModelVersion(
        name="sentiment-classifier",
        version="v6",
        labels=["negative", "neutral", "positive"],
        seed=29,
        metadata_json={
            "quality": 0.95,
            "latency_ms": 20,
            "latency_jitter_ms": 2,
            "transient_error_rate": 0,
            "permanent_error_rate": 0,
        },
    )
    classifier = TorchTextClassifier(model)

    prediction = classifier.predict(
        TextRecord(
            external_id="sample",
            text="The batch evaluation result is fast, accurate, and reliable.",
            label="positive",
        )
    )

    assert prediction.label in {"negative", "neutral", "positive"}
    assert prediction.confidence > 0
    assert prediction.latency_ms >= 20


def test_text_features_have_stable_shape():
    features = build_features("The update is useful, clear, and fast.")

    assert tuple(features.shape) == (9,)
