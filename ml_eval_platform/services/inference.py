import hashlib
import re
from dataclasses import dataclass
from typing import Any

import torch

from ml_eval_platform.models import ModelVersion
from ml_eval_platform.schemas import TextRecord

STANDARD_LABELS = ("negative", "neutral", "positive")

POSITIVE_TERMS = {
    "accurate",
    "amazing",
    "awesome",
    "benefit",
    "clear",
    "delight",
    "excellent",
    "fast",
    "great",
    "happy",
    "improved",
    "love",
    "positive",
    "reliable",
    "resolved",
    "smooth",
    "strong",
    "success",
    "useful",
    "win",
}

NEGATIVE_TERMS = {
    "angry",
    "bad",
    "blocked",
    "broken",
    "bug",
    "confusing",
    "delay",
    "error",
    "failed",
    "frustrated",
    "hate",
    "issue",
    "negative",
    "poor",
    "problem",
    "regression",
    "slow",
    "unstable",
    "worse",
    "wrong",
}

NEUTRAL_TERMS = {
    "average",
    "baseline",
    "expected",
    "fine",
    "information",
    "mixed",
    "normal",
    "notice",
    "okay",
    "pending",
    "scheduled",
    "standard",
    "status",
    "typical",
    "update",
}

NEGATIONS = {"not", "never", "no", "hardly", "without"}

DEFAULT_MODEL_PROFILES: dict[str, dict[str, float]] = {
    "v1": {
        "quality": 0.62,
        "latency_ms": 34.0,
        "latency_jitter_ms": 8.5,
        "transient_error_rate": 0.04,
        "permanent_error_rate": 0.012,
    },
    "v2": {
        "quality": 0.70,
        "latency_ms": 32.0,
        "latency_jitter_ms": 7.5,
        "transient_error_rate": 0.032,
        "permanent_error_rate": 0.009,
    },
    "v3": {
        "quality": 0.77,
        "latency_ms": 30.0,
        "latency_jitter_ms": 6.5,
        "transient_error_rate": 0.026,
        "permanent_error_rate": 0.007,
    },
    "v4": {
        "quality": 0.84,
        "latency_ms": 30.5,
        "latency_jitter_ms": 5.5,
        "transient_error_rate": 0.02,
        "permanent_error_rate": 0.005,
    },
    "v5": {
        "quality": 0.89,
        "latency_ms": 27.0,
        "latency_jitter_ms": 5.0,
        "transient_error_rate": 0.016,
        "permanent_error_rate": 0.0035,
    },
    "v6": {
        "quality": 0.93,
        "latency_ms": 25.0,
        "latency_jitter_ms": 4.0,
        "transient_error_rate": 0.012,
        "permanent_error_rate": 0.002,
    },
}


class TransientInferenceError(RuntimeError):
    pass


class PermanentInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    latency_ms: float


class TorchTextClassifier(torch.nn.Module):
    def __init__(self, model_version: ModelVersion):
        super().__init__()
        self.model_version = model_version
        self.labels = tuple(model_version.labels)
        self.profile = _resolve_profile(model_version.version, model_version.metadata_json)
        self.linear = torch.nn.Linear(9, len(self.labels))
        self._configure_weights()
        self.eval()

    def _configure_weights(self) -> None:
        generator = torch.Generator().manual_seed(int(self.model_version.seed))
        if self.labels == STANDARD_LABELS:
            base_weights = torch.tensor(
                [
                    [-0.6, 1.8, -0.2, -1.3, 0.0, 0.1, 0.0, 0.7, -0.1],
                    [-0.2, -0.2, 1.4, 0.0, 0.1, 0.4, -0.2, 0.0, 0.3],
                    [1.8, -0.7, -0.2, 1.3, 0.0, 0.0, 0.2, -0.5, -0.1],
                ],
                dtype=torch.float32,
            )
            base_bias = torch.tensor([0.0, 0.15, 0.0], dtype=torch.float32)
        else:
            base_weights = torch.randn(
                (len(self.labels), 9), generator=generator, dtype=torch.float32
            )
            base_bias = torch.zeros(len(self.labels), dtype=torch.float32)

        quality = float(self.profile["quality"])
        noise_scale = max(0.02, (1.0 - quality) * 1.8)
        noise = (
            torch.randn(base_weights.shape, generator=generator, dtype=torch.float32) * noise_scale
        )

        with torch.no_grad():
            self.linear.weight.copy_(base_weights * quality + noise)
            self.linear.bias.copy_(base_bias)

    def predict(self, record: TextRecord, attempt: int = 1) -> Prediction:
        self._raise_for_configured_failures(record, attempt)
        features = build_features(record.text)
        with torch.no_grad():
            logits = self.linear(features)
            probabilities = torch.softmax(logits, dim=0)
            predicted_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_index].item())

        return Prediction(
            label=self.labels[predicted_index],
            confidence=round(confidence, 6),
            latency_ms=self._synthetic_latency(record),
        )

    def _raise_for_configured_failures(self, record: TextRecord, attempt: int) -> None:
        permanent_rate = float(self.profile["permanent_error_rate"])
        transient_rate = float(self.profile["transient_error_rate"])
        if (
            _stable_score("permanent", self.model_version.version, record.external_id)
            < permanent_rate
        ):
            raise PermanentInferenceError("model returned a non-retryable inference error")
        if (
            attempt == 1
            and _stable_score("transient", self.model_version.version, record.external_id)
            < transient_rate
        ):
            raise TransientInferenceError("transient model-serving timeout")

    def _synthetic_latency(self, record: TextRecord) -> float:
        base_latency = float(self.profile["latency_ms"])
        jitter = float(self.profile["latency_jitter_ms"])
        latency = base_latency + jitter * _stable_score(
            "latency", self.model_version.version, record.external_id
        )
        return round(latency, 4)


def build_features(text: str) -> torch.Tensor:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    token_count = max(1, len(tokens))
    positive_hits = sum(token in POSITIVE_TERMS for token in tokens)
    negative_hits = sum(token in NEGATIVE_TERMS for token in tokens)
    neutral_hits = sum(token in NEUTRAL_TERMS for token in tokens)
    negation_hits = sum(token in NEGATIONS for token in tokens)
    polarity = positive_hits - negative_hits
    punctuation_intensity = min(3, text.count("!") + text.count("?")) / 3
    question_signal = 1.0 if "?" in text else 0.0
    length_signal = min(1.0, token_count / 80)
    caps_signal = min(1.0, sum(char.isupper() for char in text) / max(1, len(text)))

    return torch.tensor(
        [
            positive_hits / token_count,
            negative_hits / token_count,
            neutral_hits / token_count,
            polarity / token_count,
            punctuation_intensity,
            question_signal,
            length_signal,
            negation_hits / token_count,
            caps_signal,
        ],
        dtype=torch.float32,
    )


def _resolve_profile(version: str, metadata: dict[str, Any]) -> dict[str, float]:
    default_profile = DEFAULT_MODEL_PROFILES.get(version, DEFAULT_MODEL_PROFILES["v6"])
    profile = {**default_profile}
    for key in default_profile:
        if key in metadata:
            profile[key] = float(metadata[key])
    return profile


def _stable_score(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)
