from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetVersionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=80)
    labels: list[str] = Field(min_length=2)
    size: int = Field(default=0, ge=0)
    checksum: str | None = Field(default=None, max_length=128)

    model_config = ConfigDict(extra="forbid")

    @field_validator("labels")
    @classmethod
    def labels_must_be_unique(cls, labels: list[str]) -> list[str]:
        normalized = [label.strip() for label in labels]
        if any(not label for label in normalized):
            raise ValueError("labels cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("labels must be unique")
        return normalized


class DatasetVersionRead(DatasetVersionCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelVersionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=80)
    labels: list[str] = Field(min_length=2)
    framework: str = Field(default="pytorch", max_length=40)
    seed: int = Field(default=13, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("labels")
    @classmethod
    def labels_must_be_unique(cls, labels: list[str]) -> list[str]:
        normalized = [label.strip() for label in labels]
        if any(not label for label in normalized):
            raise ValueError("labels cannot be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("labels must be unique")
        return normalized


class ModelVersionRead(BaseModel):
    id: int
    name: str
    version: str
    labels: list[str]
    framework: str
    seed: int
    metadata: dict[str, Any] = Field(
        validation_alias="metadata_json", serialization_alias="metadata"
    )
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class TextRecord(BaseModel):
    external_id: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=20_000)
    label: str = Field(min_length=1, max_length=80)

    model_config = ConfigDict(extra="forbid")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value


class EvaluationRequest(BaseModel):
    dataset_version_id: int = Field(gt=0)
    model_version_id: int = Field(gt=0)
    records: list[TextRecord] = Field(min_length=1, max_length=100_000)
    run_config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class EvaluationSummary(BaseModel):
    run_id: str
    status: str
    dataset_name: str
    dataset_version: str
    model_name: str
    model_version: str
    record_count: int
    accuracy: float
    p95_latency_ms: float
    error_rate: float
    failure_count: int
    started_at: datetime
    completed_at: datetime | None


class ComparisonRow(BaseModel):
    dataset_name: str
    dataset_version: str
    model_name: str
    model_version: str
    run_id: str
    accuracy: float
    p95_latency_ms: float
    error_rate: float
    failure_count: int
    record_count: int
    completed_at: datetime | None


class ReleaseGateRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    max_accuracy_drop: float = Field(default=0.02, ge=0, le=1)
    max_latency_increase: float = Field(default=0.15, ge=0, le=10)


class ReleaseGateResult(BaseModel):
    passed: bool
    baseline_run_id: str
    candidate_run_id: str
    accuracy_delta: float
    latency_delta_pct: float
    reasons: list[str]
