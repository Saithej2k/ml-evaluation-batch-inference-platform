from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ml_eval_platform.database import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_dataset_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    runs: Mapped[list["EvaluationRun"]] = relationship(back_populates="dataset_version")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    framework: Mapped[str] = mapped_column(String(40), nullable=False, default="pytorch")
    seed: Mapped[int] = mapped_column(Integer, nullable=False, default=13)
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    runs: Mapped[list["EvaluationRun"]] = relationship(back_populates="model_version")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_run_dataset_model", "dataset_version_id", "model_version_id"),
        Index("ix_run_completed_at", "completed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="running")
    record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    p95_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    error_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    run_config: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset_version: Mapped[DatasetVersion] = relationship(back_populates="runs")
    model_version: Mapped[ModelVersion] = relationship(back_populates="runs")
    predictions: Mapped[list["PredictionResult"]] = relationship(
        back_populates="evaluation_run",
        cascade="all, delete-orphan",
    )
    failure_logs: Mapped[list["FailureLog"]] = relationship(
        back_populates="evaluation_run",
        cascade="all, delete-orphan",
    )


class PredictionResult(Base):
    __tablename__ = "prediction_results"
    __table_args__ = (Index("ix_prediction_run_external_id", "evaluation_run_id", "external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(160), nullable=False)
    expected_label: Mapped[str] = mapped_column(String(80), nullable=False)
    predicted_label: Mapped[str | None] = mapped_column(String(80), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    succeeded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    evaluation_run: Mapped[EvaluationRun] = relationship(back_populates="predictions")


class FailureLog(Base):
    __tablename__ = "failure_logs"
    __table_args__ = (Index("ix_failure_run_stage", "evaluation_run_id", "stage"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    stage: Mapped[str] = mapped_column(String(80), nullable=False)
    error_type: Mapped[str] = mapped_column(String(120), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    evaluation_run: Mapped[EvaluationRun] = relationship(back_populates="failure_logs")
