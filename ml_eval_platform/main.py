from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from ml_eval_platform.config import get_settings
from ml_eval_platform.database import get_db, init_db
from ml_eval_platform.logging_config import configure_logging
from ml_eval_platform.models import DatasetVersion, EvaluationRun, ModelVersion
from ml_eval_platform.schemas import (
    ComparisonRow,
    DatasetVersionCreate,
    DatasetVersionRead,
    EvaluationRequest,
    EvaluationSummary,
    ModelVersionCreate,
    ModelVersionRead,
    ReleaseGateRequest,
    ReleaseGateResult,
)
from ml_eval_platform.services.demo_data import run_demo_batch
from ml_eval_platform.services.evaluator import evaluate_records
from ml_eval_platform.services.gates import compare_runs
from ml_eval_platform.services.serializers import run_to_comparison_row, run_to_summary

settings = get_settings()
configure_logging(settings.log_level)

DbSession = Annotated[Session, Depends(get_db)]
EvaluationLimit = Annotated[int, Query(ge=1, le=500)]
DemoRecordsPerDataset = Annotated[int, Query(ge=1, le=10_000)]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if settings.auto_create_tables:
        init_db()
    yield


app = FastAPI(
    title="ML Evaluation and Batch-Inference Platform",
    version="0.1.0",
    description="Batch NLP model evaluation service with metrics, failure logs, and release gates.",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post(
    "/datasets",
    response_model=DatasetVersionRead,
    status_code=status.HTTP_201_CREATED,
    tags=["catalog"],
)
def create_dataset(payload: DatasetVersionCreate, db: DbSession) -> DatasetVersion:
    existing = db.scalar(
        select(DatasetVersion).where(
            DatasetVersion.name == payload.name,
            DatasetVersion.version == payload.version,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="dataset version already exists"
        )
    dataset = DatasetVersion(**payload.model_dump())
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@app.get("/datasets", response_model=list[DatasetVersionRead], tags=["catalog"])
def list_datasets(db: DbSession) -> list[DatasetVersion]:
    return list(
        db.scalars(select(DatasetVersion).order_by(DatasetVersion.name, DatasetVersion.version))
    )


@app.post(
    "/models",
    response_model=ModelVersionRead,
    status_code=status.HTTP_201_CREATED,
    tags=["catalog"],
)
def create_model(payload: ModelVersionCreate, db: DbSession) -> ModelVersion:
    existing = db.scalar(
        select(ModelVersion).where(
            ModelVersion.name == payload.name,
            ModelVersion.version == payload.version,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="model version already exists"
        )

    data = payload.model_dump()
    metadata = data.pop("metadata")
    model = ModelVersion(**data, metadata_json=metadata)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@app.get("/models", response_model=list[ModelVersionRead], tags=["catalog"])
def list_models(db: DbSession) -> list[ModelVersion]:
    return list(db.scalars(select(ModelVersion).order_by(ModelVersion.name, ModelVersion.version)))


@app.post("/evaluations", response_model=EvaluationSummary, tags=["evaluation"])
def create_evaluation(payload: EvaluationRequest, db: DbSession) -> EvaluationSummary:
    dataset = db.get(DatasetVersion, payload.dataset_version_id)
    model = db.get(ModelVersion, payload.model_version_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="dataset version not found"
        )
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="model version not found")

    try:
        result = evaluate_records(
            session=db,
            dataset_version=dataset,
            model_version=model,
            records=payload.records,
            run_config=payload.run_config,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return run_to_summary(result.run)


@app.get("/evaluations", response_model=list[EvaluationSummary], tags=["evaluation"])
def list_evaluations(
    db: DbSession,
    limit: EvaluationLimit = 50,
) -> list[EvaluationSummary]:
    statement = (
        select(EvaluationRun)
        .options(joinedload(EvaluationRun.dataset_version), joinedload(EvaluationRun.model_version))
        .order_by(EvaluationRun.started_at.desc())
        .limit(limit)
    )
    return [run_to_summary(run) for run in db.scalars(statement)]


@app.get("/evaluations/compare", response_model=list[ComparisonRow], tags=["evaluation"])
def compare_evaluations(
    db: DbSession,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
) -> list[ComparisonRow]:
    statement = (
        select(EvaluationRun)
        .join(EvaluationRun.dataset_version)
        .join(EvaluationRun.model_version)
        .options(joinedload(EvaluationRun.dataset_version), joinedload(EvaluationRun.model_version))
    )
    if dataset_name:
        statement = statement.where(DatasetVersion.name == dataset_name)
    if dataset_version:
        statement = statement.where(DatasetVersion.version == dataset_version)
    statement = statement.order_by(
        DatasetVersion.name,
        DatasetVersion.version,
        ModelVersion.name,
        ModelVersion.version,
        EvaluationRun.completed_at.desc(),
    )
    return [run_to_comparison_row(run) for run in db.scalars(statement)]


@app.get("/evaluations/{run_id}", response_model=EvaluationSummary, tags=["evaluation"])
def get_evaluation(run_id: str, db: DbSession) -> EvaluationSummary:
    run = _get_run(db, run_id)
    return run_to_summary(run)


@app.post("/release-gates/check", response_model=ReleaseGateResult, tags=["release-gates"])
def check_release_gate(payload: ReleaseGateRequest, db: DbSession) -> ReleaseGateResult:
    baseline = _get_run(db, payload.baseline_run_id)
    candidate = _get_run(db, payload.candidate_run_id)
    outcome = compare_runs(
        baseline=baseline,
        candidate=candidate,
        max_accuracy_drop=payload.max_accuracy_drop,
        max_latency_increase=payload.max_latency_increase,
    )
    return ReleaseGateResult(
        passed=outcome.passed,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        accuracy_delta=outcome.accuracy_delta,
        latency_delta_pct=outcome.latency_delta_pct,
        reasons=outcome.reasons,
    )


@app.post("/demo/run", response_model=list[EvaluationSummary], tags=["demo"])
def run_demo(
    db: DbSession,
    records_per_dataset: DemoRecordsPerDataset = 3_200,
) -> list[EvaluationSummary]:
    runs = run_demo_batch(db, records_per_dataset=records_per_dataset)
    return [run_to_summary(run) for run in runs]


def _get_run(db: Session, run_id: str) -> EvaluationRun:
    run = db.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.run_id == run_id)
        .options(joinedload(EvaluationRun.dataset_version), joinedload(EvaluationRun.model_version))
    )
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="evaluation run not found"
        )
    return run
