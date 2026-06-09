from fastapi import FastAPI

from ml_eval_platform.config import get_settings
from ml_eval_platform.database import init_db
from ml_eval_platform.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="ML Evaluation and Batch-Inference Platform",
    version="0.1.0",
    description="Batch NLP model evaluation service with metrics, failure logs, and release gates.",
)


@app.on_event("startup")
def startup() -> None:
    if settings.auto_create_tables:
        init_db()


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}

