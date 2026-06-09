import hashlib
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ml_eval_platform.config import Settings
from ml_eval_platform.models import DatasetVersion, EvaluationRun, ModelVersion
from ml_eval_platform.schemas import TextRecord
from ml_eval_platform.services.evaluator import evaluate_records
from ml_eval_platform.services.inference import DEFAULT_MODEL_PROFILES, STANDARD_LABELS

DEMO_DATASETS: tuple[tuple[str, str, str], ...] = (
    ("product-reviews", "2024-01", "checkout and onboarding reviews"),
    ("support-tickets", "2024-02", "customer support messages"),
    ("social-comments", "2024-03", "short social media posts"),
    ("news-feedback", "2024-04", "reader feedback on news summaries"),
)

DEMO_MODEL_NAME = "sentiment-classifier"
DEMO_MODEL_VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6")
DEMO_LABELS = list(STANDARD_LABELS)

TEMPLATES: dict[str, tuple[str, ...]] = {
    "negative": (
        "The {context} flow is slow and confusing with a repeated error.",
        "I found a bad regression in the {context} path and it blocked the task.",
        "The update made the {context} result worse and the issue is still unresolved.",
        "This {context} experience feels unstable, broken, and frustrating.",
    ),
    "neutral": (
        "The {context} status is normal and the update is scheduled.",
        "This is an average {context} result with expected baseline behavior.",
        "The {context} notice contains standard information and a pending update.",
        "The {context} outcome is okay with mixed but typical feedback.",
    ),
    "positive": (
        "The {context} flow is fast, accurate, and reliable.",
        "I love the improved {context} experience and the result is excellent.",
        "The {context} update resolved the issue with a smooth success.",
        "This {context} change is useful, clear, and a strong win.",
    ),
}


def run_demo_batch(
    session: Session,
    records_per_dataset: int = 3_200,
    settings: Settings | None = None,
) -> list[EvaluationRun]:
    datasets, models = ensure_demo_catalog(session, records_per_dataset)
    runs: list[EvaluationRun] = []
    for dataset in datasets:
        records = build_demo_records(dataset.name, dataset.version, records_per_dataset)
        for model in models:
            result = evaluate_records(
                session=session,
                dataset_version=dataset,
                model_version=model,
                records=records,
                run_config={
                    "source": "demo",
                    "records_per_dataset": records_per_dataset,
                    "dataset_count": len(datasets),
                    "model_count": len(models),
                },
                settings=settings,
            )
            runs.append(result.run)
    return runs


def ensure_demo_catalog(
    session: Session,
    records_per_dataset: int,
) -> tuple[list[DatasetVersion], list[ModelVersion]]:
    datasets = [
        _upsert_dataset(session, name, version, records_per_dataset)
        for name, version, _context in DEMO_DATASETS
    ]
    models = [_upsert_model(session, version) for version in DEMO_MODEL_VERSIONS]
    session.commit()
    return datasets, models


def build_demo_records(dataset_name: str, dataset_version: str, count: int) -> list[TextRecord]:
    context = _context_for_dataset(dataset_name)
    records: list[TextRecord] = []
    labels = tuple(TEMPLATES.keys())
    for index in range(count):
        label = labels[_stable_int(dataset_name, dataset_version, index, modulo=len(labels))]
        template_index = _stable_int(dataset_version, label, index, modulo=len(TEMPLATES[label]))
        text = TEMPLATES[label][template_index].format(context=context)
        suffix = _noise_suffix(dataset_name, index)
        records.append(
            TextRecord(
                external_id=f"{dataset_name}-{dataset_version}-{index:06d}",
                text=f"{text} {suffix}",
                label=label,
            )
        )
    return records


def _upsert_dataset(session: Session, name: str, version: str, size: int) -> DatasetVersion:
    dataset = session.scalar(
        select(DatasetVersion).where(DatasetVersion.name == name, DatasetVersion.version == version)
    )
    if dataset is None:
        dataset = DatasetVersion(name=name, version=version, labels=DEMO_LABELS, size=size)
        session.add(dataset)
        session.flush()
    else:
        dataset.labels = DEMO_LABELS
        dataset.size = size
    return dataset


def _upsert_model(session: Session, version: str) -> ModelVersion:
    model = session.scalar(
        select(ModelVersion).where(
            ModelVersion.name == DEMO_MODEL_NAME, ModelVersion.version == version
        )
    )
    profile = DEFAULT_MODEL_PROFILES[version]
    metadata = {
        "quality": profile["quality"],
        "latency_ms": profile["latency_ms"],
        "latency_jitter_ms": profile["latency_jitter_ms"],
        "transient_error_rate": profile["transient_error_rate"],
        "permanent_error_rate": profile["permanent_error_rate"],
    }
    if model is None:
        model = ModelVersion(
            name=DEMO_MODEL_NAME,
            version=version,
            labels=DEMO_LABELS,
            framework="pytorch",
            seed=17 + int(version.removeprefix("v")),
            metadata_json=metadata,
        )
        session.add(model)
        session.flush()
    else:
        model.labels = DEMO_LABELS
        model.framework = "pytorch"
        model.metadata_json = metadata
    return model


def _context_for_dataset(dataset_name: str) -> str:
    for name, _version, context in DEMO_DATASETS:
        if name == dataset_name:
            return context
    return dataset_name.replace("-", " ")


def _noise_suffix(dataset_name: str, index: int) -> str:
    variants = (
        "The batch sample includes short text.",
        "The label was reviewed against the versioned schema.",
        "The message was collected from the current evaluation split.",
        "The request follows the accepted input schema.",
    )
    return variants[_stable_int(dataset_name, index, modulo=len(variants))]


def _stable_int(*parts: object, modulo: int) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def total_predictions(
    records_per_dataset: int, datasets: Iterable[object], models: Iterable[object]
) -> int:
    return records_per_dataset * len(tuple(datasets)) * len(tuple(models))
