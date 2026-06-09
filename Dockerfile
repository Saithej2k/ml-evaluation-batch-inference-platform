FROM python:3.12-slim

ARG INSTALL_DEV=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ml_eval_platform ./ml_eval_platform
COPY scripts ./scripts
COPY benchmarks ./benchmarks

RUN python -m pip install --upgrade pip \
    && if [ "$INSTALL_DEV" = "true" ]; then pip install ".[dev]"; else pip install .; fi

EXPOSE 8000

CMD ["uvicorn", "ml_eval_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]

