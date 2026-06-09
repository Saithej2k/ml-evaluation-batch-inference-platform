import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("AUTO_CREATE_TABLES", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///./test.sqlite")
os.environ.setdefault("STORE_PREDICTIONS", "false")

from ml_eval_platform.database import Base, get_db  # noqa: E402
from ml_eval_platform.main import app  # noqa: E402


@pytest.fixture
def db_session(tmp_path) -> Generator[Session, None, None]:
    database_path = tmp_path / "test.sqlite"
    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    with SessionLocal() as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
