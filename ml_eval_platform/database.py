from collections.abc import Generator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ml_eval_platform.config import get_settings


class Base(DeclarativeBase):
    pass


def _connect_args(database_url: str) -> dict[str, object]:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def _ensure_sqlite_parent(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    database = make_url(database_url).database
    if not database or database == ":memory:":
        return
    Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    _ensure_sqlite_parent(settings.database_url)
    return create_engine(
        settings.database_url,
        connect_args=_connect_args(settings.database_url),
        pool_pre_ping=not settings.database_url.startswith("sqlite"),
        future=True,
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False
    )


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    from ml_eval_platform import models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())
