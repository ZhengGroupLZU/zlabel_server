"""Engine / session plumbing for v2.

A single ``Database`` instance is attached to ``app.state.db``; tests build their
own with an in-memory URL and override the ``get_session`` dependency, so no
module-level global state has to be monkeypatched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import Request
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """Turn SQLite's foreign-key enforcement on for a connection.

    SQLite ships with the ``foreign_keys`` pragma **off**, which silently turns
    every ``ondelete="CASCADE"``/``"SET NULL"`` in the models into a no-op (and
    leaves orphan rows behind a delete). Alembic builds its own engine and keeps
    the pragma off on purpose: batch migrations recreate tables and enforced
    constraints would get in the way.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


class Database:
    def __init__(self, url: str, *, echo: bool = False) -> None:
        kwargs: dict = {"echo": echo, "future": True}
        is_sqlite = url.startswith("sqlite")
        if is_sqlite:
            # TestClient / worker threads share one connection.
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                # one shared in-memory database for every session
                kwargs["poolclass"] = StaticPool
        self.url = url
        self.engine: Engine = create_engine(url, **kwargs)
        if is_sqlite:
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        self.session_maker = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Transactional scope: commit on success, rollback on error."""
        session = self.session_maker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all(self) -> None:
        """Create every table (tests / first boot convenience)."""
        from app.db import models  # noqa: F401  (register the mappers)

        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()


def get_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency: one session per request, committed by the scope."""
    db: Database = request.app.state.db
    with db.session_scope() as session:
        yield session
