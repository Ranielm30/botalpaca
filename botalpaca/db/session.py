"""Database engine, session factory, and migration bootstrap."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from botalpaca.config import get_logger
from botalpaca.db.models import Base

logger = get_logger(__name__)


def _ensure_sqlite_dir(url: str) -> None:
    """SQLite files on Fly.io live on a mounted volume; create the parent."""
    marker = "sqlite+aiosqlite:///"
    if not url.startswith(marker):
        return
    raw = url[len(marker) :]
    if not raw or raw == ":memory:":
        return
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)


class Database:
    """Owns the async engine and hands out sessions.

    Enabled with SQLAlchemy's asyncio driver so SQLite and a future Postgres
    migration both work through the same repository code.
    """

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        create_all: bool = True,
        migrate: bool = False,
    ) -> None:
        self._url = url
        self._echo = echo
        self._create_all = create_all
        self._migrate = migrate
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._lock = asyncio.Lock()

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database.start() must be called before use.")
        return self._engine

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        if self._sessionmaker is None:
            raise RuntimeError("Database.start() must be called before use.")
        return self._sessionmaker

    async def start(self) -> None:
        async with self._lock:
            if self._engine is not None:
                return
            _ensure_sqlite_dir(self._url)
            connect_args: dict[str, object] = {}
            if self._url.startswith("sqlite"):
                # Long scans and reconciliation must not trip "database is locked".
                connect_args["timeout"] = 30
            self._engine = create_async_engine(
                self._url,
                echo=self._echo,
                future=True,
                pool_pre_ping=True,
                connect_args=connect_args,
            )
            self._sessionmaker = async_sessionmaker(
                self._engine, expire_on_commit=False, class_=AsyncSession
            )
            if self._migrate:
                from botalpaca.db.migrate import run_migrations

                await asyncio.to_thread(run_migrations, self._url)
            if self._create_all:
                async with self._engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
            logger.info("database.started", url=_redact_url(self._url))

    async def stop(self) -> None:
        async with self._lock:
            if self._engine is not None:
                await self._engine.dispose()
                self._engine = None
                self._sessionmaker = None
                logger.info("database.stopped")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transactional scope: commit on success, roll back on any exception."""
        async with self.sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def healthcheck(self) -> bool:
        from sqlalchemy import text

        try:
            async with self.sessionmaker() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("database.healthcheck.failed", error=str(exc))
            return False


def _redact_url(url: str) -> str:
    """Strip credentials from a DSN before it reaches the logs."""
    if "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    if "@" in rest:
        rest = rest.split("@", 1)[1]
    return f"{scheme}://{rest}"


_database: Database | None = None


def get_database() -> Database:
    global _database
    if _database is None:
        settings = _settings()
        _database = Database(settings.database_url, echo=settings.database_echo)
    return _database


def set_database(db: Database | None) -> None:
    """Injection point for tests and for the app container."""
    global _database
    _database = db


def _settings():  # avoids a circular import at module load
    from botalpaca.config import get_settings

    return get_settings()


__all__ = ["Database", "get_database", "set_database"]
