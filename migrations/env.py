"""Alembic environment for botalpaca.

The database URL is always taken from ``Settings`` (and therefore from
environment variables) so credentials and Fly.io volume paths never end up
committed in ``alembic.ini``.

Alembic is a synchronous tool, so the async ``aiosqlite`` driver is rewritten
to the plain ``sqlite3`` driver and driven through a regular synchronous engine.
"""

from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Connection

from botalpaca.config import configure_logging, get_settings
from botalpaca.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

configure_logging()

target_metadata = Base.metadata


def database_url() -> str:
    """Runtime DSN, with the async SQLite driver swapped for the sync one."""
    url = config.get_main_option("sqlalchemy.url") or get_settings().database_url
    if url.startswith("sqlite+aiosqlite:///"):
        url = "sqlite:///" + url[len("sqlite+aiosqlite:///") :]
    if url.startswith("sqlite:///"):
        raw = url[len("sqlite:///") :]
        if raw and raw != ":memory:":
            # On Fly.io the file lives on a mounted volume that may not exist yet.
            Path(raw).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return url


def run_migrations_offline() -> None:
    """Emit SQL without a DB connection (``alembic upgrade head --sql``)."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connect_args: dict[str, object] = {}
    if database_url().startswith("sqlite"):
        connect_args["timeout"] = 30
    engine = create_engine(database_url(), poolclass=pool.NullPool, connect_args=connect_args)
    try:
        with engine.connect() as connection:
            _run_sync(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
