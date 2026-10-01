"""Alembic bootstrap invoked at process start.

``Database(create_all=True)`` is convenient for tests and local runs, but a
deployed instance (Fly.io) must go through real migrations so schema changes
are versioned and auditable. ``run_migrations()`` is the programmatic entry
point used by :meth:`botalpaca.db.session.Database.start`.
"""

from __future__ import annotations

from pathlib import Path

from botalpaca.config import get_logger

logger = get_logger(__name__)

_MIGRATIONS_DIRNAME = "migrations"


def _project_root() -> Path:
    """Directory that contains ``migrations/``.

    Works both from a source checkout (``<root>/migrations``) and from an
    installed package location.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MIGRATIONS_DIRNAME).is_dir():
            return parent
    return here.parents[2]


def alembic_config(url: str | None = None):
    """Build an Alembic ``Config`` with the runtime database URL injected."""
    from alembic.config import Config

    root = _project_root()
    ini = root / "alembic.ini"
    config = Config(str(ini)) if ini.is_file() else Config()
    config.set_main_option("script_location", str(root / _MIGRATIONS_DIRNAME))
    config.set_main_option("prepend_sys_path", str(root))
    if url:
        config.set_main_option("sqlalchemy.url", url)
    # Silence alembic's own fileConfig so structlog keeps ownership of logging.
    return config


def run_migrations(url: str | None = None, revision: str = "head") -> str:
    """Upgrade the database to ``revision`` (default ``head``).

    Returns the resulting revision string. Safe to call repeatedly: Alembic
    detects the already-applied revision and does nothing.
    """
    from alembic import command

    if url is None:
        from botalpaca.config import get_settings

        url = get_settings().database_url
    # env.py rewrites the async SQLite driver to the sync one.
    config = alembic_config(url)
    logger.info("migrations.running", revision=revision)
    command.upgrade(config, revision)
    current = command.current(config, verbose=False)
    logger.info("migrations.applied", revision=getattr(current, "revision", None) or revision)
    return getattr(current, "revision", None) or revision


def current_revision(url: str | None = None) -> str | None:
    """Return the revision currently stamped in the database, if reachable."""
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    if url is None:
        from botalpaca.config import get_settings

        url = get_settings().database_url
    sync_url = url.replace("sqlite+aiosqlite:///", "sqlite:///")
    engine = create_engine(sync_url)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


__all__ = ["alembic_config", "current_revision", "run_migrations"]
