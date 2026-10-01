"""Structured logging with secret redaction.

Every log line is JSON in production and human-readable locally. Secrets are
scrubbed at the processor level so a careless ``log.info(api_key=...)`` still
cannot leak a credential.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_REDACTED = "***REDACTED***"
# Matched as substrings, lowercased, so per-environment names such as
# ``ALPACA_PAPER_SECRET_KEY`` are caught without enumerating every variant.
_SECRET_KEYS = {
    "api_key",
    "secret_key",
    "apikey",
    "apisecret",
    "token",
    "telegram_bot_token",
    "password",
    "passwd",
    "authorization",
    "secret",
    "access_token",
    "refresh_token",
    "alpaca_paper_api_key",
    "alpaca_live_api_key",
    "client_secret",
    "private_key",
    "signature",
}


def _is_secret_key(key: str) -> bool:
    """True when ``key`` names a credential, whatever its case or prefix."""
    lowered = key.lower()
    if lowered in _SECRET_KEYS:
        return True
    return any(
        lowered.endswith(known) or f"_{known}" in lowered for known in _SECRET_KEYS
    )


class SecretsFilter:
    """Structlog processor that masks secret-looking keys and values."""

    def __call__(
        self, logger: Any, name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        return self._scrub(event_dict)

    @classmethod
    def _scrub(cls, value: Any, key_hint: str | None = None) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                if isinstance(k, str) and _is_secret_key(k):
                    out[k] = _REDACTED
                else:
                    out[k] = cls._scrub(v, k)
            return out
        if isinstance(value, (list, tuple)):
            scrubbed = [cls._scrub(v, key_hint) for v in value]
            return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
        if key_hint and _is_secret_key(key_hint) and value:
            return _REDACTED
        return value


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Idempotently configure stdlib logging + structlog."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "apscheduler", "telegram.ext.Updater"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        SecretsFilter(),
    ]

    if json_output:
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer(sort_keys=True))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    # The stdlib BoundLogger/LoggerFactory pair is required because the processor
    # chain above uses ``structlog.stdlib.add_logger_name`` / ``add_log_level``,
    # which read ``logger.name``. A bare PrintLoggerFactory has no such attribute.
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)


__all__ = ["SecretsFilter", "configure_logging", "get_logger"]
