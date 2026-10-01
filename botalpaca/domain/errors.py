"""Domain-level exceptions.

Every error the system can raise intentionally lives here so callers can
distinguish expected control flow (risk blocks, confirmation needed) from
genuine bugs.
"""

from __future__ import annotations


class BotalpacaError(Exception):
    """Base class for all intentional errors raised by the system."""


class ConfigurationError(BotalpacaError):
    """Invalid or missing configuration detected at startup."""


class SecretNotConfiguredError(ConfigurationError):
    """Required credentials for an environment are missing."""


class AuthorizationError(BotalpacaError):
    """Telegram user is not on the allowlist."""


class ValidationError(BotalpacaError):
    """User-supplied input failed validation."""


class EnvironmentMismatchError(BotalpacaError):
    """An order was attempted against a non-active trading environment.

    This is the security barrier. It must never be bypassed.
    """


class KillSwitchError(BotalpacaError):
    """The kill switch is engaged; no new orders may be submitted."""


class RiskRejectedError(BotalpacaError):
    """The risk engine blocked an otherwise valid order."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class ConfirmationRequiredError(BotalpacaError):
    """The operation needs an explicit confirmation step."""


class DuplicateOrderError(BotalpacaError):
    """An equivalent order is already in flight or working."""


class OrderRejectedByBrokerError(BotalpacaError):
    """Alpaca refused the order. The message is sanitized of secrets."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class UnsupportedOrderShapeError(BotalpacaError):
    """The requested combination of order type/class is not supported by Alpaca.

    Raised instead of silently degrading a protection, so the Position
    Protection Manager can perform an explicit safe transition.
    """


class DataQualityError(BotalpacaError):
    """Market data was insufficient or malformed for a computation."""


class ReconciliationError(BotalpacaError):
    """Persistent state and Alpaca could not be reconciled."""


class NotFoundError(BotalpacaError):
    """A requested entity does not exist."""


__all__ = [
    "AuthorizationError",
    "BotalpacaError",
    "ConfigurationError",
    "ConfirmationRequiredError",
    "DataQualityError",
    "DuplicateOrderError",
    "EnvironmentMismatchError",
    "KillSwitchError",
    "NotFoundError",
    "OrderRejectedByBrokerError",
    "ReconciliationError",
    "RiskRejectedError",
    "SecretNotConfiguredError",
    "UnsupportedOrderShapeError",
    "ValidationError",
]
