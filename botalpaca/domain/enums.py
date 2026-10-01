"""Domain enums: the vocabulary shared across every layer.

These are framework-agnostic. Nothing in this module imports SQLAlchemy,
Alpaca, or Telegram, so it can safely be used from the domain, persistence,
and interface layers alike.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum with readable ``str()`` and stable values."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class TradingEnvironment(StrEnum):
    """Fully isolated trading accounts. Never mix state between these."""

    PAPER = "PAPER"
    REAL = "REAL"

    @property
    def is_paper(self) -> bool:
        return self is TradingEnvironment.PAPER

    @property
    def is_real(self) -> bool:
        return self is TradingEnvironment.REAL

    @property
    def other(self) -> TradingEnvironment:
        return TradingEnvironment.REAL if self.is_paper else TradingEnvironment.PAPER

    @property
    def badge(self) -> str:
        return "🟢" if self.is_paper else "🔴"

    @property
    def label(self) -> str:
        """Human-readable name. Never rely on the emoji alone: the environment
        must be legible as text in every message and audit row."""
        return "ALPACA PAPER" if self.is_paper else "ALPACA REAL"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class SignalDirection(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def entry_side(self) -> OrderSide:
        return OrderSide.BUY if self is SignalDirection.LONG else OrderSide.SELL

    @property
    def exit_side(self) -> OrderSide:
        return OrderSide.SELL if self is SignalDirection.LONG else OrderSide.BUY


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"
    IOC = "ioc"
    FOK = "fok"


class OrderClass(StrEnum):
    SIMPLE = "simple"
    BRACKET = "bracket"
    OCO = "oco"
    OTO = "oto"


class Quality(StrEnum):
    """Opportunity quality buckets used for thresholds and display."""

    ALTA = "ALTA"
    MEDIA = "MEDIA"
    BAJA = "BAJA"
    NO_OPERABLE = "NO OPERABLE"

    @property
    def emoji(self) -> str:
        return {
            Quality.ALTA: "🟢",
            Quality.MEDIA: "🟡",
            Quality.BAJA: "🟠",
            Quality.NO_OPERABLE: "⚫",
        }[self]


class ExitReason(StrEnum):
    """Why a position was closed. Persisted for the trade journal."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    BREAK_EVEN = "BREAK_EVEN"
    MOMENTUM_LOSS = "MOMENTUM_LOSS"
    STRUCTURE_INVALIDATION = "STRUCTURE_INVALIDATION"
    TREND_DETERIORATION = "TREND_DETERIORATION"
    VOLUME_SHIFT = "VOLUME_SHIFT"
    OPPOSITE_SIGNAL = "OPPOSITE_SIGNAL"
    TIME_STOP = "TIME_STOP"
    MANUAL = "MANUAL"
    PARTIAL = "PARTIAL"
    BRACKET_CHILD = "BRACKET_CHILD"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    UNKNOWN = "UNKNOWN"


class TradeStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

    @property
    def is_open(self) -> bool:
        return self is TradeStatus.OPEN


class MarketRegime(StrEnum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class ProtectionKind(StrEnum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    NONE = "NONE"


class SetupType(StrEnum):
    TREND_FOLLOWING = "TREND_FOLLOWING"
    EMA_CONTINUATION = "EMA_CONTINUATION"
    MOMENTUM = "MOMENTUM"
    BREAKOUT = "BREAKOUT"
    BREAKOUT_VOLUME = "BREAKOUT_VOLUME"
    PULLBACK = "PULLBACK"
    MEAN_REVERSION = "MEAN_REVERSION"
    SR_BOUNCE = "SR_BOUNCE"
    VWAP_REVERSION = "VWAP_REVERSION"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    RELATIVE_STRENGTH = "RELATIVE_STRENGTH"
    MULTI_TIMEFRAME = "MULTI_TIMEFRAME"


class OrderEnvironmentMismatch(StrEnum):
    """Reason an order was rejected by the environment barrier."""

    NOT_ACTIVE_ENVIRONMENT = "NOT_ACTIVE_ENVIRONMENT"
    MISSING_ENVIRONMENT = "MISSING_ENVIRONMENT"
    KILL_SWITCH = "KILL_SWITCH"


__all__ = [
    "ExitReason",
    "MarketRegime",
    "OrderClass",
    "OrderEnvironmentMismatch",
    "OrderSide",
    "OrderType",
    "ProtectionKind",
    "Quality",
    "SetupType",
    "SignalDirection",
    "StrEnum",
    "TimeInForce",
    "TradeStatus",
    "TradingEnvironment",
]
