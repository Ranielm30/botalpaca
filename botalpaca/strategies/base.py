"""Strategy base contract.

Every strategy is a pure function from a :class:`TechnicalSnapshot` to a
:class:`StrategySignal` (or ``None`` when its setup is not present).

The contract deliberately returns ``None`` instead of a weak signal: the
confluence engine decides quality, and a strategy that fires on everything
would pollute it.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from botalpaca.config import StrategySettings
from botalpaca.domain import (
    SetupType,
    SignalDirection,
    StrategySignal,
    TechnicalSnapshot,
)

__all__ = ["BaseStrategy", "TradeLevels", "levels_from_atr", "make_fingerprint"]


@dataclass(frozen=True, slots=True)
class TradeLevels:
    """Price levels for a setup: entry, protective stop, profit target(s)."""

    direction: SignalDirection
    entry: float
    stop: float
    targets: tuple[float, ...]

    @property
    def primary_target(self) -> float:
        return self.targets[0] if self.targets else self.entry

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_per_share(self) -> float:
        return abs(self.primary_target - self.entry)

    @property
    def rr(self) -> float:
        risk = self.risk_per_share
        if risk <= 0:
            return 0.0
        return self.reward_per_share / risk


def levels_from_atr(
    direction: SignalDirection,
    entry: float,
    atr: float,
    *,
    stop_multiplier: float,
    target_multiplier: float,
) -> TradeLevels:
    """Build ATR-based levels. Raises when ATR is unusable.

    Rejecting a non-positive ATR here is deliberate: a zero ATR would produce
    a zero-width stop, which is the single most dangerous input a bracket
    order can receive.
    """
    if atr <= 0:
        raise ValueError("ATR must be positive to derive stop and target levels")
    stop_distance = atr * stop_multiplier
    if direction is SignalDirection.LONG:
        stop = entry - stop_distance
        target = entry + atr * target_multiplier
    else:
        stop = entry + stop_distance
        target = entry - atr * target_multiplier
    return TradeLevels(
        direction=direction, entry=entry, stop=stop, targets=(float(target),)
    )


def make_fingerprint(
    symbol: str, strategy: SetupType, direction: SignalDirection, timeframe: str
) -> str:
    """Stable dedup key for a symbol+strategy+direction+timeframe.

    Deliberately excludes price: the same setup re-detected at a slightly
    different price is the same opportunity, not a new one.
    """
    raw = f"{symbol.upper()}|{strategy.value}|{direction.value}|{timeframe}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]  # noqa: S324


class BaseStrategy(ABC):
    """Interface every strategy implements."""

    #: Identity
    setup: SetupType
    #: Human label used in Telegram output.
    display_name: str = ""
    #: Directional bias; ``None`` means the strategy reads both sides.
    bias: SignalDirection | None = None
    #: Whether the strategy needs a higher-timeframe confirmation to fire.
    requires_mtf: bool = False

    def __init__(self, settings: StrategySettings | None = None) -> None:
        self.settings = settings or StrategySettings()

    @abstractmethod
    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        """Return a signal when the setup is present, else ``None``."""

    # -- helpers shared by every strategy -----------------------------------

    def _build_signal(
        self,
        snapshot: TechnicalSnapshot,
        *,
        direction: SignalDirection,
        score: float,
        levels: TradeLevels,
        reasons: list[str],
        confluences: list[str],
        invalidation: str,
        extra: dict[str, object] | None = None,
    ) -> StrategySignal:
        return StrategySignal(
            strategy=self.setup,
            direction=direction,
            score=float(max(0.0, min(100.0, score))),
            entry=levels.entry,
            stop=levels.stop,
            targets=list(levels.targets),
            rr=levels.rr,
            reasons=reasons,
            confluences=confluences,
            invalidation=invalidation,
            timeframe=snapshot.timeframe,
            extra=extra or {},
        )
