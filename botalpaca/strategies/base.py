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
    StructureState,
    TechnicalSnapshot,
)

__all__ = [
    "BaseStrategy",
    "TradeLevels",
    "levels_from_atr",
    "make_fingerprint",
    "refine_levels_with_structure",
]


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


# How far past its ATR budget a structural stop may reach. Real levels
# sit a little wider than an ATR multiple; without slack the refinement
# would reject the very levels it exists to use.
MAX_STRUCTURE_STOP_BUDGET = 2.0
# Measured results (Cesar Alvarez, Better System Trader) put the profitable
# range for an initial stop at 2-4 ATR, with 2.5-3 the usual pick. Past that,
# the target has to travel far enough to stop filling often.
MAX_STOP_ATR = 3.0
# A stop placed exactly on a swing level gets taken out by the wick that
# tests it. The level is the reference, not the trigger.
STRUCTURE_STOP_BUFFER_ATR = 0.5


def refine_levels_with_structure(
    levels: TradeLevels,
    structure: StructureState | None,
    atr: float,
    *,
    min_rr: float,
    min_stop_atr: float = 0.5,
) -> TradeLevels:
    """Move the levels onto real structure, so the R:R finally means something.

    ``levels_from_atr`` places the stop and the target as multiples of the same
    ATR, so ``(target - entry) / (entry - stop)`` collapses to
    ``target_multiplier / stop_multiplier`` -- a constant, the same for every
    symbol and every day. A risk filter built on that ratio can never reject
    anything.

    Here the stop goes to the level price must actually defend, and the target
    goes to the next obstacle in the way. Both are market prices, so the ratio
    becomes a measurement of the setup rather than of the configuration.

    A level is only adopted when it is genuinely usable:

    * the stop must sit beyond a support the market has tested (or resistance
      for a short), and never so close that ordinary noise reaches it;
    * the target must clear ``min_rr``; if the closest obstacle does not, the
      next one is taken, and the ATR target is kept only when nothing on the
      chart is reachable.

    Anything unusable falls back to what the strategy already chose, so this
    can only sharpen a setup, never break one.
    """
    if structure is None or atr <= 0:
        return levels

    long = levels.direction is SignalDirection.LONG
    defensive = structure.supports if long else structure.resistances
    obstacle = structure.resistances if long else structure.supports

    def _beyond(price: float, reference: float) -> bool:
        return price < reference if long else price > reference

    def _distance(price: float) -> float:
        return abs(price - levels.entry)

    # -- stop: just beyond the nearest level the price has to defend --------
    # The ATR stop sets the budget: how much risk the strategy was willing to
    # take. Structure decides how much room is actually needed, inside two
    # limits. Beyond the strategy budget (times MAX_STRUCTURE_STOP_BUDGET),
    # because an unbounded level would silently multiply the risk. And beyond
    # MAX_STOP_ATR, because measured results put the good range for an initial
    # stop at 2-4 ATR: wider stops need proportionally wider targets to keep
    # the ratio, and targets that far out stop filling at all.
    #
    # The NEAREST usable level wins, plus a noise buffer so the wick has to
    # travel before the support is even tested. Taking the deepest level
    # instead -- which is what this used to do -- widens the risk for no gain:
    # the extra distance buys no information, it only makes the ratio harder
    # to clear, and a wide stop is exactly what turned every setup into a
    # rejection downstream.
    budget = _distance(levels.stop)
    cap = min(budget * MAX_STRUCTURE_STOP_BUDGET, MAX_STOP_ATR * atr)
    buffer_size = STRUCTURE_STOP_BUFFER_ATR * atr
    stop = levels.stop
    usable = sorted(
        (
            lvl
            for lvl in defensive
            if _beyond(lvl.price, levels.entry)
            and _distance(lvl.price) + buffer_size >= min_stop_atr * atr
            and _distance(lvl.price) + buffer_size <= cap
        ),
        key=lambda lvl: _distance(lvl.price),
    )
    if usable:
        nearest = usable[0].price
        # The stop has to sit on the far side of the level, never on it.
        stop = nearest - buffer_size if long else nearest + buffer_size

    # -- target: the first obstacle that actually pays for the risk ---------
    # Walking outwards until the ratio clears ``min_rr`` is what the docstring
    # always promised. The previous code took the NEAREST obstacle instead,
    # which minimised the reward while the stop was being widened -- the two
    # choices together drove every ratio down and the filter rejected
    # everything the scanner produced.
    #
    # If no obstacle on the chart clears the ratio, the setup genuinely does
    # not pay for its risk and MIN_RR is free to say so. In that case the
    # farthest real obstacle is used rather than a projection: it is price the
    # market has actually defended, and a target nobody reached is not a
    # target. Only with nothing at all on the chart is the risk projected.
    risk = abs(stop - levels.entry)
    target = levels.primary_target
    reachable = sorted(
        (lvl for lvl in obstacle if not _beyond(lvl.price, levels.entry)),
        key=lambda lvl: _distance(lvl.price),
    )
    if reachable:
        paying = [lvl for lvl in reachable if risk > 0 and _distance(lvl.price) >= risk * min_rr]
        target = (paying[0] if paying else reachable[-1]).price
    elif risk > 0:
        projected = risk * min_rr
        target = levels.entry + projected if long else levels.entry - projected

    if stop == levels.stop and target == levels.primary_target:
        return levels
    # Alpaca rejects a price that breaks the sub-penny increment, and a raw
    # structural level carries every decimal the bar data had: 1194.4686506181379
    # came straight off a resistance. Prices at or above $1.00 take two decimals.
    return TradeLevels(
        direction=levels.direction,
        entry=levels.entry,
        stop=round(float(stop), 2),
        targets=(round(float(target), 2),),
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
