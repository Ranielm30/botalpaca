"""Strategy registry and ensemble orchestration."""

from __future__ import annotations

from collections.abc import Iterable

from botalpaca.config import StrategySettings
from botalpaca.domain import SetupType, SignalDirection, StrategySignal, TechnicalSnapshot
from botalpaca.strategies.base import BaseStrategy, make_fingerprint
from botalpaca.strategies.breakout import (
    BreakoutStrategy,
    BreakoutVolumeStrategy,
    PullbackStrategy,
    VolatilityExpansionStrategy,
)
from botalpaca.strategies.reversion import (
    MeanReversionStrategy,
    SupportResistanceBounceStrategy,
    VwapReversionStrategy,
)
from botalpaca.strategies.trend import (
    EmaTrendContinuationStrategy,
    MomentumStrategy,
    MultiTimeframeStrategy,
    RelativeStrengthStrategy,
    TrendFollowingStrategy,
)

__all__ = [
    "ALL_STRATEGIES",
    "STRATEGY_REGISTRY",
    "StrategyEngine",
    "build_strategies",
    "make_fingerprint",
]


def _all_strategy_classes() -> tuple[type[BaseStrategy], ...]:
    return (
        TrendFollowingStrategy,
        EmaTrendContinuationStrategy,
        MomentumStrategy,
        BreakoutStrategy,
        BreakoutVolumeStrategy,
        PullbackStrategy,
        MeanReversionStrategy,
        SupportResistanceBounceStrategy,
        VwapReversionStrategy,
        VolatilityExpansionStrategy,
        RelativeStrengthStrategy,
        MultiTimeframeStrategy,
    )


ALL_STRATEGIES: tuple[type[BaseStrategy], ...] = _all_strategy_classes()
STRATEGY_REGISTRY: dict[SetupType, type[BaseStrategy]] = {
    cls.setup: cls for cls in ALL_STRATEGIES
}


def build_strategies(
    settings: StrategySettings | None = None, *, only: Iterable[SetupType] | None = None
) -> list[BaseStrategy]:
    """Instantiate the strategy set, optionally filtered by setup type."""
    settings = settings or StrategySettings()
    selected = set(only) if only is not None else set(STRATEGY_REGISTRY)
    return [
        STRATEGY_REGISTRY[setup](settings)
        for setup in STRATEGY_REGISTRY
        if setup in selected
    ]


class StrategyEngine:
    """Runs every strategy and merges same-direction signals into one.

    Merge rule: the highest-scoring signal for a direction provides the
    levels (so a weak mean-reversion setup cannot drag a strong trend
    entry's stop somewhere dangerous), and the reasons of every contributing
    strategy are unioned as confluence evidence.
    """

    def __init__(
        self,
        settings: StrategySettings | None = None,
        strategies: Iterable[BaseStrategy] | None = None,
    ) -> None:
        self.settings = settings or StrategySettings()
        self._strategies = list(strategies) if strategies is not None else build_strategies(self.settings)

    @property
    def strategies(self) -> list[BaseStrategy]:
        return list(self._strategies)

    def run(self, snapshot: TechnicalSnapshot) -> list[StrategySignal]:
        """Return one merged signal per direction, best first."""
        collected: list[StrategySignal] = []
        for strategy in self._strategies:
            try:
                signal = strategy.detect(snapshot)
            except ValueError:
                # An unusable ATR or malformed series means "no setup",
                # never a crash that would abort the whole scan.
                continue
            if signal is not None:
                collected.append(signal)
        return self._merge(collected, snapshot.timeframe)

    def _merge(
        self, signals: list[StrategySignal], timeframe: str
    ) -> list[StrategySignal]:
        by_direction: dict[SignalDirection, list[StrategySignal]] = {}
        for sig in signals:
            by_direction.setdefault(sig.direction, []).append(sig)

        merged: list[StrategySignal] = []
        for direction, group in by_direction.items():
            group.sort(key=lambda s: s.score, reverse=True)
            best = group[0]
            reasons = list(dict.fromkeys(r for s in group for r in s.reasons))
            confluences = list(dict.fromkeys(c for s in group for c in s.confluences))
            # Multi-strategy agreement is itself evidence.
            if len(group) > 1:
                confluences.append(f"{len(group)} estrategias coinciden")
                reasons.append(
                    "acuerdo de " + ", ".join(s.strategy.value for s in group)
                )
            agreement_bonus = min(15.0, (len(group) - 1) * 7.5)
            merged.append(
                StrategySignal(
                    strategy=best.strategy,
                    direction=direction,
                    score=min(100.0, best.score + agreement_bonus),
                    entry=best.entry,
                    stop=best.stop,
                    targets=list(best.targets),
                    rr=best.rr,
                    reasons=reasons,
                    confluences=confluences,
                    invalidation=best.invalidation,
                    timeframe=timeframe,
                    extra={
                        "contributing_strategies": [s.strategy.value for s in group],
                        "signal_count": len(group),
                    },
                )
            )
        merged.sort(key=lambda s: s.score, reverse=True)
        return merged
