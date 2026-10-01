"""Breakout, volatility-expansion and pullback strategies."""

from __future__ import annotations

from botalpaca.domain import (
    SetupType,
    SignalDirection,
    StrategySignal,
    TechnicalSnapshot,
)
from botalpaca.strategies.base import BaseStrategy, levels_from_atr

__all__ = [
    "BreakoutStrategy",
    "BreakoutVolumeStrategy",
    "PullbackStrategy",
    "VolatilityExpansionStrategy",
]

# A breakout needs a minimum amount of range expansion to be worth trading.
MIN_EXPANSION_PERCENTILE = 60.0
# Volume confirmation threshold for the volume-confirmed variant.
CONFIRMATION_VOLUME = 1.3


class _BreakoutBase(BaseStrategy):
    """Shared detection: close beyond the N-bar extreme on expanding range."""

    breakout_lookback = 20
    require_expansion = True
    min_score_floor = 45.0

    def _levels_for_breakout(
        self, snapshot: TechnicalSnapshot, direction: SignalDirection, atr: float
    ):  # noqa: ANN202
        """Stop goes below the broken level (or the range low), not at ATR distance.

        A breakout stop placed at a fixed ATR distance from entry can land
        *inside* the prior range, where it is trivially swept. Anchoring to
        the structural level is what makes the stop meaningful.
        """
        i = snapshot.indicators
        entry = i.close
        if direction is SignalDirection.LONG:
            anchor = i.lowest_20
            stop = (anchor - 0.25 * atr) if anchor else entry - 1.5 * atr
            target = entry + (entry - stop) * 2.0
        else:
            anchor = i.highest_20
            stop = (anchor + 0.25 * atr) if anchor else entry + 1.5 * atr
            target = entry - (stop - entry) * 2.0
        if stop <= 0:
            stop = entry - 1.5 * atr if direction is SignalDirection.LONG else entry + 1.5 * atr
        from botalpaca.strategies.base import TradeLevels

        return TradeLevels(direction=direction, entry=entry, stop=stop, targets=(target,))

    def _detect_breakout(
        self, snapshot: TechnicalSnapshot, *, require_volume: bool
    ) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        if not atr:
            return None
        if not snapshot.structure.breakout:
            return None
        direction = snapshot.structure.breakout_direction
        if direction is None:
            return None

        pct = snapshot.volatility.bb_width_pctile
        if self.require_expansion and pct is not None and pct < MIN_EXPANSION_PERCENTILE:
            return None

        rel_vol = i.rel_volume
        if require_volume and (rel_vol is None or rel_vol < CONFIRMATION_VOLUME):
            return None

        levels = self._levels_for_breakout(snapshot, direction, atr)
        if levels.risk_per_share <= 0 or levels.rr < 1.0:
            return None

        confluences = ["ruptura de rango", f"ancho Bollinger percentil {pct:.0f}"] if pct else ["ruptura de rango"]
        reasons = [
            "cierre fuera del rango de 20 velas",
            f"soporte/resistencia previa en ${levels.stop:,.2f}",
        ]
        if rel_vol:
            reasons.append(f"volumen relativo {rel_vol:.2f}x")
        if snapshot.structure.retest:
            reasons.append("retest del nivel roto")

        score = self.min_score_floor + 25.0 * snapshot.trend.strength + 20.0 * snapshot.volume.score
        if require_volume:
            score += 10.0
        if snapshot.structure.retest:
            score += 5.0
        return self._build_signal(
            snapshot,
            direction=direction,
            score=score,
            levels=levels,
            reasons=reasons,
            confluences=confluences,
            invalidation=(
                f"cierre de nuevo dentro del rango (bajo ${levels.stop:,.2f})"
                if direction is SignalDirection.LONG
                else f"cierre de nuevo dentro del rango (sobre ${levels.stop:,.2f})"
            ),
            extra={"bb_width_percentile": pct, "rel_volume": rel_vol},
        )


class BreakoutStrategy(_BreakoutBase):
    setup = SetupType.BREAKOUT
    display_name = "Breakout"
    requires_mtf = True

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        return self._detect_breakout(snapshot, require_volume=False)


class BreakoutVolumeStrategy(_BreakoutBase):
    setup = SetupType.BREAKOUT_VOLUME
    display_name = "Breakout + Volume"
    requires_mtf = True
    min_score_floor = 50.0

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        return self._detect_breakout(snapshot, require_volume=True)


class PullbackStrategy(BaseStrategy):
    """Enter a pullback toward a rising support inside a trending move."""

    setup = SetupType.PULLBACK
    display_name = "Pullback"

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        ema20, ema50 = i.ema_20, i.ema_50
        if not atr or ema20 is None or ema50 is None or i.ema_200 is None:
            return None
        if abs(ema20 - ema50) / ema50 < 0.003:
            return None
        if (snapshot.momentum.adx or 0) < 20:
            return None

        if ema20 > ema50 > i.ema_200:
            # Pulling back means price sits above EMA50 but within reach of EMA20.
            if not (ema50 < i.close <= ema20 + 0.5 * atr):
                return None
            if i.rsi_14 is not None and i.rsi_14 < 40:
                return None  # momentum already broken
            direction = SignalDirection.LONG
            stop = ema50 - 0.5 * atr
            target = ema20 + (ema20 - stop) * 2.0
            reasons = ["tendencia alcista intacta", "retroceso hacia EMA20 con soporte en EMA50"]
        elif ema20 < ema50 < i.ema_200:
            if not (ema50 > i.close >= ema20 - 0.5 * atr):
                return None
            if i.rsi_14 is not None and i.rsi_14 > 60:
                return None
            direction = SignalDirection.SHORT
            stop = ema50 + 0.5 * atr
            target = ema20 - (stop - ema20) * 2.0
            reasons = ["tendencia bajista intacta", "retroceso hacia EMA20 con resistencia en EMA50"]
        else:
            return None

        from botalpaca.strategies.base import TradeLevels

        levels = TradeLevels(direction=direction, entry=i.close, stop=stop, targets=(target,))
        if levels.rr < 1.2:
            return None
        return self._build_signal(
            snapshot,
            direction=direction,
            score=58.0 + 22.0 * snapshot.trend.strength + 20.0 * snapshot.volume.score,
            levels=levels,
            reasons=reasons,
            confluences=["retroceso", "soporte EMA50"],
            invalidation=(
                f"cierre bajo EMA50 (${ema50:,.2f})"
                if direction is SignalDirection.LONG
                else f"cierre sobre EMA50 (${ema50:,.2f})"
            ),
        )


class VolatilityExpansionStrategy(BaseStrategy):
    """Trade the expansion out of a Bollinger squeeze."""

    setup = SetupType.VOLATILITY_EXPANSION
    display_name = "Volatility Expansion"
    min_expansion_percentile = 70.0

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        pct = snapshot.volatility.bb_width_pctile
        if not atr or pct is None or pct < self.min_expansion_percentile:
            return None

        upper, lower, mid = i.bb_upper, i.bb_lower, i.bb_middle
        if upper is None or lower is None or mid is None:
            return None
        if (i.close - lower) / (upper - lower) > 0.8:
            direction = SignalDirection.LONG
        elif (i.close - lower) / (upper - lower) < 0.2:
            direction = SignalDirection.SHORT
        else:
            return None  # mid-band: no expansion direction yet

        if snapshot.trend.direction is not None and snapshot.trend.direction is not direction:
            return None

        levels = levels_from_atr(
            direction, i.close, atr,
            stop_multiplier=self.settings.atr_stop_multiplier * 0.8,
            target_multiplier=self.settings.atr_target_multiplier,
        )
        return self._build_signal(
            snapshot,
            direction=direction,
            score=52.0 + 25.0 * snapshot.volatility.score + 15.0 * snapshot.trend.strength,
            levels=levels,
            reasons=[
                f"expansión de volatilidad (percentil {pct:.0f})",
                "precio en el extremo de la banda",
            ],
            confluences=["expansión Bollinger", "squeeze liberado"],
            invalidation="re-entrada en la banda media",
            extra={"bb_width_percentile": pct},
        )
