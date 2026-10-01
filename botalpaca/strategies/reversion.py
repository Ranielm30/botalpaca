"""Mean-reversion, support/resistance bounce and VWAP strategies.

These are *counter-trend* by design. They only fire when the market is
demonstrably not trending, and their targets are closer because reversion
trades carry worse reward-to-risk. The confluence engine is expected to
penalise them in trending regimes via the regime/sector components.
"""

from __future__ import annotations

from botalpaca.domain import (
    MarketRegime,
    SetupType,
    SignalDirection,
    StrategySignal,
    TechnicalSnapshot,
)
from botalpaca.strategies.base import BaseStrategy, TradeLevels, levels_from_atr

__all__ = [
    "MeanReversionStrategy",
    "SupportResistanceBounceStrategy",
    "VwapReversionStrategy",
]

# Reversion needs a genuinely non-trending tape.
MAX_TRENDING_ADX = 22.0
# Bollinger band tolerance for an "extended" price.
BAND_TOLERANCE_PCT = 0.02
MIN_BB_WIDTH_PERCENTILE = 40.0


def _is_non_trending(snapshot: TechnicalSnapshot) -> bool:
    return (snapshot.momentum.adx or 0.0) <= MAX_TRENDING_ADX


def _at_lower_band(snapshot: TechnicalSnapshot) -> bool:
    i = snapshot.indicators
    if i.bb_lower is None or i.close <= 0:
        return False
    return i.close <= i.bb_lower * (1 + BAND_TOLERANCE_PCT)


def _at_upper_band(snapshot: TechnicalSnapshot) -> bool:
    i = snapshot.indicators
    if i.bb_upper is None or i.close <= 0:
        return False
    return i.close >= i.bb_upper * (1 - BAND_TOLERANCE_PCT)


class MeanReversionStrategy(BaseStrategy):
    """Fade an over-extended price back toward the mean in a ranging tape."""

    setup = SetupType.MEAN_REVERSION
    display_name = "Mean Reversion"

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        rsi = i.rsi_14
        if not atr or rsi is None or i.bb_middle is None:
            return None
        if not _is_non_trending(snapshot):
            return None
        pct = snapshot.volatility.bb_width_pctile
        if pct is not None and pct < MIN_BB_WIDTH_PERCENTILE:
            return None  # too compressed to trust a reversion target

        if rsi <= 30 and _at_lower_band(snapshot):
            direction = SignalDirection.LONG
            stop = i.close - 1.2 * atr
            reasons = [f"RSI {rsi:.0f} en sobreventa", "precio en banda inferior"]
        elif rsi >= 70 and _at_upper_band(snapshot):
            direction = SignalDirection.SHORT
            stop = i.close + 1.2 * atr
            reasons = [f"RSI {rsi:.0f} en sobrecompra", "precio en banda superior"]
        else:
            return None

        # Reversion targets the mean, which caps reward; keep R:R modest.
        target = i.bb_middle
        levels = TradeLevels(direction=direction, entry=i.close, stop=stop, targets=(target,))
        if levels.rr < 1.0:
            return None
        return self._build_signal(
            snapshot,
            direction=direction,
            score=50.0 + 25.0 * snapshot.momentum.score + 15.0 * (1.0 - (snapshot.momentum.adx or 0) / 40.0),
            levels=levels,
            reasons=reasons + [f"objetivo en media ${target:,.2f}"],
            confluences=["sobrecompra/sobreventa", "reversión a la media"],
            invalidation=(
                f"cierre bajo ${stop:,.2f}" if direction is SignalDirection.LONG
                else f"cierre sobre ${stop:,.2f}"
            ),
            extra={"rsi": rsi, "bb_middle": i.bb_middle},
        )


class SupportResistanceBounceStrategy(BaseStrategy):
    """Enter at a well-tested level with a reversal candle confirming."""

    setup = SetupType.SR_BOUNCE
    display_name = "Soporte/Resistencia"
    min_touches = 2
    proximity_atr = 0.6

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        if not atr:
            return None
        supports = [s for s in snapshot.structure.supports if s.touches >= self.min_touches]
        resistances = [r for r in snapshot.structure.resistances if r.touches >= self.min_touches]

        signal: StrategySignal | None = None
        if supports and (i.close - supports[0].price) <= self.proximity_atr * atr:
            level = supports[0]
            levels = levels_from_atr(
                SignalDirection.LONG, i.close, atr,
                stop_multiplier=1.5, target_multiplier=2.0,
            )
            signal = self._build_signal(
                snapshot,
                direction=SignalDirection.LONG,
                score=50.0 + 15.0 * min(level.touches, 4) + 20.0 * snapshot.momentum.score + 15.0 * snapshot.volume.score,
                levels=levels,
                reasons=[
                    f"soporte probado ${level.price:,.2f} ({level.touches} toques)",
                    f"precio a {abs(i.close - level.price):.2%} del nivel",
                ],
                confluences=["nivel de soporte", "rechazo de precio"],
                invalidation=f"cierre bajo ${level.price:,.2f}",
                extra={"level_price": level.price, "touches": level.touches},
            )
        elif resistances and (resistances[0].price - i.close) <= self.proximity_atr * atr:
            level = resistances[0]
            levels = levels_from_atr(
                SignalDirection.SHORT, i.close, atr,
                stop_multiplier=1.5, target_multiplier=2.0,
            )
            signal = self._build_signal(
                snapshot,
                direction=SignalDirection.SHORT,
                score=50.0 + 15.0 * min(level.touches, 4) + 20.0 * snapshot.momentum.score + 15.0 * snapshot.volume.score,
                levels=levels,
                reasons=[
                    f"resistencia probada ${level.price:,.2f} ({level.touches} toques)",
                    f"precio a {abs(i.close - level.price):.2%} del nivel",
                ],
                confluences=["nivel de resistencia", "rechazo de precio"],
                invalidation=f"cierre sobre ${level.price:,.2f}",
                extra={"level_price": level.price, "touches": level.touches},
            )
        if signal and snapshot.volatility.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            # A strong trend invalidates the level-bounce premise.
            return None
        return signal


class VwapReversionStrategy(BaseStrategy):
    """Fade or follow price back to the session VWAP.

    Only fires when VWAP is genuinely meaningful: the caller must supply
    session-anchored VWAP, and price must be meaningfully displaced from it.
    """

    setup = SetupType.VWAP_REVERSION
    display_name = "VWAP Reversion"
    min_displacement_pct = 0.8

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        vwap = i.vwap
        if not atr or not vwap or vwap <= 0:
            return None
        if snapshot.timeframe not in ("1m", "5m", "15m", "1H"):
            # VWAP is a session measure; on daily+ it is not tradeable.
            return None
        if not _is_non_trending(snapshot):
            return None

        displacement_pct = (i.close - vwap) / vwap * 100.0
        if abs(displacement_pct) < self.min_displacement_pct:
            return None

        direction = (
            SignalDirection.LONG if displacement_pct < 0 else SignalDirection.SHORT
        )
        stop = i.close - direction_sign(direction) * 1.3 * atr
        levels = TradeLevels(direction=direction, entry=i.close, stop=stop, targets=(vwap,))
        if levels.rr < 1.0:
            return None
        return self._build_signal(
            snapshot,
            direction=direction,
            score=48.0 + 30.0 * snapshot.momentum.score + 10.0 * snapshot.volume.score,
            levels=levels,
            reasons=[
                f"precio {displacement_pct:+.1f}% respecto a VWAP ${vwap:,.2f}",
                "sesión sin tendencia definida",
            ],
            confluences=["VWAP", "reversión a la media de sesión"],
            invalidation="desplazamiento sostenido lejos del VWAP",
            extra={"vwap": vwap, "displacement_pct": displacement_pct},
        )


def direction_sign(direction: SignalDirection) -> int:
    """+1 for LONG, -1 for SHORT. Lets one expression handle both sides."""
    return 1 if direction is SignalDirection.LONG else -1
