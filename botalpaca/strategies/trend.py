"""Trend and momentum strategies."""

from __future__ import annotations

from botalpaca.domain import (
    SetupType,
    SignalDirection,
    StrategySignal,
    TechnicalSnapshot,
)
from botalpaca.strategies.base import BaseStrategy, levels_from_atr

__all__ = [
    "EmaTrendContinuationStrategy",
    "MomentumStrategy",
    "MultiTimeframeStrategy",
    "RelativeStrengthStrategy",
    "TrendFollowingStrategy",
]


def _long_ready(snapshot: TechnicalSnapshot) -> tuple[bool, list[str]]:
    t = snapshot.trend
    reasons: list[str] = []
    ok = True
    if not t.price_above_ema20:
        ok = False
        reasons.append("precio bajo EMA20")
    if t.ema_alignment <= 0:
        ok = False
        reasons.append("EMAs no ordenadas al alza")
    if t.ema20_slope is not None and t.ema20_slope <= 0:
        ok = False
        reasons.append("pendiente EMA20 no positiva")
    if ok:
        reasons.append("tendencia alcista confirmada por EMAs")
    return ok, reasons


def _short_ready(snapshot: TechnicalSnapshot) -> tuple[bool, list[str]]:
    t = snapshot.trend
    reasons: list[str] = []
    ok = True
    if t.price_above_ema20:
        ok = False
        reasons.append("precio sobre EMA20")
    if t.ema_alignment >= 0:
        ok = False
        reasons.append("EMAs no ordenadas a la baja")
    if t.ema20_slope is not None and t.ema20_slope >= 0:
        ok = False
        reasons.append("pendiente EMA20 no negativa")
    if ok:
        reasons.append("tendencia bajista confirmada por EMAs")
    return ok, reasons


class TrendFollowingStrategy(BaseStrategy):
    """Ride an established, ADX-confirmed trend."""

    setup = SetupType.TREND_FOLLOWING
    display_name = "Trend Following"
    requires_mtf = True
    min_adx = 25.0

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        if not atr:
            return None
        adx = i.adx_14 or 0.0
        if adx < self.min_adx:
            return None

        if snapshot.trend.direction is SignalDirection.LONG and (i.plus_di or 0) > (i.minus_di or 0):
            ready, reasons = _long_ready(snapshot)
            if not ready:
                return None
            direction = SignalDirection.LONG
            reasons.insert(0, f"ADX {adx:.1f} con +DI dominando")
        elif snapshot.trend.direction is SignalDirection.SHORT and (i.minus_di or 0) > (i.plus_di or 0):
            ready, reasons = _short_ready(snapshot)
            if not ready:
                return None
            direction = SignalDirection.SHORT
            reasons.insert(0, f"ADX {adx:.1f} con -DI dominando")
        else:
            return None

        levels = levels_from_atr(
            direction,
            i.close,
            atr,
            stop_multiplier=self.settings.atr_stop_multiplier,
            target_multiplier=self.settings.atr_target_multiplier,
        )
        score = 55.0 + 25.0 * snapshot.trend.strength + 20.0 * snapshot.momentum.score
        return self._build_signal(
            snapshot,
            direction=direction,
            score=score,
            levels=levels,
            reasons=reasons,
            confluences=[f"ADX {adx:.1f}", "estructura de tendencia"],
            invalidation=(
                "cierre por debajo de EMA50" if direction is SignalDirection.LONG
                else "cierre por encima de EMA50"
            ),
            extra={"adx": adx, "plus_di": i.plus_di, "minus_di": i.minus_di},
        )


class EmaTrendContinuationStrategy(BaseStrategy):
    """Enter a pullback that holds inside an established EMA trend."""

    setup = SetupType.EMA_CONTINUATION
    display_name = "EMA Trend Continuation"

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        ema20, ema50 = i.ema_20, i.ema_50
        if not atr or ema20 is None or ema50 is None:
            return None
        if i.ema_200 is None:
            return None
        if abs(ema20 - ema50) / ema50 < 0.002:
            return None  # EMAs too flat to define a trend

        bullish_trend = ema20 > ema50 > i.ema_200
        bearish_trend = ema20 < ema50 < i.ema_200

        signal: StrategySignal | None = None
        if bullish_trend and ema20 <= i.close <= ema20 + atr:
            levels = levels_from_atr(
                SignalDirection.LONG, i.close, atr,
                stop_multiplier=self.settings.atr_stop_multiplier,
                target_multiplier=self.settings.atr_target_multiplier,
            )
            signal = self._build_signal(
                snapshot,
                direction=SignalDirection.LONG,
                score=60.0 + 20.0 * snapshot.trend.strength + 20.0 * snapshot.volume.score,
                levels=levels,
                reasons=[
                    "tendencia EMA20>EMA50>EMA200",
                    "retroceso contenido en EMA20",
                ],
                confluences=["orden EMA", "retroceso a EMA20"],
                invalidation="cierre bajo EMA20",
            )
        elif bearish_trend and ema20 - atr <= i.close <= ema20:
            levels = levels_from_atr(
                SignalDirection.SHORT, i.close, atr,
                stop_multiplier=self.settings.atr_stop_multiplier,
                target_multiplier=self.settings.atr_target_multiplier,
            )
            signal = self._build_signal(
                snapshot,
                direction=SignalDirection.SHORT,
                score=60.0 + 20.0 * snapshot.trend.strength + 20.0 * snapshot.volume.score,
                levels=levels,
                reasons=[
                    "tendencia EMA20<EMA50<EMA200",
                    "retroceso contenido en EMA20",
                ],
                confluences=["orden EMA", "retroceso a EMA20"],
                invalidation="cierre sobre EMA20",
            )
        if signal and snapshot.volume.rel_volume and snapshot.volume.rel_volume >= 1.2:
            signal.reasons.append("volumen acompaña el retroceso")
        return signal


class MomentumStrategy(BaseStrategy):
    """MACD histogram expansion plus RSI strength in a trending tape."""

    setup = SetupType.MOMENTUM
    display_name = "Momentum"

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        hist, rsi = i.macd_hist, i.rsi_14
        if not atr or hist is None or rsi is None:
            return None
        if (snapshot.trend.ema_alignment == 0) and (snapshot.momentum.adx or 0) < 25:
            return None

        if hist > 0 and rsi >= 55 and snapshot.trend.direction is SignalDirection.LONG:
            direction = SignalDirection.LONG
            reasons = ["histograma MACD expandiéndose", f"RSI {rsi:.0f} en zona de fuerza"]
        elif hist < 0 and rsi <= 45 and snapshot.trend.direction is SignalDirection.SHORT:
            direction = SignalDirection.SHORT
            reasons = ["histograma MACD contractándose", f"RSI {rsi:.0f} en zona de debilidad"]
        else:
            return None

        levels = levels_from_atr(
            direction, i.close, atr,
            stop_multiplier=self.settings.atr_stop_multiplier * 1.2,
            target_multiplier=self.settings.atr_target_multiplier,
        )
        return self._build_signal(
            snapshot,
            direction=direction,
            score=50.0 + 30.0 * snapshot.momentum.score + 20.0 * snapshot.trend.strength,
            levels=levels,
            reasons=reasons,
            confluences=["momentum", "MACD", f"RSI {rsi:.0f}"],
            invalidation=("RSI bajo 50" if direction is SignalDirection.LONG else "RSI sobre 50"),
            extra={"rsi": rsi, "macd_hist": hist},
        )


class RelativeStrengthStrategy(BaseStrategy):
    """A symbol outperforming its benchmark in a supportive backdrop."""

    setup = SetupType.RELATIVE_STRENGTH
    display_name = "Relative Strength"
    min_rs = 1.5

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        i = snapshot.indicators
        atr = i.atr_14
        rs = snapshot.context.rs_vs_benchmark
        if not atr or rs is None:
            return None
        if abs(rs) < self.min_rs:
            return None
        direction = SignalDirection.LONG if rs > 0 else SignalDirection.SHORT
        # Counter-trend RS against a hostile benchmark is a short setup.
        bench = snapshot.context.benchmark_trend
        if bench is not None and bench is not direction and (snapshot.trend.ema_alignment == 0):
            return None

        levels = levels_from_atr(
            direction, i.close, atr,
            stop_multiplier=self.settings.atr_stop_multiplier,
            target_multiplier=self.settings.atr_target_multiplier,
        )
        return self._build_signal(
            snapshot,
            direction=direction,
            score=55.0 + 20.0 * snapshot.trend.strength + 15.0 * abs(rs) / 5.0,
            levels=levels,
            reasons=[f"fuerza relativa {rs:+.1f}% vs {snapshot.context.benchmark}"],
            confluences=["fuerza relativa", "comparación con índice"],
            invalidation="pérdida de la ventaja relativa",
            extra={"relative_strength": rs, "benchmark": snapshot.context.benchmark},
        )


class MultiTimeframeStrategy(BaseStrategy):
    """Confluence of the analysis timeframe with a higher-timeframe trend.

    Expects the caller to attach the higher timeframe as
    ``extra["htf_direction"]`` / ``extra["htf_alignment"]``; without it the
    strategy abstains rather than assuming agreement.
    """

    setup = SetupType.MULTI_TIMEFRAME
    display_name = "Multi-Timeframe"
    requires_mtf = True

    def detect(self, snapshot: TechnicalSnapshot) -> StrategySignal | None:
        atr = snapshot.indicators.atr_14
        htf_direction = snapshot.htf_direction
        # Absent higher-timeframe data means "unknown", never "aligned".
        if htf_direction is None or not atr:
            return None
        local = snapshot.trend.direction
        if local is None or local is not htf_direction:
            return None
        levels = levels_from_atr(
            local, snapshot.indicators.close, atr,
            stop_multiplier=self.settings.atr_stop_multiplier * 1.1,
            target_multiplier=self.settings.atr_target_multiplier,
        )
        return self._build_signal(
            snapshot,
            direction=local,
            score=65.0 + 20.0 * snapshot.trend.strength + 15.0 * snapshot.momentum.score,
            levels=levels,
            reasons=[
                f"alineación multi-timeframe: {snapshot.timeframe} y "
                f"{snapshot.htf_timeframe} en {local.value}",
            ],
            confluences=["multi-timeframe", "confirmación de marco superior"],
            invalidation=(
                "pérdida de la tendencia del marco superior"
                if local is SignalDirection.LONG
                else "ganancia de la tendencia del marco superior"
            ),
            extra={"htf_direction": htf_direction.value, "htf_timeframe": snapshot.htf_timeframe},
        )
