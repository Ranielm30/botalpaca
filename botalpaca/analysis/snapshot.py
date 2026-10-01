"""Technical snapshot assembly: bars in, complete analysis out."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from botalpaca.analysis.market_context import (
    build_market_context,
    compute_correlation,
    compute_relative_strength,
    sector_for_symbol,
)
from botalpaca.analysis.technical import (
    analyze_momentum,
    analyze_structure,
    analyze_trend,
    analyze_volatility,
    analyze_volume,
)
from botalpaca.config import StrategySettings
from botalpaca.domain import Bar, Quote, TechnicalSnapshot
from botalpaca.indicators.service import (
    bb_width_percentile,
    compute_indicators,
    obv_slope_pct,
    realized_volatility_pct,
)

__all__ = ["analyze_symbol", "assess_data_quality"]

# Warm-up requirements per indicator, used for the data-quality score.
REQUIRED_BARS = 30
GOOD_BARS = 200
EMA200_BARS = 200


def assess_data_quality(
    bars: Sequence[Bar], indicators: object, *, missing_indicators: int = 0
) -> float:
    """0..1 confidence in the underlying data.

    Feeds the confluence engine: a thin dataset must not produce the same
    score as a deep one just because the available fields look "good".
    """
    n = len(bars)
    depth = min(1.0, n / float(GOOD_BARS))
    has_long_ema = getattr(indicators, "ema_200", None) is not None
    long_bonus = 0.1 if has_long_ema else 0.0
    penalty = 0.05 * missing_indicators
    score = 0.4 + 0.4 * depth + long_bonus - penalty
    if n < REQUIRED_BARS:
        return 0.0
    return max(0.0, min(1.0, score))


def analyze_symbol(
    symbol: str,
    bars: Sequence[Bar],
    *,
    timeframe: str = "1D",
    quote: Quote | None = None,
    settings: StrategySettings | None = None,
    benchmark_indicators: object | None = None,
    benchmark_bars: Sequence[Bar] | None = None,
    sector_indicators: object | None = None,
    benchmark_symbol: str = "SPY",
    market_open: bool = False,
) -> TechnicalSnapshot:
    """Run the full analysis pipeline for one symbol on one timeframe.

    Raises :class:`~botalpaca.domain.DataQualityError` when the bar series is
    too short, so callers can skip the symbol instead of scoring thin data.
    """
    settings = settings or StrategySettings()
    symbol = symbol.upper()
    indicators = compute_indicators(
        bars, timeframe, min_bars=max(settings.min_bars_required, REQUIRED_BARS)
    )

    obv_slope = obv_slope_pct(bars)
    bb_pct = bb_width_percentile(bars)
    realized_vol = realized_volatility_pct(bars)

    trend = analyze_trend(indicators, bars)
    momentum = analyze_momentum(indicators, bars)
    volume = analyze_volume(indicators, obv_slope)
    volatility = analyze_volatility(indicators, bb_percentile=bb_pct, realized_vol=realized_vol)
    structure = analyze_structure(bars, indicators)

    sector = sector_for_symbol(symbol)
    rel_strength = None
    correlation = None
    if benchmark_indicators is not None:
        rel_strength = compute_relative_strength(indicators, benchmark_indicators)  # type: ignore[arg-type]
    if benchmark_bars:
        correlation = compute_correlation(bars, benchmark_bars)

    context = build_market_context(
        symbol_indicators=indicators,
        benchmark_symbol=benchmark_symbol,
        benchmark_indicators=benchmark_indicators,  # type: ignore[arg-type]
        benchmark_bars=benchmark_bars,
        sector_name=sector,
        sector_indicators=sector_indicators,  # type: ignore[arg-type]
        relative_strength=rel_strength,
        correlation=correlation,
        market_open=market_open,
    )

    missing = sum(
        1
        for v in (indicators.ema_200, indicators.ema_100, indicators.adx_14, indicators.atr_14)
        if v is None
    )
    quality = assess_data_quality(bars, indicators, missing_indicators=missing)

    return TechnicalSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        as_of=bars[-1].timestamp if bars else dt.datetime.now(dt.UTC),
        indicators=indicators,
        trend=trend,
        momentum=momentum,
        volume=volume,
        volatility=volatility,
        structure=structure,
        context=context,
        quote=quote,
        data_quality=quality,
    )
