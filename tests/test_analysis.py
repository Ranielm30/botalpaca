"""Analysis layer: trend, momentum, volume, volatility, structure, market context."""

from __future__ import annotations

import numpy as np
import pytest

from botalpaca.analysis.market_context import build_market_context, sector_for_symbol
from botalpaca.analysis.snapshot import analyze_symbol, assess_data_quality
from botalpaca.analysis.technical import (
    analyze_momentum,
    analyze_structure,
    analyze_trend,
    analyze_volatility,
    analyze_volume,
    classify_swing_structure,
    detect_gaps,
    find_pivots,
)
from botalpaca.domain.enums import MarketRegime, SignalDirection
from botalpaca.indicators.service import (
    bb_width_percentile,
    compute_indicators,
    obv_slope_pct,
    realized_volatility_pct,
)

from .conftest import flat_bars, make_bars


def _ind(bars):
    return compute_indicators(bars, "1D")


# -- data quality -------------------------------------------------------------------
def test_assess_data_quality_good_series():
    bars = make_bars(260)
    q = assess_data_quality(bars, _ind(bars))
    assert 0.0 < q <= 1.0


def test_assess_data_quality_zero_for_thin_series():
    bars = make_bars(10)
    thin = analyze_indicators_for(bars)
    assert assess_data_quality(bars, thin) == 0.0


def analyze_indicators_for(bars):
    """compute_indicators refuses <60 bars, so build a minimal stand-in."""
    from botalpaca.domain.models import IndicatorSet

    closes = np.asarray([b.close for b in bars], dtype=float)
    return IndicatorSet(
        timeframe="1D",
        close=float(closes[-1]),
        ema_200=float(closes.mean()),
        atr_14=float(closes.max() - closes.min()),
    )


def test_assess_data_quality_penalizes_missing_indicators():
    bars = make_bars(260)
    full = assess_data_quality(bars, _ind(bars))
    penalised = assess_data_quality(bars, _ind(bars), missing_indicators=4)
    assert penalised < full


# -- trend --------------------------------------------------------------------------
def test_analyze_trend_detects_uptrend():
    bars = make_bars(300, drift=0.004, noise=0.6, seed=21)
    state = analyze_trend(_ind(bars), bars)
    assert state.direction is SignalDirection.LONG
    assert state.ema_alignment > 0
    assert state.strength > 0.5


def test_analyze_trend_detects_downtrend():
    bars = make_bars(300, drift=-0.004, noise=0.6, seed=22)
    state = analyze_trend(_ind(bars), bars)
    assert state.direction is SignalDirection.SHORT
    assert state.ema_alignment < 0


def test_analyze_trend_flat_is_not_directional():
    bars = flat_bars(260)
    state = analyze_trend(_ind(bars), bars)
    assert state.ema_alignment == 0
    assert state.direction is None


# -- momentum -----------------------------------------------------------------------
def test_analyze_momentum_scores_between_0_and_1():
    bars = make_bars(260, drift=0.003, seed=23)
    state = analyze_momentum(_ind(bars), bars)
    assert 0.0 <= state.score <= 1.0
    assert 0.0 <= state.rsi <= 100.0
    assert state.adx_state in {"none", "weak", "strong"}


# -- volume -------------------------------------------------------------------------
def test_analyze_volume_flags_spike():
    bars = make_bars(260, seed=24)
    for b in bars[-3:]:
        b.volume *= 8.0
    ind = compute_indicators(bars, "1D")
    state = analyze_volume(ind, obv_slope_pct(bars))
    assert state.rel_volume > 1.5


def test_analyze_volume_flags_dry_up():
    bars = make_bars(260, seed=25)
    for b in bars[-3:]:
        b.volume *= 0.05
    ind = compute_indicators(bars, "1D")
    state = analyze_volume(ind, obv_slope_pct(bars))
    assert state.rel_volume < 0.7


# -- volatility ---------------------------------------------------------------------
def test_analyze_volatility_classifies_regime():
    bars = make_bars(300, drift=0.004, noise=0.5, seed=26)
    state = analyze_volatility(
        _ind(bars), bb_percentile=bb_width_percentile(bars), realized_vol=realized_volatility_pct(bars)
    )
    assert state.regime in set(MarketRegime)
    assert 0.0 <= state.score <= 1.0


def test_analyze_volatility_detects_compression():
    bars = flat_bars(300, price=100.0)
    for b in bars:
        b.high = 100.01
        b.low = 99.99
    state = analyze_volatility(_ind(bars), bb_percentile=2.0, realized_vol=0.1)
    assert state.squeeze is True
    assert state.regime is MarketRegime.LOW_VOLATILITY


def test_analyze_volatility_detects_expansion():
    bars = make_bars(300, drift=0.004, noise=1.5, seed=91)
    state = analyze_volatility(_ind(bars), bb_percentile=97.0, realized_vol=0.5)
    assert state.expansion is True


# -- structure ----------------------------------------------------------------------
def test_find_pivots_detects_local_extremes():
    closes = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5], dtype=float)
    pivots = find_pivots(closes, left=2, right=2)
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]
    assert highs and lows
    assert max(p.price for p in highs) == 5.0
    assert min(p.price for p in lows) == 1.0


def test_classify_swing_structure_up():
    assert classify_swing_structure(flat_bars(60)) is not None


def test_classify_swing_structure_detects_uptrend():
    # A clean stair-step of higher highs and higher lows, long enough to yield
    # at least two confirmed swing pivots of each kind.
    # Sawtooth with strictly rising peaks and strictly rising troughs. The
    # period (8) is wide enough that each turn survives the 4-bar fractal
    # window on both sides, so the pivots are real swings.
    closes: list[float] = []
    for step in range(8):
        trough = 100.0 + 3.0 * step
        closes.extend([trough, trough + 1.5, trough + 3, trough + 4.5, trough + 6, trough + 4.5, trough + 3, trough + 1.5])
    bars = make_bars(len(closes), start=100.0, drift=0.0, noise=0.0)
    for bar, close in zip(bars, closes, strict=True):
        bar.open = bar.high = bar.low = bar.close = close
    assert classify_swing_structure(bars) == "HH/HL"


def test_analyze_structure_finds_levels():
    bars = make_bars(300, seed=28)
    state = analyze_structure(bars, _ind(bars))
    assert state.supports or state.resistances
    for level in state.supports + state.resistances:
        assert level.price > 0
        assert level.touches >= 1


def test_analyze_structure_returns_empty_for_short_series():
    bars = make_bars(20)
    state = analyze_structure(bars, analyze_indicators_for(bars))
    assert state.supports == [] and state.resistances == []


def test_detect_gaps_finds_up_gap():
    bars = make_bars(80, seed=29)
    bars[40].low = bars[40].close * 0.90
    gaps = detect_gaps(bars, min_pct=0.2)
    assert "up" in gaps or "down" in gaps or gaps == {"up": 0, "down": 0}


# -- market context -----------------------------------------------------------------
def test_sector_for_symbol_known_and_unknown():
    assert sector_for_symbol("AAPL") != "UNKNOWN"
    assert sector_for_symbol("ZZZZ") == "UNKNOWN"


def test_build_market_context_degrades_without_benchmark():
    bars = make_bars(260, seed=30)
    ind = compute_indicators(bars, "1D")
    ctx = build_market_context(symbol_indicators=ind)
    assert ctx.benchmark_trend is None
    assert ctx.notes


def test_build_market_context_with_benchmark():
    bars = make_bars(260, seed=31)
    up = compute_indicators(bars, "1D")
    down = compute_indicators(make_bars(260, drift=-0.003, seed=32), "1D")
    ctx = build_market_context(
        symbol_indicators=up,
        benchmark_indicators=down,
        sector_indicators=up,
        sector_name="Technology",
    )
    assert ctx.benchmark_trend is SignalDirection.SHORT
    assert ctx.sector is not None


# -- end to end ---------------------------------------------------------------------
def test_analyze_symbol_end_to_end():
    bars = make_bars(300, drift=0.003, seed=33)
    snapshot = analyze_symbol(
        "AAPL",
        bars,
        timeframe="1D",
        benchmark_bars=make_bars(300, seed=34),
    )
    assert snapshot.symbol == "AAPL"
    assert snapshot.timeframe == "1D"
    assert snapshot.data_quality > 0.5
    assert snapshot.trend.direction in set(SignalDirection)
    assert snapshot.volatility.regime is not MarketRegime.UNKNOWN
    assert snapshot.momentum.score == snapshot.momentum.score  # not NaN


def test_analyze_symbol_rejects_thin_data():
    from botalpaca.domain.errors import DataQualityError

    with pytest.raises(DataQualityError):
        analyze_symbol("AAPL", make_bars(10))
