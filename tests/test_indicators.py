"""Indicator correctness: pure numpy implementations + the IndicatorSet service."""

from __future__ import annotations

import numpy as np
import pytest

from botalpaca.domain.errors import DataQualityError
from botalpaca.indicators import core
from botalpaca.indicators.service import compute_indicators

from .conftest import flat_bars, make_bars


def test_sma_matches_manual_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = core.sma(values, 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[4] == pytest.approx(4.0)


def test_ema_converges_to_constant_series():
    values = np.full(200, 50.0)
    out = core.ema(values, 20)
    assert out[-1] == pytest.approx(50.0, abs=1e-9)


def test_ema_is_seeded_with_sma():
    values = np.array([float(i) for i in range(1, 31)])
    out = core.ema(values, 10)
    assert out[9] == pytest.approx(values[:10].mean())


def test_rsi_bounds_and_extremes():
    rising = np.arange(1, 60, dtype=float)
    assert 0.0 <= core.rsi(rising, 14)[-1] <= 100.0
    assert core.rsi(rising, 14)[-1] > 90.0
    falling = np.arange(60, 1, -1, dtype=float)
    assert core.rsi(falling, 14)[-1] < 10.0


def test_rsi_of_flat_series_is_neutral():
    out = core.rsi(np.full(60, 42.0), 14)
    assert out[-1] == pytest.approx(50.0, abs=0.5)


def test_atr_positive_and_tracks_range():
    bars = make_bars(120)
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    atr = core.atr(highs, lows, closes, 14)
    assert np.isnan(atr[0])
    assert atr[-1] > 0


def test_adx_is_nan_until_warmup_then_valid():
    bars = make_bars(200, seed=3)
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    adx, plus, minus = core.adx(highs, lows, closes, 14)
    # Wilder's ADX needs 2 * period bars before the first value.
    assert np.isnan(adx[:28]).all()
    assert np.isfinite(adx[28])
    assert 0.0 <= adx[-1] <= 100.0
    assert 0.0 <= plus[-1] <= 100.0
    assert 0.0 <= minus[-1] <= 100.0


def test_bollinger_bands_ordered():
    closes = np.array([b.close for b in make_bars(120)])
    upper, mid, lower = core.bollinger_bands(closes, 20, 2.0)
    valid = ~np.isnan(upper)
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_obv_follows_price_direction():
    closes = np.array([10.0, 11.0, 10.5, 12.0])
    volumes = np.array([100.0, 100.0, 100.0, 100.0])
    out = core.obv(closes, volumes)
    assert out[1] == 100.0
    assert out[2] == 0.0
    assert out[3] == 100.0


def test_roc_positive_on_rise():
    closes = np.array([float(i) for i in range(1, 30)])
    assert core.roc(closes, 10)[-1] > 0


def test_vwap_is_cumulative_typical_price_average():
    h = np.array([10.0, 20.0, 30.0])
    lows = np.array([10.0, 20.0, 30.0])
    closes = np.array([10.0, 20.0, 30.0])
    volumes = np.ones(3)
    out = core.vwap(h, lows, closes, volumes)
    assert out[-1] == pytest.approx(20.0)


def test_vwap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        core.vwap(np.ones(3), np.ones(3), np.ones(3), np.ones(2))


def test_vwap_session_resets_each_day():
    h = np.array([10.0, 20.0, 30.0, 40.0])
    lows = np.array([10.0, 20.0, 30.0, 40.0])
    closes = np.array([10.0, 20.0, 30.0, 40.0])
    volumes = np.ones(4)
    ts = np.array(["2024-01-01T10:00", "2024-01-01T11:00", "2024-01-02T10:00", "2024-01-02T11:00"])
    out = core.vwap_session(ts, h, lows, closes, volumes)
    assert out[1] == pytest.approx(15.0)
    assert out[3] == pytest.approx(35.0)


def test_highest_lowest_and_slope():
    values = np.array([1.0, 5.0, 3.0, 2.0, 9.0])
    assert core.highest(values, 5)[-1] == 9.0
    assert core.lowest(values, 5)[-1] == 1.0
    rising = np.arange(50, dtype=float)
    assert core.linear_slope(rising, 20) > 0


def test_linear_slope_is_normalized_percent_per_bar():
    # Least-squares slope of a step at the end of the window, expressed as
    # percent per bar relative to the mean of the window.
    values = np.full(30, 100.0)
    values[-1] = 110.0
    window = values[-20:]
    x = np.arange(20, dtype=float)
    slope_pct = np.polyfit(x, window, 1)[0] / window.mean() * 100.0
    assert core.linear_slope(values, 20) == pytest.approx(slope_pct, rel=1e-9)
    assert core.linear_slope(values, 20) > 0.0


def test_compute_indicators_returns_full_set():
    bars = make_bars(260)
    ind = compute_indicators(bars, "1D")
    assert ind.timeframe == "1D"
    assert ind.close == pytest.approx(bars[-1].close)
    assert ind.ema_200 > 0
    assert 0 <= ind.rsi_14 <= 100
    assert ind.atr_14 > 0


def test_compute_indicators_rejects_short_series():
    with pytest.raises(DataQualityError):
        compute_indicators(make_bars(10), "1D")


def test_compute_indicators_rejects_non_positive_close():
    bars = flat_bars(80)
    bars[-1] = bars[-1].model_copy(update={"close": 0.0})
    with pytest.raises(DataQualityError):
        compute_indicators(bars, "1D")


def test_compute_indicators_rejects_inverted_bar():
    bars = flat_bars(80)
    bars[-1] = bars[-1].model_copy(update={"high": 1.0, "low": 500.0})
    with pytest.raises(DataQualityError):
        compute_indicators(bars, "1D")


def test_realized_volatility_and_correlation_take_bar_series():
    from botalpaca.indicators.service import correlation, realized_volatility_pct

    bars = make_bars(200, seed=11)
    assert realized_volatility_pct(bars, 20) >= 0.0
    assert realized_volatility_pct(make_bars(5), 20) is None

    # A perfectly linear co-move is the same series rescaled, so r == 1.
    other = [
        b.model_copy(update={"close": b.close * 1.5 + 0.5}) for b in make_bars(200, seed=11)
    ]
    assert correlation(bars, other) == pytest.approx(1.0, abs=1e-6)
    assert correlation(bars, make_bars(200, seed=77)) < 1.0
