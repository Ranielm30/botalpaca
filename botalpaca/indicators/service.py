"""Bar-window -> IndicatorSet conversion.

This is the single place where raw candles become the numeric inputs the
analysis and strategy layers consume.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from botalpaca.domain import Bar, DataQualityError, IndicatorSet
from botalpaca.indicators import core as ind


def _col(bars: Sequence[Bar], attr: str) -> np.ndarray:
    return np.asarray([getattr(b, attr) for b in bars], dtype=np.float64)


def _last_finite(series: np.ndarray) -> float | None:
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def compute_indicators(
    bars: Sequence[Bar],
    timeframe: str = "1D",
    *,
    min_bars: int = 60,
    session_anchored_vwap: bool = True,
) -> IndicatorSet:
    """Build a complete indicator set from an oldest->newest bar series.

    Raises :class:`DataQualityError` when there is not enough history, because
    emitting a partial set here would silently produce confident-looking
    signals built on incomplete data.
    """
    if len(bars) < min_bars:
        raise DataQualityError(
            f"{timeframe}: need at least {min_bars} bars, received {len(bars)}"
        )
    if len(bars) < 30:
        raise DataQualityError(f"{timeframe}: need at least 30 bars, received {len(bars)}")

    highs, lows, closes, volumes = _col(bars, "high"), _col(bars, "low"), _col(bars, "close"), _col(bars, "volume")
    if np.any(closes <= 0):
        raise DataQualityError(f"{timeframe}: non-positive close prices in series")
    if np.any(highs < lows):
        raise DataQualityError(f"{timeframe}: inconsistent bars (high < low)")

    ema9 = ind.ema(closes, 9)
    ema20 = ind.ema(closes, 20)
    ema21 = ind.ema(closes, 21)
    ema50 = ind.ema(closes, 50)
    ema100 = ind.ema(closes, 100)
    ema200 = ind.ema(closes, 200)
    sma20 = ind.sma(closes, 20)
    sma50 = ind.sma(closes, 50)
    rsi14 = ind.rsi(closes, 14)
    macd_line, macd_signal, macd_hist = ind.macd(closes)
    adx14, plus_di, minus_di = ind.adx(highs, lows, closes, 14)
    atr14 = ind.atr(highs, lows, closes, 14)
    bb_u, bb_m, bb_l = ind.bollinger_bands(closes, 20, 2.0)
    obv_series = ind.obv(closes, volumes)
    roc10 = ind.roc(closes, 10)
    vol_sma20 = ind.sma(volumes, 20)
    highest20 = ind.highest(closes, 20)
    lowest20 = ind.lowest(closes, 20)
    highest52 = ind.highest(closes, 252) if closes.size >= 252 else np.full(closes.size, np.nan)
    lowest52 = ind.lowest(closes, 252) if closes.size >= 252 else np.full(closes.size, np.nan)

    if session_anchored_vwap:
        stamps = np.asarray([b.timestamp for b in bars])
        vwap_series = ind.vwap_session(stamps, highs, lows, closes, volumes)
    else:
        vwap_series = ind.vwap(highs, lows, closes, volumes)

    close = float(closes[-1])
    atr_val = _last_finite(atr14)
    rel_vol = None
    vol_sma_val = _last_finite(vol_sma20)
    if vol_sma_val and vol_sma_val > 0:
        rel_vol = float(volumes[-1]) / vol_sma_val

    bb_u_val = _last_finite(bb_u)
    bb_l_val = _last_finite(bb_l)
    bb_m_val = _last_finite(bb_m)
    bb_width = None
    if bb_u_val is not None and bb_l_val is not None and bb_m_val not in (None, 0):
        bb_width = (bb_u_val - bb_l_val) / bb_m_val

    # The rejection wick of the last bar, in ATR units. ATR rather than the
    # bar's own range, so "big enough to matter" means the same thing on a
    # quiet tape and on a violent one.
    bull_rej = bear_rej = None
    if atr_val and atr_val > 0:
        bull_frac, bear_frac = ind.rejection_wicks(
            _col(bars, "open"), highs, lows, closes
        )
        bull_rej = bull_frac * float(highs[-1] - lows[-1]) / atr_val
        bear_rej = bear_frac * float(highs[-1] - lows[-1]) / atr_val

    return IndicatorSet(
        timeframe=timeframe,
        close=close,
        ema_9=_last_finite(ema9),
        ema_20=_last_finite(ema20),
        ema_21=_last_finite(ema21),
        ema_50=_last_finite(ema50),
        ema_100=_last_finite(ema100),
        ema_200=_last_finite(ema200),
        sma_20=_last_finite(sma20),
        sma_50=_last_finite(sma50),
        rsi_14=_last_finite(rsi14),
        macd=_last_finite(macd_line),
        macd_signal=_last_finite(macd_signal),
        macd_hist=_last_finite(macd_hist),
        adx_14=_last_finite(adx14),
        plus_di=_last_finite(plus_di),
        minus_di=_last_finite(minus_di),
        atr_14=atr_val,
        atr_pct=(atr_val / close * 100.0) if atr_val and close > 0 else None,
        bb_upper=bb_u_val,
        bb_middle=bb_m_val,
        bb_lower=bb_l_val,
        bb_width=bb_width,
        obv=_last_finite(obv_series),
        roc_10=_last_finite(roc10),
        vwap=_last_finite(vwap_series),
        volume_sma_20=vol_sma_val,
        rel_volume=rel_vol,
        highest_20=_last_finite(highest20),
        lowest_20=_last_finite(lowest20),
        highest_52w=_last_finite(highest52),
        lowest_52w=_last_finite(lowest52),
        bull_rejection_atr=bull_rej,
        bear_rejection_atr=bear_rej,
    )


def obv_slope_pct(bars: Sequence[Bar], lookback: int = 10) -> float | None:
    """Normalised OBV slope: rising OBV confirms the move, flat does not."""
    if len(bars) < lookback + 2:
        return None
    closes, volumes = _col(bars, "close"), _col(bars, "volume")
    series = ind.obv(closes, volumes)
    slope = ind.linear_slope(series, lookback)
    if np.isnan(slope):
        return None
    return float(slope)


def bb_width_percentile(bars: Sequence[Bar], lookback: int = 120) -> float | None:
    """Percentile of the current Bollinger width within the recent window.

    High percentile = expansion (breakout conditions). Low = squeeze.
    """
    if len(bars) < 30:
        return None
    closes = _col(bars, "close")
    up, mid, low = ind.bollinger_bands(closes, 20, 2.0)
    widths = np.where((mid > 0) & np.isfinite(mid), (up - low) / mid, np.nan)
    window = widths[-lookback:]
    current = widths[-1]
    if not np.isfinite(current):
        return None
    valid = window[np.isfinite(window)]
    if valid.size < 10:
        return None
    return float((valid <= current).sum() / valid.size * 100.0)


def realized_volatility_pct(bars: Sequence[Bar], lookback: int = 20) -> float | None:
    """Annualised-ish stdev of recent log returns, in percent."""
    if len(bars) < lookback + 1:
        return None
    closes = _col(bars, "close")
    window = closes[-(lookback + 1) :]
    if np.any(window <= 0):
        return None
    rets = np.diff(np.log(window))
    return float(np.std(rets, ddof=1) * np.sqrt(252) * 100.0)


def returns_series(bars: Sequence[Bar]) -> np.ndarray:
    closes = _col(bars, "close")
    if closes.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(closes) / closes[:-1]


def correlation(a: Sequence[Bar], b: Sequence[Bar]) -> float | None:
    """Pearson correlation of returns. Requires aligned lengths."""
    n = min(len(a), len(b))
    if n < 20:
        return None
    ra = returns_series(a[-n:])
    rb = returns_series(b[-n:])
    m = min(ra.size, rb.size)
    if m < 20:
        return None
    x, y = ra[-m:], rb[-m:]
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


__all__ = [
    "bb_width_percentile",
    "compute_indicators",
    "correlation",
    "obv_slope_pct",
    "realized_volatility_pct",
    "returns_series",
]
