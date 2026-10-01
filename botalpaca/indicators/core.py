"""Technical indicators.

Pure functions over numpy arrays / pandas Series. No I/O, no globals, no
hidden state, so each indicator is directly unit-testable and deterministic.

Conventions
-----------
* Inputs are float64 numpy arrays ordered oldest -> newest.
* Indicators that need a warm-up period return ``None`` for the leading values
  rather than emitting a misleading number. Callers must treat ``None`` as
  "not enough data", never as zero.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

__all__ = [
    "adx",
    "atr",
    "bollinger_bands",
    "ema",
    "highest",
    "linear_slope",
    "macd",
    "mean",
    "obv",
    "roc",
    "rsi",
    "sma",
    "std",
    "stochastic",
    "true_range",
    "vwap",
    "vwap_session",
]

ArrayLike = np.ndarray | list[float] | pd.Series


def _as_array(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("indicator inputs must be one-dimensional")
    return arr


def sma(values: ArrayLike, period: int) -> np.ndarray:
    """Simple moving average with NaN warm-up."""
    arr = _as_array(values)
    if period <= 0:
        raise ValueError("period must be positive")
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size < period:
        return out
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def ema(values: ArrayLike, period: int) -> np.ndarray:
    """Exponential moving average seeded with the first ``period`` SMA.

    Seeding with an SMA (rather than the first close) removes the bias that
    makes a naive EMA diverge from chart-platform values.
    """
    arr = _as_array(values)
    if period <= 0:
        raise ValueError("period must be positive")
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size < period:
        return out
    alpha = 2.0 / (period + 1.0)
    seed = float(np.mean(arr[:period]))
    out[period - 1] = seed
    prev = seed
    for i in range(period, arr.size):
        prev = alpha * arr[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def rsi(values: ArrayLike, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index."""
    arr = _as_array(values)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size <= period:
        return out
    delta = np.diff(arr)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, arr.size):
        if i > period:
            idx = i - 1
            avg_gain = (avg_gain * (period - 1) + gains[idx]) / period
            avg_loss = (avg_loss * (period - 1) + losses[idx]) / period
        if avg_loss == 0.0:
            out[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd(
    values: ArrayLike, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(macd_line, signal_line, histogram)``."""
    arr = _as_array(values)
    if arr.size < slow + signal:
        empty = np.full(arr.shape, np.nan, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    ema_fast = ema(arr, fast)
    ema_slow = ema(arr, slow)
    macd_line = ema_fast - ema_slow
    valid = ~np.isnan(macd_line)
    signal_line = np.full(arr.shape, np.nan, dtype=np.float64)
    # The signal EMA must be seeded from the first *valid* macd value.
    first_valid = int(np.argmax(valid))
    macd_valid = macd_line[first_valid:]
    sig = ema(macd_valid, signal)
    signal_line[first_valid:] = sig
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(
    high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> np.ndarray:
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    if not (h.size == l.size == c.size):
        raise ValueError("high, low and close must have the same length")
    out = np.full(h.shape, np.nan, dtype=np.float64)
    if h.size == 0:
        return out
    out[0] = h[0] - l[0]
    if h.size > 1:
        prev_close = c[:-1]
        out[1:] = np.maximum.reduce(
            [h[1:] - l[1:], np.abs(h[1:] - prev_close), np.abs(l[1:] - prev_close)]
        )
    return out


def atr(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> np.ndarray:
    """Wilder's Average True Range."""
    tr = true_range(high, low, close)
    out = np.full(tr.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(tr)
    if not valid.any():
        return out
    first = int(np.argmax(valid))
    tr_valid = tr[first:]
    if tr_valid.size < period:
        return out
    prev = float(np.mean(tr_valid[:period]))
    out[first + period - 1] = prev
    for i in range(period, tr_valid.size):
        prev = (prev * (period - 1) + tr_valid[i]) / period
        out[first + i] = prev
    return out


def adx(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder's ADX. Returns ``(adx, plus_di, minus_di)``."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = h.size
    adx_out = np.full(n, np.nan, dtype=np.float64)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    if n < period * 2 + 1:
        return adx_out, plus_di, minus_di

    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(h, l, c)[1:]

    def _wilder(vals: np.ndarray) -> np.ndarray:
        """Wilder smoothing: first value is a sum, then recursive average."""
        out = np.full(vals.size, np.nan, dtype=np.float64)
        if vals.size < period:
            return out
        prev = float(np.sum(vals[:period]))
        out[period - 1] = prev
        for i in range(period, vals.size):
            prev = prev - (prev / period) + vals[i]
            out[i] = prev
        return out

    tr_s = _wilder(tr)
    plus_s = _wilder(plus_dm)
    minus_s = _wilder(minus_dm)

    # DX is defined from the first smoothed value onward (bar index `period`).
    dx_series = np.full(n, np.nan, dtype=np.float64)
    for i in range(tr_s.size):
        if np.isnan(tr_s[i]) or tr_s[i] == 0:
            continue
        pdi = 100.0 * (plus_s[i] / tr_s[i])
        mdi = 100.0 * (minus_s[i] / tr_s[i])
        plus_di[i + 1] = pdi
        minus_di[i + 1] = mdi
        denom = pdi + mdi
        dx_series[i + 1] = 100.0 * abs(pdi - mdi) / denom if denom != 0 else 0.0

    # ADX is the Wilder-smoothed DX; it needs a second full period of DX.
    valid_dx = dx_series[~np.isnan(dx_series)]
    if valid_dx.size < period:
        return adx_out, plus_di, minus_di
    prev = float(np.mean(valid_dx[:period]))
    first = period * 2  # bar index where the first ADX value becomes valid
    if first >= n:
        return adx_out, plus_di, minus_di
    adx_out[first] = prev
    for j in range(period, valid_dx.size):
        prev = (prev * (period - 1) + valid_dx[j]) / period
        idx = first + j - period + 1
        if idx < n:
            adx_out[idx] = prev
    return adx_out, plus_di, minus_di


def bollinger_bands(
    values: ArrayLike, period: int = 20, std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(upper, middle, lower)``."""
    arr = _as_array(values)
    n = arr.size
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return upper, middle, lower
    for i in range(period - 1, n):
        window = arr[i - period + 1 : i + 1]
        mean = float(np.mean(window))
        sd = float(np.std(window, ddof=0))
        middle[i] = mean
        upper[i] = mean + std_dev * sd
        lower[i] = mean - std_dev * sd
    return upper, middle, lower


def obv(close: ArrayLike, volume: ArrayLike) -> np.ndarray:
    """On-Balance Volume."""
    c, v = _as_array(close), _as_array(volume)
    if c.size != v.size:
        raise ValueError("close and volume must have the same length")
    out = np.zeros(c.size, dtype=np.float64)
    if c.size < 2:
        return out
    sign = np.sign(np.diff(c))
    out[1:] = np.cumsum(sign * v[1:])
    return out


def roc(values: ArrayLike, period: int = 10) -> np.ndarray:
    """Rate of Change, in percent."""
    arr = _as_array(values)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size <= period:
        return out
    prev = arr[:-period]
    base = np.where(prev == 0, np.nan, prev)
    out[period:] = ((arr[period:] - base) / base) * 100.0
    return out


def stochastic(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14, smooth: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(%K, %D)``."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = h.size
    k = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = float(np.max(h[i - period + 1 : i + 1]))
        ll = float(np.min(l[i - period + 1 : i + 1]))
        rng = hh - ll
        k[i] = 50.0 if rng == 0 else (c[i] - ll) / rng * 100.0
    d = sma(np.nan_to_num(k, nan=np.nan), smooth)
    d = np.where(np.isnan(k), np.nan, d)
    return k, d


def vwap(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike
) -> np.ndarray:
    """Cumulative VWAP over the whole series.

    This is a session-anchored VWAP. It is only meaningful on a series that
    starts at a session boundary; the scanner therefore anchors it per
    session, and the value is reported as ``None`` when the caller cannot
    guarantee that anchoring.
    """
    h, l, c, v = _as_array(high), _as_array(low), _as_array(close), _as_array(volume)
    if not (h.size == l.size == c.size == v.size):
        raise ValueError("vwap inputs must have the same length")
    n = h.size
    out = np.full(n, np.nan, dtype=np.float64)
    cum_pv = 0.0
    cum_vol = 0.0
    for i in range(n):
        typical = (h[i] + l[i] + c[i]) / 3.0
        cum_pv += typical * v[i]
        cum_vol += v[i]
        out[i] = cum_pv / cum_vol if cum_vol > 0 else np.nan
    return out


def vwap_session(
    timestamps: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
) -> np.ndarray:
    """Session-anchored VWAP: resets at each calendar day boundary."""
    out: list[float] = []
    cum_pv = 0.0
    cum_vol = 0.0
    current_day: dt.date | None = None
    h, l, c, v = _as_array(high), _as_array(low), _as_array(close), _as_array(volume)
    for i in range(len(timestamps)):
        stamp = timestamps[i]
        day = stamp.date() if isinstance(stamp, dt.datetime) else dt.date.fromisoformat(str(stamp)[:10])
        if day != current_day:
            current_day = day
            cum_pv = 0.0
            cum_vol = 0.0
        typical = (h[i] + l[i] + c[i]) / 3.0
        cum_pv += typical * v[i]
        cum_vol += v[i]
        out.append(cum_pv / cum_vol if cum_vol > 0 else float("nan"))
    return np.asarray(out, dtype=np.float64)


def highest(values: ArrayLike, period: int) -> np.ndarray:
    arr = _as_array(values)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    for i in range(period - 1, arr.size):
        out[i] = float(np.max(arr[i - period + 1 : i + 1]))
    return out


def lowest(values: ArrayLike, period: int) -> np.ndarray:
    arr = _as_array(values)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    for i in range(period - 1, arr.size):
        out[i] = float(np.min(arr[i - period + 1 : i + 1]))
    return out


def mean(values: ArrayLike) -> float:
    arr = _as_array(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def std(values: ArrayLike) -> float:
    arr = _as_array(values)
    return float(np.std(arr, ddof=0)) if arr.size else float("nan")


def linear_slope(values: ArrayLike, lookback: int | None = None) -> float:
    """Least-squares slope normalized per bar. Positive = rising.

    Used to detect EMA slope decay, which is an early momentum warning.
    """
    arr = _as_array(values)
    if arr.size < 2:
        return 0.0
    data = arr[-lookback:] if lookback else arr
    data = data[~np.isnan(data)]
    if data.size < 2:
        return 0.0
    x = np.arange(data.size, dtype=np.float64)
    denom = ((x - x.mean()) ** 2).sum()
    if denom == 0:
        return 0.0
    slope = ((x - x.mean()) * (data - data.mean())).sum() / denom
    scale = float(np.mean(np.abs(data)))
    if scale == 0:
        return 0.0
    return float(slope / scale * 100.0)
