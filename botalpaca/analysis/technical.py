"""Technical analysis: turns an IndicatorSet into directional states.

Each ``analyze_*`` function is pure and returns a state object with a
0..1 ``strength``/``score`` plus human-readable ``notes`` that end up in the
Telegram explanation. The notes are not decoration: they are the audit trail
for why the system believed what it believed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from botalpaca.domain import (
    Bar,
    IndicatorSet,
    MarketRegime,
    MomentumState,
    SignalDirection,
    StructureLevel,
    StructureState,
    TrendState,
    VolatilityState,
    VolumeState,
)
from botalpaca.indicators import core as ind

__all__ = [
    "analyze_momentum",
    "analyze_structure",
    "analyze_trend",
    "analyze_volatility",
    "analyze_volume",
    "classify_swing_structure",
    "detect_gaps",
    "find_pivots",
]

# Tunable thresholds live here as named constants, not inline magic numbers.
ADX_TRENDING = 25.0
ADX_STRONG = 40.0
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
RSI_NEUTRAL_LOW = 45.0
RSI_NEUTRAL_HIGH = 55.0
VOLUME_SPIKE = 1.5
VOLUME_DRY = 0.7
BB_SQUEEZE_PERCENTILE = 25.0
BB_EXPANSION_PERCENTILE = 80.0
SQUEEZE_CURRENT_PERCENTILE = 30.0
EXPANSION_CURRENT_PERCENTILE = 75.0
HIGH_ATR_PCT = 4.0
LOW_ATR_PCT = 1.2


def _safe(value: float | None, default: float = 0.0) -> float:
    return default if value is None or not np.isfinite(value) else value


# Relative tolerance used when ordering two averages. Without it, floating
# point noise makes a perfectly flat series look like it has a bearish EMA
# stack (ema_200 == 100.0000000000001 > ema_50 == 100.0), which would emit a
# spurious directional bias.
REL_TOLERANCE = 1e-9


def _above(fast: float | None, slow: float | None) -> bool:
    """True when ``fast`` is meaningfully above ``slow`` (tolerant to FP noise)."""
    return _compare(fast, slow) > 0


def _compare(fast: float | None, slow: float | None) -> int:
    """Three-way comparison: +1 above, -1 below, 0 equal/indeterminate.

    Returning 0 for "equal" matters: a flat series must not accumulate
    bearish votes just because ``close > ema`` is false when the two are
    numerically identical.
    """
    if fast is None or slow is None or not np.isfinite(fast) or not np.isfinite(slow):
        return 0
    if slow == 0.0:
        return 1 if fast > 0.0 else 0
    delta = fast - slow
    if delta > abs(slow) * REL_TOLERANCE:
        return 1
    if delta < -abs(slow) * REL_TOLERANCE:
        return -1
    return 0


def _ema_alignment(ind_set: IndicatorSet) -> int:
    """Score EMA stack ordering from -4 (fully bearish) to +4 (fully bullish)."""
    close = ind_set.close
    pairs = [
        (ind_set.ema_9, ind_set.ema_20),
        (ind_set.ema_20, ind_set.ema_50),
        (ind_set.ema_50, ind_set.ema_200),
        (ind_set.close, ind_set.ema_20),
    ]
    score = 0
    for fast, slow in pairs:
        if fast is None or slow is None or slow == 0:
            continue
        if _above(fast, slow):
            score += 1
        elif _above(slow, fast):
            score -= 1
    _ = close
    return score


def analyze_trend(ind_set: IndicatorSet, bars: Sequence[Bar] | None = None) -> TrendState:
    """EMA stack, price location vs averages, slopes, crossovers, structure."""
    state = TrendState()
    notes: list[str] = []

    state.ema_alignment = _ema_alignment(ind_set)
    close = ind_set.close

    for attr, avg in (
        ("price_above_ema20", ind_set.ema_20),
        ("price_above_ema50", ind_set.ema_50),
        ("price_above_ema200", ind_set.ema_200),
    ):
        comparison = _compare(close, avg)
        # None (rather than False) when price sits exactly on the average, so
        # the vote loop can tell "not above" from "indifferent".
        setattr(state, attr, None if comparison == 0 else comparison > 0)

    if bars:
        closes = np.asarray([b.close for b in bars], dtype=np.float64)
        ema20_series = ind.ema(closes, 20)
        ema50_series = ind.ema(closes, 50)
        state.ema20_slope = ind.linear_slope(ema20_series, lookback=10)
        state.ema50_slope = ind.linear_slope(ema50_series, lookback=10)
        # Detect crossovers in the recent window.
        lookback = min(30, closes.size - 1)
        if lookback > 5:
            recent20 = ema20_series[-lookback:]
            recent50 = ema50_series[-lookback:]
            valid = ~np.isnan(recent20) & ~np.isnan(recent50)
            if valid.sum() >= 2:
                diff = recent20[valid] - recent50[valid]
                signs = np.sign(diff)
                cross_up = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]
                cross_dn = np.where((signs[:-1] >= 0) & (signs[1:] < 0))[0]
                state.golden_cross = bool(cross_up.size)
                state.death_cross = bool(cross_dn.size)
                if cross_up.size and cross_up[-1] >= len(signs) - 3:
                    notes.append("Cruce dorado EMA20/EMA50 reciente")
                if cross_dn.size and cross_dn[-1] >= len(signs) - 3:
                    notes.append("Cruce de muerte EMA20/EMA50 reciente")

    if ind_set.ema_50 is not None and ind_set.ema_200 is not None:
        if _above(ind_set.ema_50, ind_set.ema_200):
            notes.append("EMA50 sobre EMA200 (estructura alcista)")
        elif _above(ind_set.ema_200, ind_set.ema_50):
            notes.append("EMA50 bajo EMA200 (estructura bajista)")
        else:
            notes.append("EMA50 y EMA200 planas")

    if bars:
        state.structure = classify_swing_structure(bars)

    # Direction decision.
    bullish_votes = 0
    bearish_votes = 0
    for flag in (state.price_above_ema20, state.price_above_ema50, state.price_above_ema200):
        if flag:
            bullish_votes += 1
        elif flag is False:
            bearish_votes += 1
    if state.ema_alignment > 0:
        bullish_votes += 1
    elif state.ema_alignment < 0:
        bearish_votes += 1
    if _safe(state.ema20_slope) > 0.01:
        bullish_votes += 1
    elif _safe(state.ema20_slope) < -0.01:
        bearish_votes += 1

    if bullish_votes > bearish_votes:
        state.direction = SignalDirection.LONG
    elif bearish_votes > bullish_votes:
        state.direction = SignalDirection.SHORT
    else:
        # A tie carries no directional information. Leaving ``direction`` as
        # None forces downstream strategies to abstain instead of guessing.
        state.direction = None

    # Strength: vote dominance blended with ADX conviction.
    total_votes = bullish_votes + bearish_votes
    dominance = abs(bullish_votes - bearish_votes) / total_votes if total_votes else 0.0
    adx = _safe(ind_set.adx_14)
    adx_conviction = min(1.0, adx / ADX_STRONG) if adx > 0 else 0.0
    state.strength = float(np.clip(0.6 * dominance + 0.4 * adx_conviction, 0.0, 1.0))

    state.notes = notes
    return state


def analyze_momentum(ind_set: IndicatorSet, bars: Sequence[Bar] | None = None) -> MomentumState:
    """RSI, MACD histogram, ADX, ROC, and divergence detection."""
    state = MomentumState()
    notes: list[str] = []
    rsi_val = ind_set.rsi_14

    if rsi_val is not None:
        state.rsi = rsi_val
        if rsi_val >= RSI_OVERBOUGHT:
            state.rsi_state = "overbought"
            notes.append(f"RSI {rsi_val:.1f} en sobrecompra")
        elif rsi_val <= RSI_OVERSOLD:
            state.rsi_state = "oversold"
            notes.append(f"RSI {rsi_val:.1f} en sobreventa")
        elif rsi_val >= RSI_NEUTRAL_HIGH:
            state.rsi_state = "bullish"
        elif rsi_val <= RSI_NEUTRAL_LOW:
            state.rsi_state = "bearish"
        else:
            state.rsi_state = "neutral"

    hist = ind_set.macd_hist
    if hist is not None and ind_set.macd is not None:
        state.macd_hist = hist
        if hist > 0:
            state.macd_state = "positive"
            notes.append("Histograma MACD positivo")
        else:
            state.macd_state = "negative"
            notes.append("Histograma MACD negativo")

    adx = ind_set.adx_14
    if adx is not None:
        state.adx = adx
        if adx >= ADX_STRONG:
            state.adx_state = "strong"
            notes.append(f"ADX {adx:.1f} tendencia fuerte")
        elif adx >= ADX_TRENDING:
            state.adx_state = "trending"
        else:
            state.adx_state = "ranging"
            notes.append(f"ADX {adx:.1f} mercado lateral")

    state.roc = ind_set.roc_10

    # Divergence: price makes a new extreme but momentum does not follow.
    if bars and len(bars) >= 60 and rsi_val is not None:
        state.divergence = _detect_divergence(bars)
        if state.divergence:
            notes.append(f"Divergencia {state.divergence}")

    # Momentum score 0..1: combine RSI positioning, MACD sign, ADX.
    rsi_component = 0.0
    if rsi_val is not None:
        rsi_component = np.clip((rsi_val - 30.0) / 40.0, 0.0, 1.0)
    macd_component = 1.0 if (hist is not None and hist > 0) else 0.0
    adx_component = min(1.0, _safe(adx) / ADX_STRONG) if adx else 0.0
    state.score = float(np.clip(0.4 * rsi_component + 0.3 * macd_component + 0.3 * adx_component, 0.0, 1.0))
    state.notes = notes
    return state


def _detect_divergence(bars: Sequence[Bar]) -> str | None:
    closes = np.asarray([b.close for b in bars], dtype=np.float64)
    rsi_series = ind.rsi(closes, 14)
    pivots = find_pivots(closes, left=5, right=5)
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]
    if len(highs) >= 2:
        p1, p2 = highs[-2], highs[-1]
        r1, r2 = rsi_series[p1.index], rsi_series[p2.index]
        if np.isfinite(r1) and np.isfinite(r2) and p2.price > p1.price and r2 < r1:
            return "bajista"
    if len(lows) >= 2:
        p1, p2 = lows[-2], lows[-1]
        r1, r2 = rsi_series[p1.index], rsi_series[p2.index]
        if np.isfinite(r1) and np.isfinite(r2) and p2.price < p1.price and r2 > r1:
            return "alcista"
    return None


def analyze_volume(ind_set: IndicatorSet, obv_slope: float | None = None) -> VolumeState:
    """Relative volume, spike/dry classification, OBV confirmation."""
    state = VolumeState()
    notes: list[str] = []
    rel = ind_set.rel_volume

    if rel is not None:
        state.rel_volume = rel
        if rel >= VOLUME_SPIKE:
            state.volume_state = "spike"
            notes.append(f"Volumen {rel:.2f}x promedio (spike)")
        elif rel <= VOLUME_DRY:
            state.volume_state = "dry"
            notes.append(f"Volumen {rel:.2f}x promedio (seco)")
        else:
            state.volume_state = "normal"

    if obv_slope is not None:
        state.obv_slope = obv_slope
        state.obv_confirming = obv_slope > 0

    # Volume score: spike confirmation is good, but a spike against the
    # direction of the move is distribution, not confirmation.
    spike_component = 0.0
    if rel is not None:
        if rel >= VOLUME_SPIKE:
            spike_component = min(1.0, (rel - VOLUME_SPIKE) / VOLUME_SPIKE + 0.5)
        else:
            spike_component = 0.35
    obv_component = 0.5
    if obv_slope is not None:
        obv_component = float(np.clip(0.5 + obv_slope, 0.0, 1.0))
    state.score = float(np.clip(0.6 * spike_component + 0.4 * obv_component, 0.0, 1.0))
    state.notes = notes
    return state


def analyze_volatility(
    ind_set: IndicatorSet,
    *,
    bb_percentile: float | None = None,
    realized_vol: float | None = None,
) -> VolatilityState:
    """ATR, Bollinger width, squeeze vs expansion, and regime classification."""
    state = VolatilityState()
    notes: list[str] = []
    state.atr = ind_set.atr_14
    state.atr_pct = ind_set.atr_pct
    state.bb_width = ind_set.bb_width
    state.bb_width_pctile = bb_percentile

    if bb_percentile is not None:
        if bb_percentile <= SQUEEZE_CURRENT_PERCENTILE:
            state.squeeze = True
            notes.append(f"Bollinger estrecho (percentil {bb_percentile:.0f}) - squeeze")
        elif bb_percentile >= EXPANSION_CURRENT_PERCENTILE:
            state.expansion = True
            notes.append(f"Bollinger expandido (percentil {bb_percentile:.0f})")

    atr_pct = _safe(ind_set.atr_pct)
    adx = _safe(ind_set.adx_14)

    # Regime classification: directional strength first, then vol character.
    above_ema50 = (ind_set.ema_50 is not None and ind_set.close > ind_set.ema_50)
    below_ema50 = (ind_set.ema_50 is not None and ind_set.close < ind_set.ema_50)
    if adx >= ADX_TRENDING and above_ema50:
        state.regime = MarketRegime.TRENDING_UP
    elif adx >= ADX_TRENDING and below_ema50:
        state.regime = MarketRegime.TRENDING_DOWN
    elif atr_pct >= HIGH_ATR_PCT:
        state.regime = MarketRegime.HIGH_VOLATILITY
        notes.append("Régimen: alta volatilidad")
    elif adx < ADX_TRENDING and atr_pct <= LOW_ATR_PCT:
        state.regime = MarketRegime.LOW_VOLATILITY
        notes.append("Régimen: baja volatilidad (compresión)")
    elif adx < ADX_TRENDING:
        state.regime = MarketRegime.RANGING
        notes.append("Régimen: lateral")

    vol_component = 0.5
    if bb_percentile is not None:
        # Mid-range width is "healthy"; extremes reduce confidence.
        vol_component = float(np.clip(1.0 - abs(bb_percentile - 50.0) / 50.0, 0.0, 1.0))
    atr_component = 0.5
    if atr_pct:
        atr_component = float(np.clip(1.0 - abs(atr_pct - 2.5) / 5.0, 0.0, 1.0))
    state.score = float(np.clip(0.5 * vol_component + 0.5 * atr_component, 0.0, 1.0))
    if realized_vol:
        notes.append(f"Volatilidad realizada anualizada {realized_vol:.1f}%")
    state.notes = notes
    return state


def find_pivots(prices: np.ndarray, left: int = 5, right: int = 5):  # noqa: ANN202
    """Fractal pivots: a bar higher/lower than ``left`` bars each side.

    Returns pivot objects with ``.index``, ``.price``, ``.kind``.
    """
    from botalpaca.domain import StructureLevel  # noqa: F401  (re-export convenience)

    n = prices.size
    pivots: list[_Pivot] = []
    for i in range(left, n - right):
        window = prices[i - left : i + right + 1]
        center = prices[i]
        if center >= window.max():
            pivots.append(_Pivot(i, float(center), "high"))
        if center <= window.min():
            pivots.append(_Pivot(i, float(center), "low"))
    return pivots


class _Pivot:
    __slots__ = ("index", "kind", "price")

    def __init__(self, index: int, price: float, kind: str) -> None:
        self.index = index
        self.price = price
        self.kind = kind

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Pivot({self.index}, {self.price:.2f}, {self.kind})"


def analyze_structure(
    bars: Sequence[Bar], ind_set: IndicatorSet, *, lookback: int = 120
) -> StructureState:
    """Support/resistance from swing pivots, plus breakout/retest/gap flags."""
    state = StructureState()
    notes: list[str] = []
    if len(bars) < 30:
        return state

    recent = list(bars)[-lookback:]
    closes = np.asarray([b.close for b in recent], dtype=np.float64)
    pivots = find_pivots(closes, left=4, right=4)
    current = ind_set.close
    tolerance = _safe(ind_set.atr_14) * 0.5

    supports: list[StructureLevel] = []
    resistances: list[StructureLevel] = []
    for p in pivots:
        if p.price < current:
            supports.append(
                StructureLevel(price=p.price, kind="support", touches=1, last_touch=recent[p.index].timestamp)
            )
        else:
            resistances.append(
                StructureLevel(price=p.price, kind="resistance", touches=1, last_touch=recent[p.index].timestamp)
            )

    # Merge nearby levels (within half an ATR) and count touches.
    state.supports = _merge_levels(supports, tolerance, keep="support")
    state.resistances = _merge_levels(resistances, tolerance, keep="resistance")

    if state.supports:
        notes.append(f"Soporte más cercano: ${state.supports[0].price:.2f}")
    if state.resistances:
        notes.append(f"Resistencia más cercana: ${state.resistances[0].price:.2f}")

    # Breakout: close beyond the recent 20-bar extreme on expanding volume.
    if ind_set.highest_20 and ind_set.lowest_20:
        if current > ind_set.highest_20:
            state.breakout = True
            state.breakout_direction = SignalDirection.LONG
            notes.append("Ruptura alcista del rango de 20 velas")
        elif current < ind_set.lowest_20:
            state.breakout = True
            state.breakout_direction = SignalDirection.SHORT
            notes.append("Ruptura bajista del rango de 20 velas")

    # Retest: after a breakout, price pulls back toward the broken level.
    if state.breakout:
        level = ind_set.highest_20 if state.breakout_direction == SignalDirection.LONG else ind_set.lowest_20
        if level is not None and abs(current - level) <= _safe(ind_set.atr_14):
            state.retest = True
            notes.append("Retest del nivel roto")

    # In range: price between established support and resistance.
    if state.supports and state.resistances:
        state.in_range = state.supports[0].price < current < state.resistances[0].price

    gaps = detect_gaps(recent)
    state.gap_up = gaps["up"] > 0
    state.gap_down = gaps["down"] > 0
    if gaps["up"] > 0:
        notes.append(f"{gaps['up']} gap(s) alcista(s) reciente(s)")
    if gaps["down"] > 0:
        notes.append(f"{gaps['down']} gap(s) bajista(s) reciente(s)")

    state.notes = notes
    return state


def _merge_levels(levels: list[StructureLevel], tolerance: float, *, keep: str) -> list[StructureLevel]:
    """Cluster pivots into distinct levels, sorted by distance from price."""
    if not levels:
        return []
    ordered = sorted(levels, key=lambda lv: lv.price)
    merged: list[StructureLevel] = [ordered[0]]
    tol = tolerance if tolerance > 0 else 0.01
    for lv in ordered[1:]:
        last = merged[-1]
        if abs(lv.price - last.price) <= tol:
            touches = last.touches + lv.touches
            last.price = (last.price * last.touches + lv.price * lv.touches) / touches
            last.touches = touches
        else:
            merged.append(lv)
    # Support: nearest below (highest price). Resistance: nearest above (lowest).
    if keep == "support":
        return sorted(merged, key=lambda lv: -lv.price)
    return sorted(merged, key=lambda lv: lv.price)


def detect_gaps(bars: Sequence[Bar], min_pct: float = 0.2) -> dict[str, int]:
    """Count gap-up and gap-down events by comparing consecutive opens."""
    up = 0
    down = 0
    for prev, cur in zip(bars, bars[1:], strict=False):
        if prev.close <= 0:
            continue
        gap_pct = (cur.open - prev.close) / prev.close * 100.0
        if gap_pct >= min_pct:
            up += 1
        elif gap_pct <= -min_pct:
            down += 1
    return {"up": up, "down": down}


def classify_swing_structure(bars: Sequence[Bar]) -> str:
    """Label the swing structure as HH/HL, LH/LL, or mixed."""
    if len(bars) < 30:
        return "indefinida"
    recent = list(bars)[-60:]
    closes = np.asarray([b.close for b in recent], dtype=np.float64)
    highs = [p for p in find_pivots(closes, 4, 4) if p.kind == "high"]
    lows = [p for p in find_pivots(closes, 4, 4) if p.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "indefinida"
    higher_high = highs[-1].price > highs[-2].price
    higher_low = lows[-1].price > lows[-2].price
    if higher_high and higher_low:
        return "HH/HL"
    if not higher_high and not higher_low:
        return "LH/LL"
    return "mixta"
