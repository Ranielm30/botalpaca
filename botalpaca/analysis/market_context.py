"""Market context: benchmarks, sector, relative strength, market regime.

Market data is shared across PAPER and REAL deliberately — a chart of SPY is
identical for both accounts. Only financial *state* must be isolated, and
that isolation lives in the trading layer, not here.
"""

from __future__ import annotations

from collections.abc import Sequence

from botalpaca.domain import Bar, IndicatorSet, MarketContext, MarketRegime, SignalDirection

__all__ = [
    "INDEX_SYMBOLS",
    "SECTOR_ETFS",
    "SYMBOL_SECTOR_MAP",
    "build_market_context",
    "compute_correlation",
    "compute_relative_strength",
    "sector_for_symbol",
]

# Primary sector proxy per stock. Used for sector exposure limits and
# sector-level confluence. Unknown symbols map to "UNKNOWN", which the risk
# engine treats as its own concentration bucket.
SYMBOL_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "AMD": "Technology", "INTC": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology",
    "CSCO": "Technology", "QCOM": "Technology", "MU": "Technology",
    "TXN": "Technology", "AMAT": "Technology", "PLTR": "Technology",
    "NOW": "Technology", "INTU": "Technology", "IBM": "Technology",
    "SMCI": "Technology", "ARM": "Technology", "PANW": "Technology",
    # Communication Services
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "DIS": "Communication", "CMCSA": "Communication", "T": "Communication",
    "VZ": "Communication", "TMUS": "Communication", "EA": "Communication",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "UBER": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    # Consumer Staples
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "PM": "Consumer Staples",
    # Health Care
    "UNH": "Health Care", "LLY": "Health Care", "PFE": "Health Care",
    "MRK": "Health Care", "TMO": "Health Care", "ABBV": "Health Care",
    "ISRG": "Health Care", "AMGN": "Health Care", "JNJ": "Health Care",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "WFC": "Financials", "V": "Financials", "MA": "Financials",
    "AXP": "Financials", "BLK": "Financials", "SCHW": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "PSX": "Energy",
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "GE": "Industrials",
    "BA": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "UNP": "Industrials",
    # Utilities / Real Estate / Materials
    "NEE": "Utilities", "DUK": "Utilities",
    "PLD": "Real Estate", "AMT": "Real Estate", "SPG": "Real Estate",
    "LIN": "Materials", "SHW": "Materials", "FCX": "Materials",
    # Crypto-linked / thematic
    "COIN": "Crypto", "MSTR": "Crypto", "MARA": "Crypto",
    "ARKK": "Thematic", "SMH": "Semiconductors", "SOXX": "Semiconductors",
}

SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLC": "Communication",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLV": "Health Care",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "SMH": "Semiconductors",
}

INDEX_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"}

_BENCHMARK_ADX_TRENDING = 25.0
_HIGH_ATR_PCT = 4.0
_RANGING_ADX = 20.0


def sector_for_symbol(symbol: str) -> str:
    """Best-effort sector classification. ETF map first, then the stock map."""
    sym = symbol.upper()
    if sym in SECTOR_ETFS:
        return SECTOR_ETFS[sym]
    if sym in SYMBOL_SECTOR_MAP:
        return SYMBOL_SECTOR_MAP[sym]
    if sym in INDEX_SYMBOLS:
        return "Index"
    return "UNKNOWN"


def _regime_from(indicators: IndicatorSet) -> MarketRegime:
    adx = indicators.adx_14 or 0.0
    atr_pct = indicators.atr_pct or 0.0
    if adx >= _BENCHMARK_ADX_TRENDING and indicators.close > (indicators.ema_50 or 0.0):
        return MarketRegime.TRENDING_UP
    if adx >= _BENCHMARK_ADX_TRENDING and indicators.close < (indicators.ema_50 or 0.0):
        return MarketRegime.TRENDING_DOWN
    if atr_pct >= _HIGH_ATR_PCT:
        return MarketRegime.HIGH_VOLATILITY
    if adx < _RANGING_ADX:
        return MarketRegime.RANGING
    return MarketRegime.UNKNOWN


def _trend_from(indicators: IndicatorSet) -> SignalDirection | None:
    ema50, ema200 = indicators.ema_50, indicators.ema_200
    if ema50 is None or ema200 is None:
        return None
    if indicators.close > ema50 > ema200:
        return SignalDirection.LONG
    if indicators.close < ema50 < ema200:
        return SignalDirection.SHORT
    return None


def compute_relative_strength(
    symbol_indicators: IndicatorSet, benchmark_indicators: IndicatorSet
) -> float | None:
    """Percentage-point difference in 10-bar momentum vs the benchmark."""
    sym_roc = symbol_indicators.roc_10
    bench_roc = benchmark_indicators.roc_10
    if sym_roc is None or bench_roc is None:
        return None
    return float(sym_roc - bench_roc)


def compute_correlation(
    symbol_bars: Sequence[Bar], benchmark_bars: Sequence[Bar]
) -> float | None:
    """Correlation of returns over the overlapping window."""
    from botalpaca.indicators.service import correlation as _corr

    return _corr(symbol_bars, benchmark_bars)


def build_market_context(
    *,
    symbol_indicators: IndicatorSet,
    benchmark_symbol: str = "SPY",
    benchmark_indicators: IndicatorSet | None = None,
    benchmark_bars: Sequence[Bar] | None = None,
    sector_name: str | None = None,
    sector_indicators: IndicatorSet | None = None,
    relative_strength: float | None = None,
    correlation: float | None = None,
    market_open: bool = False,
) -> MarketContext:
    """Assemble broad-market context for one symbol.

    Missing benchmark or sector data *degrades* the context instead of
    inventing a value: an unknown market regime must never be silently
    treated as neutral, because the confluence engine would then score
    long trades in a falling market as if the backdrop supported them.
    """
    ctx = MarketContext(
        benchmark=benchmark_symbol.upper(),
        is_market_open=market_open,
        sector=sector_name,
        correlation_to_benchmark=correlation,
        rs_vs_benchmark=relative_strength,
        market_volatility_pct=symbol_indicators.atr_pct,
    )
    notes: list[str] = []

    if benchmark_indicators is not None:
        ctx.benchmark_trend = _trend_from(benchmark_indicators)
        ctx.benchmark_regime = _regime_from(benchmark_indicators)
        if benchmark_bars and len(benchmark_bars) >= 2 and benchmark_bars[-2].close > 0:
            ctx.benchmark_change_pct = (
                (benchmark_indicators.close - benchmark_bars[-2].close)
                / benchmark_bars[-2].close
                * 100.0
            )
        trend_label = {
            SignalDirection.LONG: "alcista",
            SignalDirection.SHORT: "bajista",
            None: "sin dirección clara",
        }[ctx.benchmark_trend]
        notes.append(f"{ctx.benchmark} en tendencia {trend_label}")
        if ctx.benchmark_regime == MarketRegime.HIGH_VOLATILITY:
            notes.append("Benchmark con volatilidad elevada")
    else:
        notes.append(f"Datos de {benchmark_symbol.upper()} no disponibles")

    if sector_indicators is not None:
        ctx.sector_trend = _trend_from(sector_indicators)
    elif sector_name:
        notes.append(f"Sector {sector_name} sin datos propios")

    if relative_strength is not None:
        if relative_strength > 1.0:
            notes.append(f"Fuerza relativa positiva (+{relative_strength:.1f}% vs {ctx.benchmark})")
        elif relative_strength < -1.0:
            notes.append(f"Fuerza relativa negativa ({relative_strength:.1f}% vs {ctx.benchmark})")

    if correlation is not None and abs(correlation) > 0.7:
        notes.append(f"Alta correlación con {ctx.benchmark} ({correlation:.2f})")

    ctx.notes = notes
    return ctx
