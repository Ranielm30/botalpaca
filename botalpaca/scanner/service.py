"""Layer 5 — Market scanner.

Scans the configured universe, runs the whole analysis pipeline per symbol and
returns *ranked* opportunities rather than a flat list. The ranking is the
product: raw score alone would favour noisy micro-caps, so the final ordering
mixes the confluence score, the strategy's own conviction, the R:R, liquidity
and a recency/decay factor.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

from botalpaca.analysis import analyze_symbol
from botalpaca.analysis.market_context import SECTOR_ETFS, sector_for_symbol
from botalpaca.config import ScannerSettings, StrategySettings, get_settings
from botalpaca.config.logging import get_logger
from botalpaca.confluence import ConfluenceEngine
from botalpaca.domain import (
    DataQualityError,
    Opportunity,
    Quality,
    TechnicalSnapshot,
    TradingEnvironment,
)
from botalpaca.market import MarketDataService
from botalpaca.strategies import StrategyEngine

log = get_logger(__name__)

# Ranking weights. They must sum to 1.0.
RANK_WEIGHTS = {
    "score": 0.45,
    "rr": 0.20,
    "strategy": 0.15,
    "liquidity": 0.10,
    "risk": 0.10,
}


@dataclass
class ScannedSymbol:
    """A symbol that produced at least one signal, with its snapshot."""

    snapshot: TechnicalSnapshot
    opportunities: list[Opportunity] = field(default_factory=list)
    error: str | None = None


@dataclass
class ScanResult:
    environment: TradingEnvironment
    timeframe: str
    started_at: dt.datetime
    finished_at: dt.datetime | None = None
    scanned: int = 0
    failed: int = 0
    opportunities: list[Opportunity] = field(default_factory=list)
    snapshots: dict[str, TechnicalSnapshot] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or dt.datetime.now(dt.UTC)
        return (end - self.started_at).total_seconds()

    @property
    def tradable(self) -> list[Opportunity]:
        return [o for o in self.opportunities if o.tradable]

    def best(self) -> list[Opportunity]:
        return self.tradable[:10]


class MarketScanner:
    """Configurable universe scanner with fingerprint-based deduplication."""

    def __init__(
        self,
        market: MarketDataService,
        *,
        environment: TradingEnvironment,
        strategies: StrategyEngine | None = None,
        confluence: ConfluenceEngine | None = None,
        settings: ScannerSettings | None = None,
        strategy_settings: StrategySettings | None = None,
    ) -> None:
        self._market = market
        self.environment = environment
        self.settings = settings or get_settings().scanner
        self.strategy_settings = strategy_settings or get_settings().strategy
        self.strategies = strategies or StrategyEngine(self.strategy_settings)
        self.confluence = confluence or ConfluenceEngine(
            strategy_settings=self.strategy_settings
        )
        self._seen_fingerprints: dict[str, dt.datetime] = {}

    # ------------------------------------------------------------------ public

    async def scan(
        self,
        *,
        symbols: Sequence[str] | None = None,
        timeframe: str = "1D",
        limit: int | None = None,
        min_score: float | None = None,
        only_tradable: bool = False,
        record_history: bool = True,
    ) -> ScanResult:
        """Analyze the universe and return ranked opportunities."""
        universe = [s.upper() for s in (symbols or self.settings.universe)]
        if limit:
            universe = universe[:limit]
        threshold = (
            min_score if min_score is not None else self.strategy_settings.min_score_to_report
        )
        started = dt.datetime.now(dt.UTC)
        result = ScanResult(
            environment=self.environment, timeframe=timeframe, started_at=started
        )

        if not universe:
            result.finished_at = dt.datetime.now(dt.UTC)
            return result

        market_open = await self._market.is_market_open()
        benchmark = await self._benchmark_bundle(timeframe)
        sector_cache: dict[str, object] = {}

        semaphore = asyncio.Semaphore(max(1, self.settings.max_concurrent_analyses))

        async def one(symbol: str) -> ScannedSymbol | None:
            async with semaphore:
                return await self._analyze(
                    symbol,
                    timeframe=timeframe,
                    market_open=market_open,
                    benchmark=benchmark,
                    sector_cache=sector_cache,
                )

        gathered = await asyncio.gather(*(one(s) for s in universe), return_exceptions=True)

        opportunities: list[Opportunity] = []
        for symbol, outcome in zip(universe, gathered, strict=False):
            if isinstance(outcome, BaseException):
                result.failed += 1
                result.errors[symbol] = str(outcome)
                log.warning("scanner.symbol_failed", symbol=symbol, error=str(outcome))
                continue
            if outcome is None:
                result.failed += 1
                result.errors[symbol] = "sin datos suficientes"
                continue
            result.scanned += 1
            result.snapshots[symbol] = outcome.snapshot
            for opportunity in outcome.opportunities:
                if opportunity.score < threshold:
                    continue
                if only_tradable and not opportunity.tradable:
                    continue
                if record_history and not self._is_new(opportunity.fingerprint):
                    log.info("scanner.duplicate_skipped", symbol=symbol)
                    continue
                opportunities.append(opportunity)

        ranked = self._rank(opportunities)
        result.opportunities = ranked[: self.strategy_settings.max_signals_per_scan]
        result.finished_at = dt.datetime.now(dt.UTC)
        self._prune_fingerprints()
        log.info(
            "scanner.completed",
            scanned=result.scanned,
            failed=result.failed,
            opportunities=len(result.opportunities),
            tradable=len(result.tradable),
            duration_s=round(result.duration_seconds, 2),
            environment=self.environment.value,
        )
        return result

    async def analyze_one(
        self, symbol: str, *, timeframe: str = "1D"
    ) -> tuple[TechnicalSnapshot, list[Opportunity]] | None:
        """Full pipeline for a single symbol (used by /analizar TICKER)."""
        market_open = await self._market.is_market_open()
        benchmark = await self._benchmark_bundle(timeframe)
        scanned = await self._analyze(
            symbol.upper(),
            timeframe=timeframe,
            market_open=market_open,
            benchmark=benchmark,
            sector_cache={},
            record_history=False,
        )
        if scanned is None:
            return None
        return scanned.snapshot, self._rank(scanned.opportunities)

    async def analyze_all_timeframes(
        self, symbol: str, *, timeframes: Sequence[str] | None = None
    ) -> list[tuple[TechnicalSnapshot, list[Opportunity]]]:
        results = []
        for timeframe in timeframes or (
            self.strategy_settings.default_timeframe,
            self.strategy_settings.lower_timeframe,
            self.strategy_settings.higher_timeframe,
        ):
            outcome = await self.analyze_one(symbol, timeframe=timeframe)
            if outcome is not None:
                results.append(outcome)
        return results

    # ---------------------------------------------------------------- internals

    async def _benchmark_bundle(
        self, timeframe: str
    ) -> tuple[TechnicalSnapshot | None, object, list]:
        """Benchmark bars + indicators, shared by every symbol in the scan."""
        symbol = self.settings.benchmark
        try:
            bars = await self._market.get_bars(symbol, timeframe, limit=300)
        except Exception as exc:  # noqa: BLE001 - context is optional, not fatal
            log.warning("scanner.benchmark_unavailable", symbol=symbol, error=str(exc))
            return None, None, []
        if len(bars) < 30:
            return None, None, []
        from botalpaca.indicators import compute_indicators

        indicators = compute_indicators(bars, timeframe, min_bars=30)
        return None, indicators, bars

    async def _analyze(
        self,
        symbol: str,
        *,
        timeframe: str,
        market_open: bool,
        benchmark: tuple[TechnicalSnapshot | None, object, list],
        sector_cache: dict[str, object],
        record_history: bool = True,
    ) -> ScannedSymbol | None:
        _, benchmark_indicators, benchmark_bars = benchmark
        bars = await self._market.get_bars(symbol, timeframe, limit=300)
        if len(bars) < max(30, self.strategy_settings.min_bars_required):
            return None

        quote = None
        try:
            quote = await self._market.get_quote(symbol)
        except Exception:  # noqa: BLE001 - a missing quote only costs the spread
            log.debug("scanner.quote_unavailable", symbol=symbol)

        sector = sector_for_symbol(symbol)
        sector_indicators = None
        sector_etf = SECTOR_ETFS.get(sector)
        if sector_etf and sector_etf not in sector_cache:
            try:
                sector_bars = await self._market.get_bars(sector_etf, timeframe, limit=300)
                if len(sector_bars) >= 30:
                    from botalpaca.indicators import compute_indicators

                    sector_cache[sector_etf] = compute_indicators(
                        sector_bars, timeframe, min_bars=30
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug("scanner.sector_unavailable", sector_etf=sector_etf, error=str(exc))
        if sector_etf:
            sector_indicators = sector_cache.get(sector_etf)

        try:
            snapshot = analyze_symbol(
                symbol,
                bars,
                timeframe=timeframe,
                quote=quote,
                settings=self.strategy_settings,
                benchmark_indicators=benchmark_indicators,
                benchmark_bars=benchmark_bars or None,
                sector_indicators=sector_indicators,
                benchmark_symbol=self.settings.benchmark,
                market_open=market_open,
            )
        except DataQualityError as exc:
            log.info("scanner.insufficient_data", symbol=symbol, error=str(exc))
            return None

        await self._attach_htf(snapshot, symbol)

        signals = self.strategies.run(snapshot)
        if not signals:
            return ScannedSymbol(snapshot=snapshot)

        opportunities: list[Opportunity] = []
        for signal in signals:
            spread_pct = quote.spread_pct if quote else None
            liquidity = await self._liquidity(symbol, timeframe)
            opportunity = self.confluence.score(
                snapshot,
                signal,
                environment=self.environment,
                avg_dollar_volume=liquidity,
                spread_pct=spread_pct,
            )
            # Liquidity is ranking input, not a score input; keep it alongside the
            # other pre-trade facts so /analizar and the ranker agree.
            opportunity.historical.setdefault("avg_dollar_volume", liquidity or 0.0)
            opportunities.append(opportunity)

        opportunities.sort(key=lambda o: o.score, reverse=True)
        return ScannedSymbol(
            snapshot=snapshot, opportunities=opportunities[: self.strategy_settings.max_signals_per_scan]
        )

    async def _attach_htf(self, snapshot: TechnicalSnapshot, symbol: str) -> None:
        """Fill the higher-timeframe verdict used by multi-timeframe scoring."""
        htf = self.strategy_settings.higher_timeframe
        if snapshot.timeframe == htf:
            snapshot.htf_timeframe = snapshot.timeframe
            snapshot.htf_direction = snapshot.trend.direction
            return
        try:
            bars = await self._market.get_bars(symbol, htf, limit=250)
            if len(bars) < 30:
                return
            htf_snapshot = analyze_symbol(symbol, bars, timeframe=htf, settings=self.strategy_settings)
        except Exception as exc:  # noqa: BLE001 - unknown HTF must stay unknown
            log.debug("scanner.htf_unavailable", symbol=symbol, htf=htf, error=str(exc))
            return
        snapshot.htf_timeframe = htf
        snapshot.htf_direction = htf_snapshot.trend.direction
        if snapshot.htf_direction is None:
            return
        if snapshot.trend.direction is None:
            snapshot.htf_alignment = 0
        elif snapshot.htf_direction == snapshot.trend.direction:
            snapshot.htf_alignment = 1
        else:
            snapshot.htf_alignment = -1

    async def _liquidity(self, symbol: str, timeframe: str) -> float | None:
        try:
            return await self._market.avg_dollar_volume(symbol, timeframe, bars=30)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ ranking

    def _rank(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        scored = [(self._rank_score(o), o) for o in opportunities]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [o for _, o in scored]

    def _rank_score(self, opportunity: Opportunity) -> float:
        """Blend the factors that actually predict a tradeable, liquid setup."""
        best_signal = max(
            (s.score for s in opportunity.signals), default=float(opportunity.score)
        )
        liquidity = float(opportunity.historical.get("avg_dollar_volume") or 0.0)
        liquidity_component = 100.0 if liquidity >= 100_000_000 else 50.0
        risk_component = 100.0 - abs(100.0 - opportunity.breakdown.risk)

        raw = (
            RANK_WEIGHTS["score"] * opportunity.score
            + RANK_WEIGHTS["rr"] * min(100.0, opportunity.rr * 25.0)
            + RANK_WEIGHTS["strategy"] * min(100.0, best_signal)
            + RANK_WEIGHTS["liquidity"] * liquidity_component
            + RANK_WEIGHTS["risk"] * max(0.0, min(100.0, risk_component))
        )
        if opportunity.quality is Quality.ALTA:
            raw *= 1.05
        elif opportunity.quality is Quality.BAJA:
            raw *= 0.85
        elif opportunity.quality is Quality.NO_OPERABLE:
            raw *= 0.6
        return raw

    # ------------------------------------------------------- fingerprint memory

    def _is_new(self, fingerprint: str) -> bool:
        if not fingerprint:
            return True
        now = dt.datetime.now(dt.UTC)
        seen = self._seen_fingerprints.get(fingerprint)
        if seen is not None:
            if (now - seen).total_seconds() < self.settings.signal_dedup_window_minutes * 60:
                return False
        self._seen_fingerprints[fingerprint] = now
        return True

    def _prune_fingerprints(self) -> None:
        if len(self._seen_fingerprints) <= self.settings.max_fingerprint_history:
            return
        ordered = sorted(self._seen_fingerprints.items(), key=lambda kv: kv[1])
        for key, _ in ordered[: len(self._seen_fingerprints) // 2]:
            self._seen_fingerprints.pop(key, None)

    def reset_fingerprints(self) -> None:
        self._seen_fingerprints.clear()


__all__ = ["RANK_WEIGHTS", "MarketScanner", "ScanResult", "ScannedSymbol"]
