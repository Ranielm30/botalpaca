"""Layer 12 — Statistical learning engine.

Every metric is computed from SQLite rows scoped to exactly one environment.
Sample size is always reported and never hidden: conclusions drawn from small
samples are labelled as such.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Sequence

from botalpaca.config import AnalyticsSettings, get_settings
from botalpaca.config.logging import get_logger
from botalpaca.db import Database, TradeRepository
from botalpaca.db.models import TradeModel
from botalpaca.domain import StatisticalSummary, SymbolStats, TradingEnvironment

log = get_logger(__name__)


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator in (0, 0.0):
        return None
    return numerator / denominator


class StatisticalEngine:
    """Computes performance statistics per group, always environment-scoped."""

    def __init__(
        self,
        database: Database,
        *,
        settings: AnalyticsSettings | None = None,
    ) -> None:
        self._db = database
        self._settings = settings or get_settings().analytics

    @property
    def settings(self) -> AnalyticsSettings:
        return self._settings

    # ------------------------------------------------------------------ public

    async def overall(self, environment: TradingEnvironment) -> StatisticalSummary:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._summarize("GLOBAL", rows)

    async def by_strategy(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: t.strategy)

    async def by_setup(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: t.setup)

    async def by_timeframe(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: t.timeframe)

    async def by_regime(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: t.regime)

    async def by_sector(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: t.sector or "UNKNOWN")

    async def by_symbol(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: t.symbol)

    async def by_score_band(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: _score_band(t.score))

    async def by_weekday(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: _weekday(t.opened_at))

    async def by_hour(self, environment: TradingEnvironment) -> dict[str, StatisticalSummary]:
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        return self._group(rows, lambda t: _hour_bucket(t.opened_at))

    async def symbol_detail(
        self, environment: TradingEnvironment, symbol: str
    ) -> SymbolStats | None:
        async with self._db.session() as session:
            rows = [
                t
                for t in await TradeRepository(session).get_closed(environment)
                if t.symbol == symbol.upper()
            ]
        if not rows:
            return None
        wins = [t for t in rows if (t.pnl or 0.0) > 0]
        summary = self._summarize(symbol.upper(), rows)
        setups = self._group(rows, lambda t: t.setup)
        best_setup, best_wr = _best_setup(setups)
        return SymbolStats(
            symbol=symbol.upper(),
            environment=environment,
            total_trades=len(rows),
            wins=len(wins),
            losses=len(rows) - len(wins),
            win_rate=summary.win_rate,
            avg_r=summary.average_r or 0.0,
            best_setup=best_setup,
            best_setup_win_rate=best_wr,
            weak_condition=_weak_condition(rows),
            caveat=summary.caveat,
        )

    async def similar_setups(
        self,
        environment: TradingEnvironment,
        *,
        strategy: str | None = None,
        setup: str | None = None,
        regime: str | None = None,
        exclude_trade_id: int | None = None,
    ) -> StatisticalSummary | None:
        """Stats for trades matching the same setup context (used by entry quality)."""
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        matching = [
            t
            for t in rows
            if t.id != exclude_trade_id
            and (strategy is None or t.strategy == strategy)
            and (setup is None or t.setup == setup)
            and (regime is None or t.regime == regime)
        ]
        if not matching:
            return None
        return self._summarize("SIMILARES", matching)

    async def recommendations(self, environment: TradingEnvironment) -> list[str]:
        """Non-binding suggestions. Never mutates settings."""
        out: list[str] = []
        async with self._db.session() as session:
            rows = list(await TradeRepository(session).get_closed(environment))
        if len(rows) < self._settings.min_sample_for_significance:
            return [
                f"Muestra insuficiente ({len(rows)} operaciones, mínimo "
                f"{self._settings.min_sample_for_significance}). No se sacan conclusiones."
            ]

        by_strategy = self._group(rows, lambda t: t.strategy)
        ranked = sorted(
            (s for s in by_strategy.values() if s.sample_size >= 10),
            key=lambda s: s.average_r or -99.0,
            reverse=True,
        )
        for summary in ranked[:2]:
            if (summary.average_r or 0.0) > 0:
                out.append(
                    f"Estrategia {summary.label}: R medio {summary.average_r:+.2f} "
                    f"con {summary.sample_size} operaciones "
                    f"(win rate {summary.win_rate * 100:.0f}%)"
                )
        for summary in ranked[-2:]:
            if summary.sample_size >= 10 and (summary.average_r or 0.0) < 0:
                out.append(
                    f"Estrategia {summary.label} tiene R medio {summary.average_r:+.2f} "
                    f"con {summary.sample_size} operaciones: considerar reducir su peso"
                )

        by_symbol = self._group(rows, lambda t: t.symbol)
        for summary in by_symbol.values():
            if summary.sample_size >= 8 and summary.expectancy_r is not None and summary.expectancy_r < -0.3:
                out.append(
                    f"{summary.label}: expectativa {summary.expectancy_r:+.2f}R con "
                    f"{summary.sample_size} operaciones"
                )
        return out or ["Sin patrones significativos todavía."]

    # --------------------------------------------------------------- internals

    def _group(
        self, rows: Sequence[TradeModel], keyfn: Callable[[TradeModel], str]
    ) -> dict[str, StatisticalSummary]:
        buckets: dict[str, list[TradeModel]] = {}
        for row in rows:
            buckets.setdefault(str(keyfn(row) or "UNKNOWN"), []).append(row)
        return {label: self._summarize(label, group) for label, group in buckets.items()}

    def _summarize(self, label: str, rows: Sequence[TradeModel]) -> StatisticalSummary:
        s = self._settings
        n = len(rows)
        if n == 0:
            return StatisticalSummary(
                label=label, sample_size=0, win_rate=0.0, caveat="Sin operaciones"
            )

        ordered = sorted(rows, key=lambda t: t.closed_at or t.opened_at)
        rs = [float(t.r_multiple or 0.0) for t in ordered]
        pnls = [float(t.pnl or 0.0) for t in ordered]
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r <= 0]

        avg_win = sum(wins) / len(wins) if wins else None
        avg_loss = sum(losses) / len(losses) if losses else None
        win_rate = len(wins) / n

        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        profit_factor = _safe_div(gross_win, gross_loss)
        expectancy = sum(rs) / n

        equity_curve: list[float] = []
        cumulative = 0.0
        for pnl in pnls:
            cumulative += pnl
            equity_curve.append(cumulative)
        peak, max_dd = 0.0, 0.0
        for value in equity_curve:
            peak = max(peak, value)
            max_dd = max(max_dd, peak - value)
        max_dd_r = None
        if max(pnls) > 0 or min(pnls) < 0:
            max_dd_r = _safe_div(max_dd, abs(sum(pnls) / n) if sum(pnls) else 0.0)
        recovery_factor = _safe_div(abs(sum(pnls)), max_dd)

        sharpe = None
        if n >= s.min_sample_for_sharpe:
            sharpe = _sharpe(rs)

        streak_win = _max_streak([r > 0 for r in rs])
        streak_loss = _max_streak([r <= 0 for r in rs])

        mfe = [float(t.mfe_r) for t in ordered if t.mfe_r is not None]
        mae = [float(t.mae_r) for t in ordered if t.mae_r is not None]
        durations = [float(t.duration_seconds) for t in ordered if t.duration_seconds]

        significant = n >= s.min_sample_for_significance
        caveat = None
        if not significant:
            caveat = f"Muestra pequeña ({n} operaciones): no estadísticamente significativo"
        elif n < s.min_sample_for_sharpe:
            caveat = f"Sharpe no calculado (requiere {s.min_sample_for_sharpe} operaciones)"

        return StatisticalSummary(
            label=label,
            sample_size=n,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy_r=expectancy,
            average_r=sum(rs) / n,
            average_win=avg_win,
            average_loss=avg_loss,
            max_drawdown_r=max_dd_r,
            recovery_factor=recovery_factor,
            sharpe=sharpe,
            max_consecutive_wins=streak_win,
            max_consecutive_losses=streak_loss,
            total_pnl=sum(pnls),
            avg_mfe_r=sum(mfe) / len(mfe) if mfe else None,
            avg_mae_r=sum(mae) / len(mae) if mae else None,
            avg_hold_minutes=sum(durations) / len(durations) / 60.0 if durations else None,
            is_significant=significant,
            caveat=caveat,
        )


# --------------------------------------------------------------------- helpers


def _sharpe(values: Sequence[float]) -> float | None:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(variance)
    if sd == 0:
        return None
    return round((mean / sd) * math.sqrt(n), 3)


def _max_streak(flags: Sequence[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _score_band(score: float | None) -> str:
    if score is None:
        return "UNKNOWN"
    if score >= 85:
        return "85-100"
    if score >= 70:
        return "70-84"
    if score >= 55:
        return "55-69"
    return "0-54"


def _weekday(when: dt.datetime | None) -> str:
    if when is None:
        return "UNKNOWN"
    return ["LUN", "MAR", "MIÉ", "JUE", "VIE", "SÁB", "DOM"][when.weekday()]


def _hour_bucket(when: dt.datetime | None) -> str:
    if when is None:
        return "UNKNOWN"
    hour = when.hour
    if hour < 9:
        return "PRE-APERTURA"
    if hour < 12:
        return "09-12"
    if hour < 14:
        return "12-14"
    if hour < 17:
        return "14-17"
    return "POST-CIERRE"


def _best_setup(summaries: dict[str, StatisticalSummary]) -> tuple[str | None, float | None]:
    candidates = [s for s in summaries.values() if s.sample_size >= 3 and (s.average_r or 0) > 0]
    if not candidates:
        return None, None
    best = max(candidates, key=lambda s: s.average_r or 0.0)
    return best.label, best.win_rate


def _weak_condition(rows: Sequence[TradeModel]) -> str | None:
    if len(rows) < 5:
        return None
    regimes: dict[str, list[float]] = {}
    for row in rows:
        regimes.setdefault(row.regime or "UNKNOWN", []).append(float(row.r_multiple or 0.0))
    worst = min(regimes.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    mean_r = sum(worst[1]) / len(worst[1])
    if mean_r < 0:
        return f"régimen {worst[0]} (R medio {mean_r:+.2f}, n={len(worst[1])})"
    return None


__all__ = ["StatisticalEngine"]
