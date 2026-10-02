"""Layer 14 — Monitoring engine.

Two background monitors, both environment-scoped and both non-destructive by
default:

* :class:`MarketMonitor` runs the scanner while the market is open and surfaces
  new opportunities with an explicit ACEPTAR / RECHAZAR gate. It never executes
  anything unless ``MONITORING_AUTO_TRADING_ENABLED`` is explicitly turned on
  (default: off).
* :class:`PositionMonitor` re-analyses open positions, compares the new score
  against the stored one, and reports *what changed and why* — it never closes a
  position unless a previously configured protection rule fires.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

from botalpaca.config import MonitoringSettings, ProtectionSettings, get_settings
from botalpaca.config.logging import get_logger
from botalpaca.db import AppStateRepository, Database, TradeRepository
from botalpaca.domain import (
    ExitReason,
    HealthStatus,
    Opportunity,
    PositionSnapshot,
    TradingEnvironment,
)
from botalpaca.market import MarketDataService
from botalpaca.protection import PositionProtectionManager
from botalpaca.scanner import MarketScanner, ScanResult

from .autonomy import AutonomousProtector, AutonomyReport

log = get_logger(__name__)

LAST_SCAN_KEY = "last_scan_at"
LAST_RECONCILE_KEY = "last_reconcile_at"
SCORE_PREFIX = "position_score:"


@dataclass
class PositionAlert:
    """Something the operator needs to know about an open or pending trade.

    Two shapes share this record. A ``deterioration`` explains why a position
    lost score. A ``fill`` announces that an entry queued outside market hours
    finally executed, carrying the price Alpaca actually paid and the levels now
    protecting it.
    """

    symbol: str
    environment: TradingEnvironment
    previous_score: float = 0.0
    current_score: float = 0.0
    changes: list[str] = field(default_factory=list)
    position: PositionSnapshot | None = None
    exit_rule_triggered: str | None = None
    kind: str = "deterioration"
    filled_at: dt.datetime | None = None
    fill_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    qty: float | None = None

    @property
    def drop(self) -> float:
        return self.previous_score - self.current_score

    @property
    def is_fill(self) -> bool:
        return self.kind == "fill"


class MarketMonitor:
    """Background opportunity discovery with a human gate."""

    def __init__(
        self,
        scanner: MarketScanner,
        market: MarketDataService,
        database: Database,
        *,
        environment: TradingEnvironment,
        notify: object | None = None,
        settings: MonitoringSettings | None = None,
    ) -> None:
        self._scanner = scanner
        self._market = market
        self._db = database
        self.environment = environment
        self._notify = notify
        self.settings = settings or get_settings().monitoring
        self.last_result: ScanResult | None = None
        self._last_alert: dict[str, dt.datetime] = {}
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    async def run_once(self) -> ScanResult:
        """One scan pass, outside market hours included (useful for testing)."""
        result = await self._scanner.scan()
        self.last_result = result
        await self._mark_scan_time()
        for opportunity in result.tradable:
            await self._maybe_alert(opportunity)
        return result

    async def loop(self, interval_seconds: int) -> None:
        """Scan loop. Sleeps out of market hours instead of hammering the API."""
        self._running = True
        try:
            while self._running:
                try:
                    is_open = await self._market.is_market_open()
                    if is_open:
                        await self.run_once()
                    else:
                        log.debug("market_monitor.closed")
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - a bad cycle must not kill the worker
                    log.exception("market_monitor.cycle_failed")
                await asyncio.sleep(interval_seconds)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    async def _maybe_alert(self, opportunity: Opportunity) -> None:
        if self._notify is None:
            return
        if opportunity.score < self.settings.opportunity_alert_min_score:
            return
        if not opportunity.tradable:
            return
        if self.settings.auto_trading_enabled:
            # Automatic execution is a deliberate, opt-in decision. Even when on,
            # the alert is still sent so the action is auditable in Telegram.
            log.warning(
                "market_monitor.auto_trading_enabled",
                symbol=opportunity.symbol,
                note="auto execution requested but requires an approved plan",
            )
        now = dt.datetime.now(dt.UTC)
        last = self._last_alert.get(opportunity.fingerprint)
        if last is not None and (now - last).total_seconds() < (
            self.settings.signal_alert_cooldown_seconds
        ):
            return
        self._last_alert[opportunity.fingerprint] = now
        await self._notify(opportunity)

    async def _mark_scan_time(self) -> None:
        async with self._db.session() as session:
            await AppStateRepository(session).set(
                LAST_SCAN_KEY, dt.datetime.now(dt.UTC).isoformat()
            )


class PositionMonitor:
    """Watches open positions for deterioration and enforces protection rules."""

    def __init__(
        self,
        portfolio: object,
        scanner: MarketScanner,
        protection: PositionProtectionManager,
        database: Database,
        *,
        environment: TradingEnvironment,
        notify: object | None = None,
        settings: MonitoringSettings | None = None,
        protection_settings: ProtectionSettings | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._scanner = scanner
        self._protection = protection
        self._db = database
        self.environment = environment
        self._notify = notify
        self.settings = settings or get_settings().monitoring
        self.protection_settings = protection_settings or get_settings().protection
        self._last_alert: dict[str, dt.datetime] = {}
        self._last_autonomy: dict[str, dt.datetime] = {}
        self._running = False
        self.autonomy = AutonomousProtector(
            protection,
            environment=environment,
            settings=self.protection_settings,
            notify=notify,
        )

    async def apply_autonomy(self, position: PositionSnapshot) -> list[AutonomyReport]:
        """Run the unattended protection rules for one position."""
        return await self.autonomy.evaluate(position)

    @property
    def running(self) -> bool:
        return self._running

    async def check_all(self) -> list[PositionAlert]:
        """Re-analyse every open position and report the changes."""
        positions = await self._portfolio.get_positions()  # type: ignore[attr-defined]
        # Fills first: an entry that executed while we were not looking is the
        # one thing the operator cannot see anywhere else.
        alerts: list[PositionAlert] = list(await self.check_fills(positions))
        for position in positions:
            if position.qty == 0:
                continue
            # Protection runs first and unattended: securing an open position
            # is more urgent than telling the operator about it.
            reports = await self.apply_autonomy(position)
            if reports:
                self._last_autonomy[position.symbol] = dt.datetime.now(dt.UTC)
            alert = await self.check_position(position)
            if alert is not None:
                alerts.append(alert)
        return alerts

    async def check_fills(
        self, positions: Sequence[PositionSnapshot]
    ) -> list[PositionAlert]:
        """Announce entries that Alpaca filled while we were waiting.

        An order sent with the market closed stays ``accepted`` until the open,
        so the position only shows up afterwards. Without this the operator finds
        a position they were never told about.
        """
        alerts: list[PositionAlert] = []
        async with self._db.session() as session:
            repo = TradeRepository(session)
            pending = await repo.get_pending(self.environment)
            live = {p.symbol.upper(): p for p in positions}
            for row in pending:
                symbol = row.symbol.upper()
                # The row's own filled_at is the authority for "has this already
                # been announced", not whether a position shows up.
                if row.filled_at is not None:
                    continue
                position = live.get(symbol)
                if position is None or not position.qty:
                    # Alpaca creates the position the instant the parent order
                    # fills, so its absence means the entry has NOT executed.
                    # Announcing here would tell the operator the trade is open
                    # while the order is still sitting in the queue.
                    continue
                filled_price = self._fill_price(row, positions)
                marked = await repo.mark_filled(
                    row.id,
                    self.environment,
                    filled_at=dt.datetime.now(dt.UTC),
                    entry_price=filled_price,
                )
                if not marked:
                    continue
                log.info(
                    "position.filled",
                    environment=self.environment.value,
                    symbol=symbol,
                    qty=row.qty,
                    price=filled_price,
                )
                alerts.append(
                    PositionAlert(
                        kind="fill",
                        symbol=symbol,
                        environment=self.environment,
                        filled_at=dt.datetime.now(dt.UTC),
                        fill_price=filled_price,
                        stop_price=row.stop_price,
                        target_price=row.target_price,
                        qty=row.qty,
                        position=position,
                        changes=["La entrada se ejecuto al abrir el mercado"],
                    )
                )
        return alerts

    def _fill_price(self, row: object, positions: Sequence[PositionSnapshot]) -> float:
        """What was actually paid, preferring the live position's average.

        A gap at the open is normal, so Alpaca's average entry beats the price
        we planned against last night's close.
        """
        for position in positions:
            if position.symbol.upper() == row.symbol.upper() and position.avg_entry_price:
                return round(float(position.avg_entry_price), 2)
        return round(float(row.entry_price or 0.0), 2)

    async def check_position(self, position: PositionSnapshot) -> PositionAlert | None:
        """Compare the current technicals against the score stored at entry."""
        previous = await self.stored_score(position.symbol)
        try:
            outcome = await self._scanner.analyze_one(position.symbol)
        except Exception as exc:  # noqa: BLE001 - never abort the sweep
            log.warning("position_monitor.analysis_failed", symbol=position.symbol, error=str(exc))
            return None
        if outcome is None:
            return None
        snapshot, opportunities = outcome
        current = opportunities[0].score if opportunities else self._score_from_snapshot(snapshot)
        await self.store_score(position.symbol, current)

        changes = self._describe_changes(snapshot, position)
        drop = previous - current
        triggered = None
        # One gate for the whole alert: the score drop is the signal, the change
        # list is only the explanation. Alerting on changes alone produced noise
        # on every flat position.
        if not self.settings.momentum_alerts_enabled or drop < self.settings.score_drop_alert_threshold:
            return None
        triggered = f"score -{drop:.0f} ({previous:.0f} → {current:.0f})"

        alert = PositionAlert(
            symbol=position.symbol,
            environment=self.environment,
            previous_score=previous,
            current_score=current,
            changes=changes,
            position=position,
            exit_rule_triggered=triggered,
        )
        await self._maybe_notify(alert)
        return alert

    # ------------------------------------------------------------------ scoring

    @staticmethod
    def _score_from_snapshot(snapshot: object) -> float:
        """Fallback score when no strategy currently proposes a setup.

        Used for the "what changed" comparison only; it is never used to gate
        execution, so a lighter weighting is acceptable here.
        """
        trend = snapshot.trend.strength * 100.0  # type: ignore[attr-defined]
        momentum = snapshot.momentum.score * 100.0  # type: ignore[attr-defined]
        volume = snapshot.volume.score * 100.0  # type: ignore[attr-defined]
        volatility = snapshot.volatility.score * 100.0  # type: ignore[attr-defined]
        quality = float(getattr(snapshot, "data_quality", 0.0))
        raw = trend * 0.35 + momentum * 0.25 + volume * 0.20 + volatility * 0.20
        return round(raw * (0.5 + 0.5 * quality), 1)

    def _describe_changes(self, snapshot: object, position: PositionSnapshot) -> list[str]:
        """Human-readable list of what actually changed for this position."""
        changes: list[str] = []
        long_position = position.qty >= 0
        trend = snapshot.trend  # type: ignore[attr-defined]
        momentum = snapshot.momentum  # type: ignore[attr-defined]
        volume = snapshot.volume  # type: ignore[attr-defined]
        indicators = snapshot.indicators  # type: ignore[attr-defined]
        close = indicators.close

        if trend.direction is not None:
            if (long_position and trend.direction.name == "SHORT") or (
                not long_position and trend.direction.name == "LONG"
            ):
                changes.append("Señal de tendencia ahora en contra de la posición")
            if trend.strength < 0.3:
                changes.append("Tendencia debilitándose")

        if long_position and indicators.ema_20 is not None and close < indicators.ema_20:
            changes.append(f"Precio debajo de EMA20 ({indicators.ema_20:,.2f})")
        if not long_position and indicators.ema_20 is not None and close > indicators.ema_20:
            changes.append(f"Precio encima de EMA20 ({indicators.ema_20:,.2f})")

        if indicators.ema_50 is not None:
            above = close > indicators.ema_50
            if long_position and not above:
                changes.append("Pérdida de EMA50")
            if not long_position and above:
                changes.append("Pérdida de EMA50 (posición short)")

        if momentum.rsi is not None:
            if long_position and momentum.rsi < 50:
                changes.append(f"RSi perdiendo fuerza ({momentum.rsi:.0f})")
            if not long_position and momentum.rsi > 50:
                changes.append(f"RSi perdiendo fuerza ({momentum.rsi:.0f})")
        if momentum.divergence:
            changes.append(f"Divergencia detectada: {momentum.divergence}")

        if volume.rel_volume is not None and volume.rel_volume < 0.8:
            changes.append("Volumen disminuyendo")
        if volume.obv_confirming is False:
            changes.append("OBV sin confirmar el movimiento")

        if snapshot.structure.retest:  # type: ignore[attr-defined]
            changes.append("Precio en retest de nivel clave")
        if not changes:
            changes.append("Sin degradación técnica detectada")
        return changes

    # ------------------------------------------------------------------ storage

    def _score_key(self, symbol: str) -> str:
        # The environment is part of the key: a PAPER score must never be read
        # back as a REAL score for the same ticker.
        return f"{SCORE_PREFIX}{self.environment.value}:{symbol.upper()}"

    async def stored_score(self, symbol: str) -> float:
        async with self._db.session() as session:
            raw = await AppStateRepository(session).get(self._score_key(symbol), None)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    async def store_score(self, symbol: str, score: float) -> None:
        async with self._db.session() as session:
            await AppStateRepository(session).set(self._score_key(symbol), score)

    async def clear_score(self, symbol: str) -> None:
        async with self._db.session() as session:
            await AppStateRepository(session).delete(self._score_key(symbol))

    # --------------------------------------------------------------- time stop

    async def enforce_time_stops(self) -> list[str]:
        """Close positions whose configured time stop has expired."""
        notes: list[str] = []
        positions = await self._portfolio.get_positions()  # type: ignore[attr-defined]
        for position in positions:
            if position.qty == 0:
                continue
            if not await self._protection.is_time_stop_due(self.environment, position.symbol):
                continue
            try:
                await self._portfolio.close_position(  # type: ignore[attr-defined]
                    position.symbol, confirmed=True
                )
                notes.append(f"⏱ {position.symbol}: cerrado por time stop")
            except Exception as exc:  # noqa: BLE001
                log.error("position_monitor.time_stop_failed", symbol=position.symbol, error=str(exc))
                notes.append(f"❌ {position.symbol}: time stop falló ({exc})")
        return notes

    # -------------------------------------------------------------- health/loop

    async def health(self) -> HealthStatus:
        try:
            database_ok = await self._db.healthcheck()
        except Exception:  # noqa: BLE001
            database_ok = False
        return HealthStatus(
            database_ok=database_ok,
            alpaca_ok=True,
            active_environment=self.environment,
            scheduler_ok=self._running,
            kill_switch=False,
        )

    async def loop(self, interval_seconds: int) -> None:
        self._running = True
        try:
            while self._running:
                try:
                    alerts = await self.check_all()
                    if alerts:
                        log.info("position_monitor.alerts", count=len(alerts))
                    await self.enforce_time_stops()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    log.exception("position_monitor.cycle_failed")
                await asyncio.sleep(interval_seconds)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    async def _maybe_notify(self, alert: PositionAlert) -> None:
        if self._notify is None:
            return
        now = dt.datetime.now(dt.UTC)
        last = self._last_alert.get(alert.symbol)
        if last is not None and (now - last).total_seconds() < (
            self.settings.position_alert_cooldown_seconds
        ):
            return
        self._last_alert[alert.symbol] = now
        await self._notify(alert)


#: Exit reasons a monitor is allowed to act on without a new user confirmation.
AUTOMATIC_EXIT_REASONS = {ExitReason.STOP_LOSS, ExitReason.TIME_STOP, ExitReason.TRAILING_STOP}

__all__ = [
    "AUTOMATIC_EXIT_REASONS",
    "LAST_RECONCILE_KEY",
    "LAST_SCAN_KEY",
    "SCORE_PREFIX",
    "MarketMonitor",
    "PositionAlert",
    "PositionMonitor",
]
