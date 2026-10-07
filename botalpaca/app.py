"""Layer 2 — Application services and dependency wiring.

This module owns the object graph for one process. Its single most important
responsibility is the PAPER/REAL barrier: every broker-facing service is
constructed *inside* an :class:`EnvironmentContext` bound to exactly one
environment, and the context that is not active is not reachable from the
execution engine at all.

Switching environments (:meth:`Application.switch_environment`) rebuilds the
context, verifies credentials, queries the account and reports balances. It
never sends an order.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from botalpaca.analysis import analyze_symbol
from botalpaca.config.logging import get_logger
from botalpaca.config.settings import Settings, get_settings
from botalpaca.confluence import ConfluenceEngine, EntryQualityExplainer
from botalpaca.db import Database, TradeModel, TradeRepository, set_database
from botalpaca.domain import (
    AccountSnapshot,
    HealthStatus,
    MarketRegime,
    Opportunity,
    OrderClass,
    OrderState,
    OrderType,
    PositionSnapshot,
    RiskAssessment,
    SetupType,
    TradePlan,
    TradingEnvironment,
)
from botalpaca.domain.enums import SignalDirection
from botalpaca.domain.errors import BotalpacaError, ConfigurationError, ValidationError
from botalpaca.execution import (
    ACTION_SUBMIT,
    AlpacaTradingClient,
    ExecutionEngine,
    OrderBuilder,
    SubmissionResult,
)
from botalpaca.journal import TradeJournal
from botalpaca.learning import StatisticalEngine
from botalpaca.market import MarketDataService
from botalpaca.monitoring import MarketMonitor, PositionAlert, PositionMonitor
from botalpaca.notifications import NotificationService
from botalpaca.portfolio import PortfolioService
from botalpaca.protection import PositionProtectionManager
from botalpaca.risk import PortfolioRiskContext, RiskEngine
from botalpaca.scanner import MarketScanner, ScanResult
from botalpaca.security import (
    ACTIVE_ENV_KEY,
    AllowList,
    CircuitBreaker,
    ConfirmationRegistry,
    RateLimiter,
    SecurityLayer,
)
from botalpaca.strategies import StrategyEngine

log = get_logger(__name__)


@dataclass(frozen=True)
class _ExitFill:
    """One filled Alpaca order that reduced or closed a position."""

    order_id: str
    order_type: str
    price: float
    qty: float
    filled_at: dt.datetime


def _classify_exit(fills: list[_ExitFill], row: TradeModel) -> tuple[str, str]:
    """Name the exit from which order actually filled.

    A bracket or OCO pair is a take-profit limit order plus a stop-loss stop
    order, and Alpaca cancels whichever leg loses. So the leg that filled is
    the whole story of how the trade ended, which is why the stored order ids
    are checked before falling back to the order kind.
    """
    ids = {f.order_id for f in fills}
    kinds = {f.order_type for f in fills}

    if row.stop_order_id and str(row.stop_order_id) in ids:
        return "STOP", "Stop de protección llenado en Alpaca"
    if row.take_profit_order_id and str(row.take_profit_order_id) in ids:
        return "TARGET", "Take profit llenado en Alpaca"
    if "trailing_stop" in kinds:
        return "TRAILING", "Stop trailing llenado en Alpaca"
    if kinds & {"stop", "stop_limit"}:
        return "STOP", "Stop llenado en Alpaca"
    if "limit" in kinds:
        return "TARGET", "Orden límite de salida llenada en Alpaca"
    return "MANUAL", "Cierre manual en Alpaca"

#: Callback used to push a rendered message plus an optional Telegram keyboard.
Sender = Callable[[str, object], Awaitable[None]]


async def _no_sender(text: str, keyboard: object = None) -> None:
    """Fallback sender used before Telegram is wired (logs only)."""
    log.info("notification.queued", text=text)


@dataclass
class OpportunityAlert:
    """Payload handed to the Telegram layer by the background monitors."""

    environment: TradingEnvironment
    opportunities: tuple[Opportunity, ...]
    alerts: tuple[PositionAlert, ...]


AlertHandler = Callable[[OpportunityAlert], Awaitable[None]]


class PositionGateway:
    """Unified read/write view of one environment's positions.

    The monitors need both ``get_positions`` (read) and ``close_position``
    (write). Neither :class:`PortfolioService` nor :class:`ExecutionEngine`
    provides both on its own, so this thin adapter composes them. The
    environment barrier still holds: the engine it wraps is already bound.
    """

    def __init__(
        self,
        portfolio: PortfolioService,
        engine: ExecutionEngine,
        journal: TradeJournal,
        environment: TradingEnvironment,
    ) -> None:
        self._portfolio = portfolio
        self._engine = engine
        self._journal = journal
        self.environment = environment

    async def get_positions(self) -> list[PositionSnapshot]:
        return await self._portfolio.get_positions()

    async def get_position(self, symbol: str) -> PositionSnapshot | None:
        return await self._portfolio.get_position(symbol)

    async def close_position(
        self, symbol: str, *, qty: str | None = None, percentage: str | None = None, confirmed: bool = True
    ) -> SubmissionResult:
        result = await self._engine.close_position(
            symbol,
            environment=self.environment,
            qty=qty,
            percentage=percentage,
            confirmed=confirmed,
        )
        price = result.order.filled_avg_price if result.order else None
        await self._journal.add_event(
            0,
            self.environment,
            "CLOSE_REQUESTED",
            note=f"close_position {symbol} qty={qty} pct={percentage} order={result.order.id if result.order else 'n/a'}",
        )
        log.info(
            "app.position_close_requested",
            environment=self.environment.value,
            symbol=symbol.upper(),
            order_id=result.order.id if result.order else None,
            filled_price=price,
        )
        return result


class EnvironmentContext:
    """Every service bound to a single trading environment.

    Two of these exist; only the active one is ever used for execution. The
    inactive one may be built lazily but its engine is never handed an order.
    """

    def __init__(
        self,
        app: Application,
        environment: TradingEnvironment,
        *,
        notify: Callable[[Opportunity], Awaitable[None]] | None = None,
        notify_position: Callable[[PositionAlert], Awaitable[None]] | None = None,
    ) -> None:
        self.app = app
        self.environment = environment
        settings: Settings = app.settings
        config = settings.alpaca(environment)

        self.market = MarketDataService(
            config,
            environment=environment,
            timeout_seconds=settings.api_timeout_seconds,
            max_retries=settings.api_max_retries,
            backoff_seconds=settings.api_retry_backoff_seconds,
        )
        self.trading_client = AlpacaTradingClient(
            config,
            timeout_seconds=settings.api_timeout_seconds,
            max_retries=settings.api_max_retries,
            backoff_seconds=settings.api_retry_backoff_seconds,
        )
        self.portfolio = PortfolioService(self.trading_client)
        self.execution = ExecutionEngine(
            self.trading_client,
            app.database,
            active_environment=environment,
            require_confirmation=True,
        )
        self.protection = PositionProtectionManager(self.execution, app.database)
        self.risk = RiskEngine(app.database, self.portfolio)
        self.journal = app.journal
        self.learning = app.learning
        self.strategies = StrategyEngine(settings.strategy)
        self.confluence = ConfluenceEngine(strategy_settings=settings.strategy)
        self.explainer = EntryQualityExplainer()
        self.scanner = MarketScanner(
            self.market,
            environment=environment,
            strategies=self.strategies,
            confluence=self.confluence,
            settings=settings.scanner,
            strategy_settings=settings.strategy,
        )
        self.market_monitor = MarketMonitor(
            self.scanner,
            self.market,
            app.database,
            environment=environment,
            notify=notify,
            settings=settings.monitoring,
        )
        self.positions = PositionGateway(
            self.portfolio, self.execution, app.journal, environment
        )
        self.position_monitor = PositionMonitor(
            self.positions,
            self.scanner,
            self.protection,
            app.database,
            environment=environment,
            notify=notify_position,
            settings=settings.monitoring,
            protection_settings=settings.protection,
        )

    # ------------------------------------------------------------------ helpers

    @property
    def is_active(self) -> bool:
        return self.app.active is self

    async def verify(self) -> AccountSnapshot:
        """Authenticate and read the account. Used on boot and on every switch."""
        account = await self.portfolio.verify_connection()
        self.trading_client.close()
        self.market.close()
        return account

    async def analyze(self, symbol: str, timeframe: str | None = None) -> tuple[object, list[Opportunity]]:
        return await self.scanner.analyze_one(symbol, timeframe=timeframe or self.app.settings.strategy.default_timeframe)

    async def risk_context(self) -> PortfolioRiskContext:
        return await self.risk.build_context(self.environment)

    async def risk_assessment(
        self,
        opportunity: Opportunity,
        context: PortfolioRiskContext | None = None,
    ) -> RiskAssessment:
        ctx = context if context is not None else await self.risk_context()
        return await self.risk.assess(opportunity, ctx)

    async def open_orders(self) -> list[OrderState]:
        return await self.execution.get_open_orders(nested=True)

    def build_plan(self, opportunity: Opportunity, assessment: RiskAssessment) -> TradePlan:
        """Turn an approved opportunity plus its risk assessment into a plan."""
        qty = assessment.suggested_qty
        if qty <= 0:
            raise ValidationError("El motor de riesgo no sugiere ninguna cantidad.")
        return TradePlan(
            symbol=opportunity.symbol.upper(),
            environment=self.environment,
            direction=opportunity.direction,
            order_type=OrderType.MARKET,
            qty=qty,
            entry=opportunity.entry,
            atr=opportunity.atr or 0.0,
            stop_loss=opportunity.stop,
            take_profit=opportunity.target,
            order_class=OrderClass.BRACKET,
            risk_amount=assessment.max_risk_amount,
            risk_pct=assessment.risk_per_trade_pct,
            rr=opportunity.rr,
            score=opportunity.score,
            strategy=opportunity.strategy,
            setup=opportunity.setup,
            timeframe=opportunity.timeframe,
            opportunity_fingerprint=opportunity.fingerprint,
            notes=list(opportunity.confluences[:6]),
        )

    async def close(self) -> None:
        self.market_monitor.stop()
        self.position_monitor.stop()
        self.market.close()
        self.trading_client.close()


def _entry_reference(
    plan: TradePlan,
    order: OrderState | None,
    opportunity: Opportunity | None,
) -> float:
    """Best known entry price for a not-yet-filled order.

    A market order may still be ``new`` when Alpaca answers, so the fill price
    is only preferred when it exists; otherwise the reference is the plan's
    limit price, then the analysed entry.
    """
    if order is not None and order.filled_avg_price:
        return float(order.filled_avg_price)
    if plan.limit_price:
        return float(plan.limit_price)
    if opportunity is not None:
        return float(opportunity.entry)
    return 0.0


def _order_filled(order: OrderState | None) -> bool:
    """Whether Alpaca already executed the entry.

    Only a real fill price counts. ``filled_qty`` is unreliable across the SDK
    versions this runs on, but ``filled_avg_price`` is only ever populated once
    there is an actual fill.
    """
    if order is None:
        return False
    return bool(order.filled_avg_price) or order.status in _FILLED_STATUSES


_FILLED_STATUSES = frozenset({"filled", "partially_filled"})


class Application:
    """Process-level container: database, security, environments, services."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sender: Sender | None = None,
        database: Database | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.started_at = dt.datetime.now(dt.UTC)
        # Production runs real Alembic migrations; create_all covers tests and
        # first-run local development where the revision files are absent.
        self.database = database or Database(
            self.settings.database_url,
            echo=self.settings.database_echo,
            create_all=self.settings.database_auto_create,
            migrate=self.settings.database_migrate,
        )
        set_database(self.database)

        self.allowlist = AllowList.from_ids(self.settings.telegram_allowed_user_ids)
        self.security = SecurityLayer(
            self.database,
            self.allowlist,
            rate_limiter=RateLimiter(per_minute=self.settings.telegram_rate_limit_per_minute),
            circuit_breaker=CircuitBreaker(),
        )
        self.confirmations = ConfirmationRegistry()
        self.notifications = NotificationService(
            sender or _no_sender, max_per_hour=self.settings.monitoring.max_alerts_per_hour
        )
        self.journal = TradeJournal(self.database)
        self.learning = StatisticalEngine(self.database)

        self._sender = sender or _no_sender
        # Fingerprint of the last reconciliation report actually sent, so an
        # unresolved problem is announced once instead of every cycle.
        self._reconcile_state: str | None = None
        self._alert_handler: AlertHandler | None = None
        self.contexts: dict[TradingEnvironment, EnvironmentContext] = {}
        self.active: EnvironmentContext
        self._switch_lock = asyncio.Lock()

    # ------------------------------------------------------------------ wiring

    def set_sender(self, sender: Sender) -> None:
        self.notifications = NotificationService(
            sender, max_per_hour=self.settings.monitoring.max_alerts_per_hour
        )
        self._sender = sender

    def set_alert_handler(self, handler: AlertHandler | None) -> None:
        self._alert_handler = handler

    @property
    def active_environment(self) -> TradingEnvironment:
        return self.active.environment

    @property
    def risk(self) -> RiskEngine:
        return self.active.risk

    @property
    def scanner(self) -> MarketScanner:
        return self.active.scanner

    @property
    def portfolio(self) -> PortfolioService:
        return self.active.portfolio

    @property
    def execution(self) -> ExecutionEngine:
        return self.active.execution

    @property
    def protection(self) -> PositionProtectionManager:
        return self.active.protection

    def _build_context(self, environment: TradingEnvironment) -> EnvironmentContext:
        async def notify_opportunity(opportunity: Opportunity) -> None:
            await self.notify_opportunity(opportunity)

        async def notify_position(alert: PositionAlert) -> None:
            await self.notify_position_alert(alert)

        context = EnvironmentContext(
            self,
            environment,
            notify=notify_opportunity,
            notify_position=notify_position,
        )
        self.contexts[environment] = context
        return context

    async def start(self) -> AccountSnapshot:
        """Boot the process. PAPER is the default unless explicitly configured."""
        await self.database.start()

        environment = self.settings.active_trading_environment
        stored = await self.security.persisted_environment()
        if stored is not None and stored is not environment:
            log.info(
                "app.persisted_environment_differs",
                configured=environment.value,
                persisted=stored.value,
                note="the configured environment wins at boot; no orders are auto-executed",
            )

        try:
            context = self._build_context(environment)
            account = await context.verify()
        except ConfigurationError:
            raise
        except BotalpacaError as exc:
            raise ConfigurationError(
                f"No se pudo inicializar el entorno {environment.value}: {exc}"
            ) from exc

        self.active = context
        await self.security.persist_environment(environment)

        log.info(
            "app.started",
            environment=environment.value,
            account=account.account_number or account.account_id,
            equity=account.equity,
        )
        return account

    async def stop(self) -> None:
        for context in list(self.contexts.values()):
            await context.close()
        self.contexts.clear()
        await self.database.stop()
        log.info("app.stopped")

    # ------------------------------------------------------- environment switch

    async def switch_environment(
        self, target: TradingEnvironment, *, confirmed_by: int | None = None
    ) -> AccountSnapshot:
        """Change the active environment.

        Requires an explicit confirmation, which the caller (Telegram) obtains
        before calling this. No order of any kind is sent while switching.
        """
        if target is self.active.environment:
            return await self.portfolio.get_account()

        async with self._switch_lock:
            log.info(
                "app.switching_environment",
                **{"from": self.active.environment.value, "to": target.value},
            )
            old = self.active
            old.market_monitor.stop()
            old.position_monitor.stop()

            try:
                context = self.contexts.get(target)
                if context is None:
                    context = self._build_context(target)
                account = await context.verify()
            except BotalpacaError as exc:
                log.error("app.switch_failed", environment=target.value, error=str(exc))
                raise

            self.active = context
            await self.security.persist_environment(target)
            if confirmed_by is not None:
                await self._persist_attribution(confirmed_by)
            log.info(
                "app.environment_switched",
                environment=target.value,
                equity=account.equity,
                buying_power=account.buying_power,
            )
            return account

    async def _persist_attribution(self, user_id: int) -> None:
        from botalpaca.db import AppStateRepository

        async with self.database.session() as session:
            await AppStateRepository(session).set(
                f"{ACTIVE_ENV_KEY}:changed_by",
                {"user_id": user_id, "at": dt.datetime.now(dt.UTC).isoformat()},
            )

    # ------------------------------------------------------------- notifications

    async def notify_opportunity(self, opportunity: Opportunity) -> None:
        if self._alert_handler is None:
            return
        await self._alert_handler(
            OpportunityAlert(
                environment=self.active_environment, opportunities=(opportunity,), alerts=()
            )
        )

    async def notify_position_alert(self, alert: PositionAlert) -> None:
        """Deliver one position-level notice.

        The autonomous protector has already composed and rendered its own card,
        so it hands over a plain string rather than a ``PositionAlert``. Routing
        that through the alert pipeline silently lost every autonomous action --
        break-even, trailing, progressive and the emergency stop all applied but
        never reached Telegram.
        """
        if self._alert_handler is None:
            return
        if isinstance(alert, str):
            await self._alert_handler(alert)
            return
        await self._alert_handler(
            OpportunityAlert(environment=self.active_environment, opportunities=(), alerts=(alert,))
        )

    # -------------------------------------------------------------- operations

    async def scan(self, **kwargs) -> ScanResult:
        return await self.scanner.scan(**kwargs)

    async def analyze_symbol(self, symbol: str, timeframe: str | None = None):
        return await self.active.analyze(symbol, timeframe)

    async def build_plan(self, opportunity: Opportunity) -> tuple[TradePlan, RiskAssessment]:
        context = await self.active.risk_context()
        assessment = await self.active.risk.assess(opportunity, context)
        if not assessment.approved:
            raise ValidationError(
                "El motor de riesgo no aprueba la operación: "
                + "; ".join(assessment.blocks or assessment.reasons)
            )
        return self.active.build_plan(opportunity, assessment), assessment

    async def _check_levels_are_still_valid(self, plan: TradePlan) -> None:
        """Refuse a bracket the market has already moved away from.

        The gate approves a setup against the price it analysed. Between that
        moment and the tap the market can travel far enough to destroy the
        geometry without breaking either level's sign, so the ratio is
        re-derived at the live price before anything reaches Alpaca. LLY was
        approved at 1.5 R:R on $1105.06 and filled at $1176.00: a real 0.08 with
        the target 0.86% away and a stop 10.6% out. Every level was still on the
        correct side, so a sign check waved it through.

        Raises ``ValidationError`` tagged with the reason so the operator knows
        which guard spoke.
        """
        if plan.stop_loss is None or plan.take_profit is None:
            return

        risk = self.settings.risk

        def _cent(value: float) -> float:
            # Prices at or above $1.00 take two decimals; below that, four.
            return round(value, 2 if value >= 1.0 else 4)

        plan.stop_loss = _cent(plan.stop_loss)
        plan.take_profit = _cent(plan.take_profit)

        price = await self.active.market.get_last_price(plan.symbol)
        if not price or not plan.entry:
            return

        def _reject(reason: str, tag: str) -> ValidationError:
            return ValidationError(
                f"{plan.symbol}: {reason} (analizado ${plan.entry:,.2f}, "
                f"mercado ${price:,.2f}, objetivo ${plan.take_profit:,.2f}, "
                f"stop ${plan.stop_loss:,.2f}) [{tag}]. "
                "El analisis quedo viejo: vuelve a ejecutar /analizar "
                "para tener niveles frescos."
            )

        # 1. Drift: the levels belong to a price the trade will never get.
        drift_pct = abs(price - plan.entry) / plan.entry * 100.0
        if drift_pct > risk.max_entry_drift_pct:
            raise _reject(
                f"el precio se movio {drift_pct:.2f}%, mas del "
                f"{risk.max_entry_drift_pct:.2f}% permitido",
                "REJECTED_PRICE_DRIFT",
            )

        if plan.direction is SignalDirection.LONG:
            target_ok = plan.take_profit > price
            stop_ok = plan.stop_loss < price
        else:
            target_ok = plan.take_profit < price
            stop_ok = plan.stop_loss > price

        if not target_ok or not stop_ok:
            problem = (
                "el objetivo ya quedo del lado equivocado del mercado"
                if not target_ok
                else "el stop ya quedo del lado equivocado del mercado"
            )
            raise _reject(problem, "REJECTED_LEVEL_SIDE")

        # 2. Target proximity: a target a whisker away pays nothing for the risk.
        remaining = abs(plan.take_profit - price)
        floor_pct = risk.min_target_distance_pct
        floor_atr = (
            risk.min_target_distance_atr * plan.atr
            if plan.atr and plan.atr > 0
            else 0.0
        )
        floor = max(price * floor_pct / 100.0, floor_atr)
        if remaining < floor:
            raise _reject(
                f"el objetivo esta a {remaining:,.2f} ({remaining / price * 100:.2f}%), "
                f"menos de lo exigido ({floor:,.2f})",
                "REJECTED_TARGET_TOO_CLOSE",
            )

        # 3. Re-derive the ratio at the price the trade will actually get.
        reward = abs(plan.take_profit - price)
        risk_usd = abs(price - plan.stop_loss)
        rr_now = reward / risk_usd if risk_usd > 0 else 0.0
        if rr_now < risk.min_rr_at_entry:
            raise _reject(
                f"el R:R ahora es {rr_now:.2f}, por debajo del minimo "
                f"{risk.min_rr_at_entry:.2f} (venia {plan.rr:.2f})",
                "REJECTED_DEGRADED_RR",
            )


    async def submit_plan(
        self,
        plan: TradePlan,
        assessment: RiskAssessment,
        *,
        user_id: int,
        opportunity: Opportunity | None = None,
    ) -> SubmissionResult:
        """Submit an approved, confirmed plan and register its protection."""
        environment = self.active_environment
        self.security.allowlist.require(user_id)
        await self.security.ensure_trading_allowed()
        await self._check_levels_are_still_valid(plan)

        result = await self.active.execution.submit(
            plan,
            risk_approved=True,
            confirmed=True,
            confirmed_by=user_id,
            risk_score=assessment.score,
        )

        order = result.order
        # A bracket entry's children come back in ``order.legs``; only legs that
        # Alpaca actually acknowledged are recorded as protection.
        legs = {leg.order_type.value: leg for leg in (order.legs if order else [])}
        stop_leg = legs.get(OrderType.STOP.value)
        tp_leg = legs.get(OrderType.LIMIT.value)

        entry_price = _entry_reference(plan, order, opportunity)

        # An order sent while the market is shut stays accepted and fills at the
        # open. Recording that honestly lets the operator see the entry as
        # PENDING and lets the monitor announce the fill instead of it
        # materialising without warning.
        filled_at = dt.datetime.now(dt.UTC) if _order_filled(order) else None

        state = await self.protection.register_entry_protection(
            environment=environment,
            symbol=plan.symbol,
            qty=plan.qty,
            entry_price=entry_price,
            initial_stop_price=plan.stop_loss,
            stop_order_id=stop_leg.id if stop_leg else None,
            take_profit_order_id=tp_leg.id if tp_leg else None,
            order_class=plan.order_class,
        )

        strategy = plan.strategy or plan.setup or SetupType.MOMENTUM
        await self.journal.open_trade(
            environment=environment,
            symbol=plan.symbol,
            direction=plan.direction,
            strategy=strategy,
            setup=plan.setup or strategy,
            timeframe=plan.timeframe,
            qty=plan.qty,
            entry_price=entry_price,
            stop_price=plan.stop_loss,
            target_price=plan.take_profit,
            score=plan.score,
            rr=plan.rr,
            risk_amount=plan.risk_amount,
            atr=opportunity.atr if opportunity else None,
            regime=(opportunity.regime if opportunity else MarketRegime.UNKNOWN).value,
            sector=opportunity.sector if opportunity else None,
            confluences=list(opportunity.confluences) if opportunity else list(plan.notes),
            entry_reason=(
                opportunity.reasons[0]
                if opportunity and opportunity.reasons
                else (plan.notes[0] if plan.notes else None)
            ),
            entry_order_id=order.id if order else None,
            client_order_id=result.client_order_id,
            stop_order_id=state.stop_order_id,
            take_profit_order_id=state.take_profit_order_id,
            time_stop_at=state.time_stop_at,
            filled_at=filled_at,
        )
        return result

    async def reconcile(self) -> list[str]:
        """Rebuild local protection state from Alpaca and vice versa."""
        environment = self.active_environment
        positions = await self.portfolio.get_positions()
        notes = await self.protection.reconcile(
            environment=environment,
            positions=positions,
            protect_missing=self.settings.protection.auto_protect_missing_stop,
        )
        notes.extend(await self._settle_finished_trades(environment, positions))
        log.info("app.reconciled", environment=environment.value, notes=len(notes))
        await self._notify_reconciliation(environment, notes)
        return notes

    # ------------------------------------------------------------ settling the ledger

    async def _settle_finished_trades(
        self, environment: TradingEnvironment, positions: list[PositionSnapshot]
    ) -> list[str]:
        """Close ledger rows for trades the broker has already finished.

        Alpaca fills a protective stop or a take-profit without being asked, so
        the position disappears from the broker while the ledger keeps saying
        OPEN. Nothing used to read that gap. A trade that ended at the stop went
        on counting as open risk, kept its capital reserved, and never reached
        the statistics. Reconciliation is already the one place that compares
        SQLite against Alpaca, so the settlement belongs here.
        """
        live = {p.symbol.upper() for p in positions if p.qty != 0}
        async with self.database.session() as session:
            rows = list(await TradeRepository(session).get_open_positions(environment))

        notes: list[str] = []
        for row in rows:
            # Reading the symbol is inside the guard on purpose: a row this
            # process cannot even name must not abort the whole cycle and
            # leave every other finished trade unsettled.
            try:
                symbol = row.symbol.upper()
                if symbol in live:
                    continue
                note = await self._settle_trade(environment, row)
            except Exception as exc:  # noqa: BLE001 - one row must not stop the rest
                log.exception("app.settle_trade_failed", error=str(exc))
                notes.append(f"⚠️ {getattr(row, 'symbol', '?')}: no se pudo cerrar en el ledger ({exc})")
                continue
            if note is not None:
                notes.append(note)
        return notes

    async def _settle_trade(self, environment: TradingEnvironment, row: TradeModel) -> str | None:
        """Settle one trade whose position is gone. Returns a note, or None."""
        from alpaca.trading.enums import QueryOrderStatus

        symbol = row.symbol.upper()
        is_long = str(row.direction).upper().endswith("LONG")
        entry = float(row.entry_price or 0.0)
        qty = float(row.qty or 0.0)

        fills = await self._exit_fills(row, is_long, QueryOrderStatus)
        if fills:
            filled_qty = sum(f.qty for f in fills)
            exit_price = sum(f.price * f.qty for f in fills) / filled_qty
            exit_reason, exit_note = _classify_exit(fills, row)
            pnl = (
                (exit_price - entry) * filled_qty
                if is_long
                else (entry - exit_price) * filled_qty
            )
        elif row.filled_at is None:
            # No position and no exit: the entry never executed.
            exit_price, pnl = entry, 0.0
            exit_reason = "ENTRADA_NO_EJECUTADA"
            exit_note = "Alpaca no tiene posición y no registra ninguna salida"
        else:
            # The position is gone but no exit order can be tied to this trade.
            # Pricing it at the last quote is an estimate, and the note says so,
            # because a silently wrong P&L corrupts the statistics forever.
            exit_price = float(await self.active.market.get_last_price(symbol) or entry)
            pnl = (exit_price - entry) * qty if is_long else (entry - exit_price) * qty
            exit_reason = "CIERRE_SIN_REGISTRO"
            exit_note = (
                f"Sin orden de salida en Alpaca; precio estimado {exit_price:,.2f}"
            )

        closed = await self.journal.close_trade(
            environment=environment,
            symbol=symbol,
            exit_price=exit_price,
            pnl=pnl,
            exit_reason=exit_reason,
            exit_reason_note=exit_note,
            exit_order_id=fills[0].order_id if fills else None,
            trade_id=row.id,
        )
        if closed is None:
            return None
        return (
            f"📕 {symbol}: cerrada en el ledger por {exit_reason} "
            f"a {exit_price:,.2f} (P&L {pnl:+,.2f}, R {closed.r_multiple:+.2f})"
        )

    async def _exit_fills(self, row: TradeModel, is_long: bool, status_enum: object) -> list[_ExitFill]:
        """The filled orders Alpaca used to close this trade, oldest first.

        Filtered by side, by fill status, and by time: an older trade on the
        same symbol also produced exits, and counting those would invent a P&L
        that this trade never earned.
        """
        want_side = "sell" if is_long else "buy"
        orders = await self.active.trading_client.get_orders(
            status=status_enum.CLOSED,
            symbols=[row.symbol.upper()],
            limit=100,
            nested=True,
        )
        opened_at = row.opened_at
        if opened_at is not None and opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=dt.UTC)

        fills: list[_ExitFill] = []
        for order in orders:
            if str(getattr(order.side, "value", order.side)).lower() != want_side:
                continue
            if str(getattr(order.status, "value", order.status)).lower() != "filled":
                continue
            if not order.filled_at or not order.filled_qty:
                continue
            filled_at = order.filled_at
            if filled_at.tzinfo is None:
                filled_at = filled_at.replace(tzinfo=dt.UTC)
            if opened_at is not None and filled_at < opened_at:
                continue
            fills.append(
                _ExitFill(
                    order_id=str(order.id),
                    order_type=str(getattr(order.type, "value", order.type)),
                    price=float(order.filled_avg_price or 0.0),
                    qty=float(order.filled_qty),
                    filled_at=filled_at,
                )
            )
        return sorted(fills, key=lambda f: f.filled_at)

    async def _notify_reconciliation(
        self, environment: TradingEnvironment, notes: list[str]
    ) -> None:
        """Tell the operator when reconciliation found something they must know.

        These notes were only ever written to the log. A position that cannot be
        protected is the single most important thing this process knows and the
        one thing the operator cannot see -- silence on a stuck position reads
        exactly like "everything is fine".

        The same unresolved problem is re-discovered every cycle, so identical
        reports are only sent when they change. A repeat that clears is worth
        saying, because silence on a recovery looks like a stale warning.
        """
        actionable = [n for n in notes if "🚨" in n or "❌" in n]
        if not actionable:
            self._reconcile_state = None
            return

        lines = []
        for note in actionable:
            # One sentence per position: the note is written for a log reader.
            symbol, _, reason = note.partition(": ")
            lines.append(f"{symbol}\n   {reason}")

        fingerprint = "\n".join(sorted(actionable))
        previous = self._reconcile_state
        self._reconcile_state = fingerprint
        if fingerprint == previous:
            return

        text = "\n".join(
            [
                "<b>Aviso de proteccion</b>",
                f"Entorno: {environment.value}",
                "",
                *lines,
                "",
                "📈 Alpaca retiene las acciones de esas ordenes hasta "
                "que el mercado abra. El bot lo reintenta solo.",
            ]
        )
        try:
            await self.notifications.send(text, force=True)
        except Exception:  # noqa: BLE001 - a notice must never break reconciliation
            log.exception("app.reconcile_notify_failed")

    async def health(self) -> HealthStatus:
        database_ok = await self.database.healthcheck()
        alpaca_ok = True
        details: dict[str, object] = {}
        try:
            account = await self.portfolio.get_account()
        except Exception as exc:  # noqa: BLE001
            alpaca_ok = False
            details["alpaca_error"] = str(exc)
            account = None
        kill, reason = await self.security.kill_switch_state()
        if reason:
            details["kill_switch_reason"] = reason
        details["contexts_built"] = sorted(env.value for env in self.contexts)
        details["equity"] = account.equity if account else None
        details["allowlisted_users"] = len(self.allowlist.allowed)
        return HealthStatus(
            database_ok=database_ok,
            alpaca_ok=alpaca_ok,
            active_environment=self.active_environment,
            scheduler_ok=self.active.market_monitor.running,
            kill_switch=kill,
            uptime_seconds=(dt.datetime.now(dt.UTC) - self.started_at).total_seconds(),
            details=details,
        )

    async def describe_order(self, order_id: str) -> OrderState | None:
        return await self.portfolio.get_order(order_id)


def build_order_preview(plan: TradePlan, environment: TradingEnvironment) -> str:
    """Human-readable one-liner used in audit rows and confirmations."""
    try:
        OrderBuilder.build_entry(plan, client_order_id="preview")
    except BotalpacaError as exc:
        return f"{plan.symbol} {plan.direction.value} plan inválido: {exc}"
    side = plan.direction.value
    size = plan.qty or plan.notional or 0
    return f"{environment.value} {side} {size:g} {plan.symbol}"


__all__ = [
    "ACTIVE_ENV_KEY",
    "ACTION_SUBMIT",
    "AlertHandler",
    "Application",
    "EnvironmentContext",
    "OpportunityAlert",
    "PositionGateway",
    "Sender",
    "analyze_symbol",
    "build_order_preview",
]
