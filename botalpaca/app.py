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
from botalpaca.db import Database, set_database
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
        if self._alert_handler is None:
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
        log.info("app.reconciled", environment=environment.value, notes=len(notes))
        return notes

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
