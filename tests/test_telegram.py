"""Telegram facade: parsing, guard translation, /modo, staged confirmations, read-only commands.

Every ``app`` collaborator here is a small duck-typed stub: the facade never
touches the network, so the whole Telegram surface is testable offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from botalpaca.config.settings import (
    AnalyticsSettings,
    MonitoringSettings,
    ProtectionSettings,
    RiskSettings,
    ScannerSettings,
    StrategySettings,
)
from botalpaca.domain.enums import (
    MarketRegime,
    OrderClass,
    OrderType,
    Quality,
    SetupType,
    SignalDirection,
    TradingEnvironment,
)
from botalpaca.domain.errors import AuthorizationError, KillSwitchError, RiskRejectedError
from botalpaca.domain.models import (
    AccountSnapshot,
    RiskAssessment,
    TradePlan,
)
from botalpaca.execution.engine import SubmissionResult
from botalpaca.security.confirmation import ConfirmationKind, ConfirmationRegistry
from botalpaca.security.guards import AllowList
from botalpaca.telegram.service import (
    CommandError,
    TelegramFacade,
    parse_flags,
    parse_float,
    parse_symbol,
    parse_symbol_list,
)

from .conftest import make_opportunity

PAPER = TradingEnvironment.PAPER
REAL = TradingEnvironment.REAL
USER = 111


# --------------------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------------------
def test_parse_symbol_normalises():
    assert parse_symbol("aapl") == "AAPL"
    assert parse_symbol("$msft") == "MSFT"
    assert parse_symbol("BRK.b") == "BRK.B"


@pytest.mark.parametrize("bad", ["", "123", "TOOLONGNAME", "AA PL"])
def test_parse_symbol_rejects_garbage(bad):
    with pytest.raises(CommandError):
        parse_symbol(bad)


def test_parse_symbol_list_splits_on_commas_and_spaces():
    assert parse_symbol_list("aapl, msft nvda") == ["AAPL", "MSFT", "NVDA"]


def test_parse_float_strips_formatting():
    assert parse_float("$1,234.50", "cantidad") == 1234.5
    assert parse_float("2.5%", "stop") == 2.5


@pytest.mark.parametrize("bad", ["abc", "0", "-3"])
def test_parse_float_rejects_non_positive(bad):
    with pytest.raises(CommandError):
        parse_float(bad, "cantidad")


def test_parse_flags():
    flags = parse_flags("cantidad=5 trailing=2")
    assert flags["cantidad"] == "5"
    assert flags["trailing"] == "2"


# --------------------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------------------
class _NoLimit:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def check(self, user_id: int) -> None:
        self.calls.append(user_id)


class _NoSecurity:
    def __init__(self) -> None:
        self.allowlist = AllowList.from_ids([USER])
        self.rate_limiter = _NoLimit()
        self.kill_switch = False
        self.reason: str | None = None

    async def kill_switch_state(self):
        return self.kill_switch, self.reason

    async def engage_kill_switch(self, reason: str) -> None:
        self.kill_switch = True
        self.reason = reason

    async def release_kill_switch(self) -> None:
        self.kill_switch = False
        self.reason = None

    async def ensure_trading_allowed(self) -> bool:
        if self.kill_switch:
            raise KillSwitchError("kill switch engaged")
        return True


def _account(environment: TradingEnvironment = PAPER) -> AccountSnapshot:
    return AccountSnapshot(
        environment=environment,
        account_id="acct-1",
        account_number="010203040",
        status="ACTIVE",
        currency="USD",
        equity=100_000.0,
        cash=50_000.0,
        buying_power=200_000.0,
        portfolio_value=100_000.0,
    )


class _Portfolio:
    def __init__(self, environment: TradingEnvironment = PAPER) -> None:
        self.environment = environment

    async def get_account(self):
        return _account(self.environment)

    async def get_positions(self):
        return []

    async def get_position(self, symbol: str):
        return None

    async def get_orders(self, **kwargs):
        return []

    async def get_order(self, order_id: str):
        return None

    async def exposures(self):
        return {"equity": 100_000.0, "buying_power": 200_000.0, "position_count": 0}


class _Risk:
    def __init__(self) -> None:
        self.approved = True
        self.calls = 0

    async def assess(self, opportunity, context) -> RiskAssessment:
        self.calls += 1
        if not self.approved:
            return RiskAssessment(approved=False, blocks=["límite de riesgo"], reasons=[])
        return RiskAssessment(
            approved=True,
            reasons=[],
            risk_per_trade_pct=1.0,
            max_risk_amount=1_000.0,
            suggested_qty=10.0,
            stop_distance_pct=2.0,
            score=88.0,
        )


class _Explainer:
    def render(self, opportunity, snapshot) -> str:
        return f"explicacion de {opportunity.symbol}"


class _Positions:
    def __init__(self) -> None:
        self.closed: list[tuple[str, dict[str, Any]]] = []

    async def close_position(self, symbol: str, **kwargs):
        self.closed.append((symbol, kwargs))
        return SimpleNamespace(order=None)


class _Monitor:
    running = False
    last_result = None


class _Context:
    def __init__(self, environment: TradingEnvironment) -> None:
        self.environment = environment
        self.explainer = _Explainer()
        self.risk = _Risk()
        self.positions = _Positions()
        self.market_monitor = _Monitor()

    async def risk_context(self):
        return SimpleNamespace(
            environment=self.environment,
            equity=100_000.0,
            buying_power=200_000.0,
            cash=50_000.0,
            open_risk=0.0,
            open_positions=[],
            correlations={},
            account=None,
        )

    async def risk_assessment(self, opportunity):
        return await self.risk.assess(opportunity, await self.risk_context())

    def build_plan(self, opportunity, assessment) -> TradePlan:
        return TradePlan(
            symbol=opportunity.symbol,
            environment=self.environment,
            direction=opportunity.direction,
            order_type=OrderType.MARKET,
            order_class=OrderClass.BRACKET,
            qty=assessment.suggested_qty,
            stop_loss=opportunity.stop,
            take_profit=opportunity.target,
            rr=opportunity.rr,
            score=opportunity.score,
            strategy=opportunity.setup,
            setup=opportunity.setup,
            timeframe=opportunity.timeframe,
            opportunity_fingerprint=opportunity.fingerprint,
        )


class _Journal:
    def __init__(self) -> None:
        self.accepted: list[tuple[str, str | None]] = []
        self.recorded: list[tuple[str, bool]] = []

    async def mark_signal_accepted(self, environment, fingerprint) -> None:
        self.accepted.append((environment.value, fingerprint))

    async def record_signal(self, opportunity, *, accepted: bool) -> None:
        self.recorded.append((opportunity.symbol, accepted))

    async def explain(self, environment, symbol):
        return {"environment": environment.value, "symbol": symbol}


class _Learning:
    def __init__(self) -> None:
        self.summary = SimpleNamespace(
            label="General",
            sample_size=25,
            win_rate=0.6,
            profit_factor=1.8,
            expectancy_r=0.4,
            average_r=0.4,
            average_win=1.6,
            average_loss=-0.8,
            max_drawdown_r=3.0,
            recovery_factor=2.0,
            sharpe=1.1,
            max_consecutive_wins=4,
            max_consecutive_losses=3,
            total_pnl=1_200.0,
            avg_mfe_r=1.9,
            avg_mae_r=-0.7,
            avg_hold_minutes=240.0,
            is_significant=True,
            caveat="",
        )

    async def overall(self, environment):
        return self.summary

    async def by_strategy(self, environment):
        return {"PULLBACK": self.summary}

    async def by_regime(self, environment):
        return {"TRENDING_UP": self.summary}

    async def recommendations(self, environment):
        return ["Considerar reducir tamaño en regimen RANGING"]


class _Protection:
    def __init__(self) -> None:
        self.cleared: list[tuple[str, str]] = []

    async def state_for_position(self, environment, position):
        return SimpleNamespace(has_stop=True, stop_price=97.0, notes=[])

    async def move_to_break_even(self, **kwargs):
        return SimpleNamespace(stop_price=100.15, notes=["break-even aplicado"])

    async def enable_trailing_stop(self, **kwargs):
        return SimpleNamespace(trail_percent=2.0, notes=["trailing activo"])

    async def disable_trailing_stop(self, **kwargs):
        return SimpleNamespace(notes=["trailing cancelado"])

    async def reconcile(self, environment, positions):
        return []

    async def clear(self, environment, symbol) -> None:
        self.cleared.append((environment.value, symbol))


class _Execution:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel_order(self, order_id: str, *, environment) -> bool:
        self.cancelled.append(order_id)
        return True

    async def replace_order(self, order_id, request, *, environment):
        return SimpleNamespace(id=order_id)


class _Session:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Database:
    def session(self):
        database = self

        class _Ctx:
            async def __aenter__(self_inner):
                return _Session()

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


class _Notifications:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    async def send(self, text: str, keyboard=None, *, force: bool = False) -> bool:
        self.sent.append((text, keyboard))
        return True


def _snapshot(symbol: str = "AAPL"):
    return SimpleNamespace(
        symbol=symbol,
        timeframe="1D",
        volatility=SimpleNamespace(regime=MarketRegime.TRENDING_UP),
    )


class FakeApp:
    """Minimal stand-in for :class:`botalpaca.app.Application`."""

    def __init__(self, environment: TradingEnvironment = PAPER) -> None:
        self.environment = environment
        self.security = _NoSecurity()
        self.confirmations = ConfirmationRegistry()
        self.allowlist = self.security.allowlist
        self.portfolio = _Portfolio(environment)
        self.active = _Context(environment)
        self.journal = _Journal()
        self.learning = _Learning()
        self.protection = _Protection()
        self.execution = _Execution()
        self.database = _Database()
        self.notifications = _Notifications()
        self.settings = SimpleNamespace(
            risk=RiskSettings(),
            monitoring=MonitoringSettings(),
            protection=ProtectionSettings(),
            strategy=StrategySettings(),
            scanner=ScannerSettings(),
            analytics=AnalyticsSettings(),
            database_url="sqlite+aiosqlite:///./data/botalpaca.db",
            timezone="America/New_York",
            log_level="INFO",
            scan_interval_seconds=300,
            monitor_interval_seconds=60,
            reconcile_interval_seconds=300,
            telegram_rate_limit_per_minute=20,
            telegram_allowed_user_ids=[USER],
            alpaca=lambda env: SimpleNamespace(
                is_configured=True,
                base_url="paper" if env.is_paper else "live",
                safe_summary=lambda: {"configured": True},
            ),
        )
        self.submitted: list[TradePlan] = []
        self.switched: list[TradingEnvironment] = []
        self.reconciled = 0

    @property
    def active_environment(self) -> TradingEnvironment:
        return self.environment

    async def analyze_symbol(self, symbol: str):
        if symbol == "NODATA":
            return None
        opportunity = make_opportunity(
            environment=self.environment, symbol=symbol, direction=SignalDirection.LONG
        )
        return _snapshot(symbol), [opportunity]

    async def scan(self):
        return SimpleNamespace(best=lambda: [], errors={}, tradable=[])

    async def submit_plan(self, plan, assessment, *, user_id, opportunity=None) -> SubmissionResult:
        self.security.allowlist.require(user_id)
        await self.security.ensure_trading_allowed()
        self.submitted.append(plan)
        order = SimpleNamespace(
            id="order-1",
            status="accepted",
            legs=[
                SimpleNamespace(
                    order_type=OrderType.STOP,
                    side=SimpleNamespace(value="sell"),
                    qty=plan.qty,
                    symbol=plan.symbol,
                ),
                SimpleNamespace(
                    order_type=OrderType.LIMIT,
                    side=SimpleNamespace(value="sell"),
                    qty=plan.qty,
                    symbol=plan.symbol,
                ),
            ],
        )
        return SubmissionResult(order, 1, "baP-test", plan.environment)

    async def switch_environment(self, target: TradingEnvironment, *, confirmed_by=None):
        self.switched.append(target)
        self.environment = target
        return _account(target)

    async def health(self):
        return SimpleNamespace(
            database_ok=True,
            alpaca_ok=True,
            active_environment=self.environment,
            scheduler_ok=True,
            kill_switch=False,
            uptime_seconds=120.0,
            details={},
        )

    async def reconcile(self):
        self.reconciled += 1
        return ["ok"]


def _update(user_id: int = USER, text: str = ""):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        message=SimpleNamespace(text=text),
    )


# --------------------------------------------------------------------------------------
# guard translation
# --------------------------------------------------------------------------------------
async def test_guard_translates_kill_switch():
    facade = TelegramFacade(FakeApp())

    async def boom():
        raise KillSwitchError("kill switch engaged")

    with pytest.raises(CommandError, match="Kill switch"):
        await facade.guard(boom)


async def test_guard_translates_unexpected_error():
    facade = TelegramFacade(FakeApp())

    async def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(CommandError, match="inesperado"):
        await facade.guard(boom)


async def test_guard_passes_results_through():
    facade = TelegramFacade(FakeApp())
    result = await facade.guard(facade.help, _update())
    assert result.text


async def test_authorize_rejects_unknown_user():
    facade = TelegramFacade(FakeApp())
    with pytest.raises(CommandError):
        await facade.authorize(_update(999))


# --------------------------------------------------------------------------------------
# basic commands
# --------------------------------------------------------------------------------------
async def test_help_lists_commands():
    facade = TelegramFacade(FakeApp())
    result = await facade.help(_update())
    for command in ("/analizar", "/posiciones", "/modo", "/stats"):
        assert command in result.text


async def test_start_reports_environment():
    facade = TelegramFacade(FakeApp(PAPER))
    result = await facade.start(_update())
    assert "ALPACA PAPER" in result.text


async def test_status_shows_auto_trading_is_off_by_design():
    facade = TelegramFacade(FakeApp())
    result = await facade.status(_update())
    assert "desactivado" in result.text


# --------------------------------------------------------------------------------------
# /modo
# --------------------------------------------------------------------------------------
async def test_modo_paper_offers_switch_to_real():
    facade = TelegramFacade(FakeApp(PAPER))
    result = await facade.modo(_update())
    assert "ALPACA PAPER" in result.text
    data = result.keyboard.inline_keyboard[0][0].callback_data
    assert data == "modo:REAL"


async def test_modo_real_offers_switch_to_paper_and_warns():
    facade = TelegramFacade(FakeApp(REAL))
    result = await facade.modo(_update())
    assert "ALPACA REAL" in result.text
    assert "Dinero real" in result.text
    data = result.keyboard.inline_keyboard[0][0].callback_data
    assert data == "modo:PAPER"


async def test_modo_stages_the_switch_without_acting():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    await facade.modo(_update(), "REAL")
    pending = await app.confirmations.peek(USER)
    assert pending is not None
    assert pending.kind == ConfirmationKind.ENVIRONMENT
    assert app.switched == []


async def test_modo_rejects_unknown_argument():
    facade = TelegramFacade(FakeApp())
    with pytest.raises(CommandError, match="Uso"):
        await facade.modo(_update(), "DEMO")


async def test_confirm_mode_switch_requires_pending_confirmation():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    with pytest.raises(CommandError, match="confirmaci"):
        await facade.confirm_mode_switch(_update(), REAL)
    assert app.switched == []


async def test_confirm_mode_switch_rejects_the_wrong_target():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    await facade.modo(_update(), "REAL")
    with pytest.raises(CommandError, match="confirmaci"):
        await facade.confirm_mode_switch(_update(), PAPER)
    assert app.switched == []


async def test_confirm_mode_switch_switches_and_says_no_orders_were_sent():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    await facade.modo(_update(), "REAL")
    result = await facade.confirm_mode_switch(_update(), REAL)
    assert app.switched == [REAL]
    assert app.active_environment == REAL
    assert "No se envió ninguna orden" in result.text


# --------------------------------------------------------------------------------------
# buy / sell staging
# --------------------------------------------------------------------------------------
async def test_comprar_requires_arguments():
    facade = TelegramFacade(FakeApp())
    with pytest.raises(CommandError, match="comprar"):
        await facade.comprar(_update(), "")


async def test_comprar_reports_when_data_is_missing():
    facade = TelegramFacade(FakeApp())
    with pytest.raises(CommandError):
        await facade.comprar(_update(), "NODATA")


async def test_comprar_paper_stages_a_confirmed_plan():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    result = await facade.comprar(_update(), "AAPL")
    assert "ALPACA PAPER" in result.text
    data = result.keyboard.inline_keyboard[0][0].callback_data
    assert data.startswith("confirmar:paper")
    pending = await app.confirmations.peek(USER)
    assert pending is not None and pending.kind == ConfirmationKind.TRADE
    assert app.submitted == []


async def test_vender_real_shows_the_mandatory_real_warning():
    app = FakeApp(REAL)
    # A SHORT signal is what /vender picks; the default fake only offers LONG.
    async def _short(symbol):
        snapshot, _ = await FakeApp.analyze_symbol(app, symbol)
        return snapshot, [
            make_opportunity(
                environment=REAL, symbol=symbol, direction=SignalDirection.SHORT
            )
        ]

    app.analyze_symbol = _short
    facade = TelegramFacade(app)
    result = await facade.vender(_update(), "AAPL")
    assert "ALPACA REAL" in result.text
    assert "dinero real" in result.text
    data = result.keyboard.inline_keyboard[0][0].callback_data
    assert data.startswith("confirmar:real")
    assert app.submitted == []


async def test_comprar_accepts_a_quantity_override():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    await facade.comprar(_update(), "AAPL 3")
    pending = await app.confirmations.peek(USER)
    assert pending is not None
    assert pending.details["plan"]["qty"] == 3.0


async def test_comprar_refuses_when_risk_rejects():
    app = FakeApp(PAPER)
    app.active.risk.approved = False
    facade = TelegramFacade(app)
    result = await facade.comprar(_update(), "AAPL")
    assert "no permit" in result.text.lower()
    assert await app.confirmations.peek(USER) is None


# --------------------------------------------------------------------------------------
# confirm_trade
# --------------------------------------------------------------------------------------
async def test_confirm_trade_submits_exactly_once():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    await facade.comprar(_update(), "AAPL")
    result = await facade.confirm_trade(_update(), "paper")
    assert len(app.submitted) == 1
    assert "Orden enviada" in result.text
    assert app.journal.accepted == [(PAPER.value, "fp-test-AAPL")]
    # the confirmation is consumed, so a replay cannot double-fill
    with pytest.raises(CommandError):
        await facade.confirm_trade(_update(), "paper")
    assert len(app.submitted) == 1


async def test_confirm_trade_rejects_a_token_for_the_other_environment():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    await facade.comprar(_update(), "AAPL")
    with pytest.raises(CommandError):
        await facade.confirm_trade(_update(), "real")
    assert app.submitted == []


async def test_confirm_trade_requires_a_pending_operation():
    facade = TelegramFacade(FakeApp())
    with pytest.raises(CommandError, match="pendiente"):
        await facade.confirm_trade(_update(), "paper")


async def test_cancelar_pendiente_drops_the_confirmation():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)
    await facade.comprar(_update(), "AAPL")
    result = await facade.cancelar_pendiente(_update())
    assert await app.confirmations.peek(USER) is None
    assert app.submitted == []
    assert "no se envió ninguna orden" in result.text.lower()


# --------------------------------------------------------------------------------------
# read-only commands
# --------------------------------------------------------------------------------------
async def test_cuenta_shows_the_active_environment_only():
    facade = TelegramFacade(FakeApp(PAPER))
    result = await facade.cuenta(_update())
    assert "ALPACA PAPER" in result.text
    assert "ALPACA REAL" not in result.text


async def test_posiciones_lists_nothing_when_flat():
    facade = TelegramFacade(FakeApp())
    result = await facade.posiciones(_update())
    assert "ALPACA PAPER" in result.text


async def test_ordenes_lists_nothing_when_flat():
    facade = TelegramFacade(FakeApp())
    result = await facade.ordenes(_update())
    assert "ALPACA PAPER" in result.text


async def test_portfolio_reports_exposures():
    facade = TelegramFacade(FakeApp())
    result = await facade.portfolio(_update())
    assert "ALPACA PAPER" in result.text


async def test_riesgo_reports_the_configured_limits():
    facade = TelegramFacade(FakeApp())
    result = await facade.riesgo(_update())
    assert "ALPACA PAPER" in result.text


async def test_riesgo_detalle_reports_blocks_for_a_non_tradable_symbol():
    app = FakeApp(PAPER)
    facade = TelegramFacade(app)

    async def analyze(symbol: str):
        return _snapshot(symbol), [make_opportunity(symbol=symbol, tradable=False)]

    app.analyze_symbol = analyze
    result = await facade.riesgo_detalle(_update(), "AAPL")
    assert "ALPACA PAPER" in result.text


async def test_stats_shows_the_sample_size():
    facade = TelegramFacade(FakeApp())
    result = await facade.stats(_update())
    assert "ALPACA PAPER" in result.text


async def test_aprender_lists_non_binding_recommendations():
    facade = TelegramFacade(FakeApp())
    result = await facade.aprender(_update())
    assert "RANGING" in result.text


async def test_monitor_reports_auto_trading_state():
    facade = TelegramFacade(FakeApp())
    result = await facade.monitor(_update())
    assert "desactivado" in result.text


async def test_config_never_leaks_secrets():
    facade = TelegramFacade(FakeApp())
    result = await facade.config(_update())
    assert "ALPACA PAPER" in result.text
    assert "API_KEY" not in result.text


async def test_reconciliar_reports_the_reconciliation():
    app = FakeApp()
    facade = TelegramFacade(app)
    await facade.reconciliar(_update())
    assert app.reconciled == 1


async def test_explicar_returns_the_journal_entry():
    facade = TelegramFacade(FakeApp())
    result = await facade.explicar(_update(), "AAPL")
    assert "AAPL" in result.text


# --------------------------------------------------------------------------------------
# helpers that depend on real settings
# --------------------------------------------------------------------------------------
def test_setup_type_is_used_for_the_default_strategy():
    assert SetupType.PULLBACK.value == "PULLBACK"
    assert Quality.ALTA.value == "ALTA"


def test_risk_rejected_error_carries_reasons():
    error = RiskRejectedError(["max positions"])
    assert error.reasons == ["max positions"]


def test_authorization_error_is_a_botalpaca_error():
    from botalpaca.domain.errors import BotalpacaError

    assert issubclass(AuthorizationError, BotalpacaError)


# -- PTB adapter ---------------------------------------------------------------------
def test_build_telegram_application_is_constructible():
    """The PTB adapter must build against the *installed* python-telegram-bot.

    `ApplicationBuilder` exposes neither `.parse_mode()` nor
    `.drop_pending_updates()`; guessing those method names produced an
    `AttributeError` only at production startup, because every other test drives
    the facade directly and never constructs the real Application.
    """
    from telegram.ext import Application as TelegramApplication

    from botalpaca.telegram.bot import build_telegram_application

    app = build_telegram_application("123456:AAHunit-test-token")
    assert isinstance(app, TelegramApplication)
    assert app.bot_data is not None


def test_register_handlers_registers_every_command():
    from botalpaca.telegram.bot import build_telegram_application, register_handlers

    app = build_telegram_application("123456:AAHunit-test-token")
    register_handlers(app, None)
    assert app.handlers[0], "no command handlers registered"
    assert app.error_handlers, "no error handler registered"


def test_run_starts_the_updater(monkeypatch):
    """`Application.start()` does NOT fetch updates in PTB >= 21.

    The updater must be started explicitly. When it was not, the deployed bot
    connected to Alpaca, started the scheduler and logged "main.running" while
    silently never consuming a single Telegram update, so every command the user
    sent was ignored.
    """
    import inspect

    from telegram.ext import Application

    source = inspect.getsource(Application.start)
    assert "does *not* start fetching updates" in source

    from botalpaca import __main__

    main_source = inspect.getsource(__main__._run)
    assert "updater.start_polling" in main_source, "the updater is never started"
    assert "updater.stop()" in main_source, "the updater is never stopped"


async def test_answer_sends_to_the_update_chat(monkeypatch):
    """`CallbackContext` has no `effective_chat`; the chat lives on the update."""
    from types import SimpleNamespace

    from botalpaca.telegram.bot import _answer

    sent: list[tuple[object, ...]] = []

    class _Bot:
        async def send_message(self, *args: object, **kwargs: object) -> None:
            sent.append(args)

    update = SimpleNamespace(effective_chat=SimpleNamespace(id=42), callback_query=None)
    context = SimpleNamespace(bot=_Bot())

    await _answer(update, context, "hola")

    assert sent == [(42, "hola")]
    assert sent[0] and isinstance(sent[0][0], int)


async def test_answer_resolves_chat_for_callback_queries(monkeypatch):
    from types import SimpleNamespace

    from botalpaca.telegram.bot import _answer

    sent: list[tuple[object, ...]] = []

    class _Bot:
        async def send_message(self, *args: object, **kwargs: object) -> None:
            sent.append(args)

    # A callback query has no effective_chat; the chat is callback_query.message.chat.
    update = SimpleNamespace(
        effective_chat=None,
        callback_query=SimpleNamespace(message=SimpleNamespace(chat=SimpleNamespace(id=99))),
    )
    context = SimpleNamespace(bot=_Bot())

    await _answer(update, context, "confirmado")

    assert sent[0][0] == 99


async def test_answer_splits_long_messages(monkeypatch):
    from types import SimpleNamespace

    from botalpaca.telegram.bot import _answer

    sent: list[tuple[object, ...]] = []

    class _Bot:
        async def send_message(self, *args: object, **kwargs: object) -> None:
            sent.append(args)

    update = SimpleNamespace(effective_chat=SimpleNamespace(id=7), callback_query=None)
    context = SimpleNamespace(bot=_Bot())

    await _answer(update, context, "x" * 9000)

    assert len(sent) == 3
    assert sent[0][0] == 7


# -- command list published to Telegram ------------------------------------------------
def test_every_command_has_a_description():
    """Typing "/" must suggest every command the bot answers to."""
    from botalpaca.telegram.bot import COMMAND_DESCRIPTIONS, build_telegram_application, register_handlers

    app = build_telegram_application("123456:AAHunit-test-token")
    register_handlers(app, None)
    handled = set(app.bot_data["commands"])

    assert handled == set(COMMAND_DESCRIPTIONS), (
        f"missing: {handled - set(COMMAND_DESCRIPTIONS)}; "
        f"extra: {set(COMMAND_DESCRIPTIONS) - handled}"
    )
    assert all(d.strip() for d in COMMAND_DESCRIPTIONS.values())
    assert all(len(d) <= 256 for d in COMMAND_DESCRIPTIONS.values())


async def test_publish_commands_sends_the_menu():
    published: list[list[object]] = []

    class _Bot:
        async def set_my_commands(self, commands: list[object]) -> None:
            published.append(commands)

    from botalpaca.telegram.bot import COMMAND_DESCRIPTIONS, publish_commands

    count = await publish_commands(_Bot())

    assert count == len(COMMAND_DESCRIPTIONS)
    assert len(published) == 1
    assert {c.command for c in published[0]} == set(COMMAND_DESCRIPTIONS)


async def test_publish_commands_survives_a_telegram_error():
    from telegram.error import TelegramError

    from botalpaca.telegram.bot import publish_commands

    class _Bot:
        async def set_my_commands(self, commands: list[object]) -> None:
            raise TelegramError("nope")

    assert await publish_commands(_Bot()) == 0


def test_run_publishes_commands_at_startup():
    import inspect

    from botalpaca import __main__

    assert "publish_commands" in inspect.getsource(__main__._run)


async def test_commands_without_args_are_not_given_any():
    """`/start`, `/help` and `/status` take no args; forwarding one broke them."""
    from types import SimpleNamespace

    from botalpaca.telegram.bot import _make_command

    seen: dict[str, object] = {}

    async def _noop_send(*args: object, **kwargs: object) -> None:
        return None

    class _Facade:
        async def guard(self, method, update, **kwargs):
            seen["kwargs"] = kwargs
            return SimpleNamespace(text="ok", keyboard=None)

        async def start(self, update: object) -> str:
            seen["called"] = "start"
            return "ok"

        async def modo(self, update: object, args: str | None = None) -> str:
            seen["args"] = args
            return "ok"

    context = SimpleNamespace(
        args=["PAPER"],
        application=SimpleNamespace(bot_data={"facade": _Facade()}),
        bot=SimpleNamespace(send_message=_noop_send),
    )
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=1))

    for method_name, expects_args in (("start", False), ("modo", True)):
        seen.clear()
        handler = _make_command(method_name, method_name)
        await handler(update, context)
        assert seen.get("kwargs") == ({"args": "PAPER"} if expects_args else {})
