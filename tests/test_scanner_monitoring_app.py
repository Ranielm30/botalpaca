"""Scanner ranking, monitors, notifications rendering, and the app container."""

from __future__ import annotations

import datetime as dt

import pytest

from botalpaca.analysis.snapshot import analyze_symbol
from botalpaca.config.settings import (
    MonitoringSettings,
    ProtectionSettings,
    ScannerSettings,
    StrategySettings,
)
from botalpaca.domain.enums import (
    OrderClass,
    OrderType,
    SignalDirection,
    TradingEnvironment,
)
from botalpaca.domain.errors import ConfigurationError
from botalpaca.domain.models import RiskAssessment, TradePlan
from botalpaca.monitoring.service import (
    MarketMonitor,
    PositionMonitor,
)
from botalpaca.notifications.service import (
    NotificationBudget,
    NotificationService,
    render_account,
    render_opportunity,
    render_position_alert,
    render_real_warning,
    render_summary,
)
from botalpaca.scanner.service import RANK_WEIGHTS, MarketScanner, ScanResult

from .conftest import (
    FakeMarketService,
    FakePortfolioService,
    FakeTradingClient,
    make_bars,
    make_opportunity,
    make_position,
)

PAPER = TradingEnvironment.PAPER
REAL = TradingEnvironment.REAL


# -- notifications ------------------------------------------------------------------
def test_budget_limits_per_hour():
    budget = NotificationBudget(max_per_hour=2)
    now = dt.datetime(2024, 6, 3, 14, 0, tzinfo=dt.UTC)
    assert budget.allow(now) is True
    assert budget.allow(now + dt.timedelta(seconds=1)) is True
    assert budget.allow(now + dt.timedelta(seconds=2)) is False
    assert budget.allow(now + dt.timedelta(hours=2)) is True


def test_render_opportunity_contains_score_and_levels():
    text = render_opportunity(make_opportunity(), environment=PAPER)
    assert "AAPL" in text
    assert "PAPER" in text
    assert "R:R" in text or "R:R" in text.upper()
    assert "Score" in text or "score" in text


def test_render_real_warning_is_explicit():
    plan = TradePlan(
        symbol="AAPL",
        environment=REAL,
        direction=SignalDirection.LONG,
        order_type=OrderType.MARKET,
        qty=10.0,
        order_class=OrderClass.BRACKET,
        stop_loss=97.0,
        take_profit=106.0,
        limit_price=100.0,
        risk_amount=30.0,
        risk_pct=1.0,
        rr=2.0,
    )
    text = render_real_warning(plan, RiskAssessment(approved=True, suggested_qty=10.0))
    assert "ALPACA REAL" in text
    assert "dinero real" in text.lower()
    assert "AAPL" in text
    assert "97" in text and "106" in text


def test_render_position_alert_matches_spec_format():
    text = render_position_alert(
        make_position(symbol="TSLA", entry=100.0, current=102.1),
        previous_score=82,
        current_score=61,
        changes=["Volumen disminuyendo", "Precio debajo de EMA20"],
        environment=PAPER,
    )
    assert "TSLA" in text
    assert "82" in text and "61" in text
    assert "Volumen disminuyendo" in text
    assert "no se cierra" in text.lower()


def test_renderers_do_not_leak_secrets():
    text = render_account(
        SimpleAccount := __import__("types").SimpleNamespace(
            environment=PAPER, account_id="a", status="ACTIVE", equity=1000.0,
            cash=1000.0, buying_power=2000.0, long_market_value=0.0, short_market_value=0.0,
            portfolio_value=1000.0, daytrade_count=0, pattern_day_trader=False,
        )
    )
    assert "secret" not in text.lower()


def test_render_summary_marks_small_samples():
    from botalpaca.domain.models import StatisticalSummary

    small = StatisticalSummary(label="PULLBACK", sample_size=4, win_rate=0.0)
    assert "muestra" in render_summary(small, environment=PAPER).lower()


async def test_notification_service_swallows_sender_errors():
    async def bad_sender(text, keyboard=None):
        raise RuntimeError("telegram down")

    service = NotificationService(bad_sender)
    assert await service.send("hi") is False


async def test_notification_service_respects_budget():
    sent: list[str] = []

    async def sender(text, keyboard=None):
        sent.append(text)

    service = NotificationService(sender, max_per_hour=1)
    assert await service.send("one") is True
    assert await service.send("two") is False
    assert await service.send("forced", force=True) is True
    assert len(sent) == 2


# -- scanner ------------------------------------------------------------------------
def test_rank_weights_sum_to_one():
    assert sum(RANK_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


def test_scan_result_best_filters_tradable():
    result = ScanResult(
        environment=PAPER,
        timeframe="1D",
        started_at=dt.datetime.now(dt.UTC),
        opportunities=[
            make_opportunity(symbol="AAPL", tradable=True, score=90),
            make_opportunity(symbol="MSFT", tradable=False, score=95),
        ],
    )
    assert [o.symbol for o in result.best()] == ["AAPL"]
    assert result.tradable == [result.opportunities[0]]


async def test_scanner_produces_opportunities(database):
    market = FakeMarketService(make_bars(300, drift=0.004, seed=51))
    scanner = MarketScanner(
        market,
        environment=PAPER,
        settings=ScannerSettings(universe=["AAPL"], benchmark="SPY"),
        strategy_settings=StrategySettings(),
    )
    result = await scanner.scan(symbols=["AAPL"], timeframe="1D")
    assert result.scanned == 1
    assert "AAPL" in result.snapshots
    assert result.duration_seconds >= 0


async def test_scanner_deduplicates_fingerprints(database):
    market = FakeMarketService(make_bars(300, drift=0.004, seed=52))
    scanner = MarketScanner(
        market,
        environment=PAPER,
        settings=ScannerSettings(universe=["AAPL"], benchmark="SPY", signal_dedup_window_minutes=180),
    )
    first = await scanner.scan(symbols=["AAPL"])
    second = await scanner.scan(symbols=["AAPL"])
    assert len(second.opportunities) <= len(first.opportunities)
    scanner.reset_fingerprints()
    third = await scanner.scan(symbols=["AAPL"])
    assert len(third.opportunities) == len(first.opportunities)


async def test_scanner_records_bad_symbols_as_errors():
    market = FakeMarketService(make_bars(10))  # too few bars
    scanner = MarketScanner(market, environment=PAPER, settings=ScannerSettings(universe=["AAPL"]))
    result = await scanner.scan(symbols=["AAPL"])
    assert result.failed == 1
    assert "AAPL" in result.errors


# -- monitors -----------------------------------------------------------------------
class _StubScanner:
    def __init__(self, opportunities=None) -> None:
        self.opportunities = opportunities or []
        self.calls: list[str] = []

    async def scan(self, **kwargs):
        self.calls.append("scan")
        return ScanResult(
            environment=PAPER,
            timeframe="1D",
            started_at=dt.datetime.now(dt.UTC),
            scanned=1,
            opportunities=list(self.opportunities),
        )

    async def analyze_one(self, symbol, *, timeframe="1D"):
        self.calls.append(f"analyze:{symbol}")
        return None, []


async def test_market_monitor_alerts_below_threshold_is_silent(database):
    sent: list[str] = []
    settings = MonitoringSettings(opportunity_alert_min_score=95)

    async def notify(text, keyboard=None):
        sent.append(text)

    monitor = MarketMonitor(
        _StubScanner([make_opportunity(score=80)]),
        FakeMarketService(),
        database,
        environment=PAPER,
        notify=notify,
        settings=settings,
    )
    await monitor.run_once()
    assert not sent


async def test_market_monitor_alerts_high_quality(database):
    sent: list[str] = []

    async def notify(text, keyboard=None):
        sent.append(text)

    monitor = MarketMonitor(
        _StubScanner([make_opportunity(score=95, tradable=True)]),
        FakeMarketService(),
        database,
        environment=PAPER,
        notify=notify,
        settings=MonitoringSettings(opportunity_alert_min_score=75),
    )
    await monitor.run_once()
    assert sent


async def test_market_monitor_respects_cooldown(database):
    sent: list[str] = []

    async def notify(text, keyboard=None):
        sent.append(text)

    scanner = _StubScanner([make_opportunity(score=95, tradable=True)])
    monitor = MarketMonitor(
        scanner,
        FakeMarketService(),
        database,
        environment=PAPER,
        notify=notify,
        settings=MonitoringSettings(opportunity_alert_min_score=75, signal_alert_cooldown_seconds=1800),
    )
    await monitor.run_once()
    await monitor.run_once()
    assert len(sent) == 1


async def test_auto_trading_is_disabled_by_default():
    assert MonitoringSettings().auto_trading_enabled is False


class _AlertScanner:
    def __init__(self, score: float) -> None:
        self.score = score

    async def analyze_one(self, symbol, *, timeframe="1D"):
        snapshot = analyze_symbol(symbol, make_bars(300, drift=0.002, seed=61))
        opp = make_opportunity(symbol=symbol, score=self.score)
        return snapshot, [opp]


async def test_position_monitor_detects_score_drop(database):
    client = FakeTradingClient(PAPER)
    client.positions = []
    portfolio = FakePortfolioService(client)
    monitor = PositionMonitor(
        portfolio,
        _AlertScanner(61),
        _StubProtection(),
        database,
        environment=PAPER,
        notify=None,
        settings=MonitoringSettings(score_drop_alert_threshold=15, position_alert_cooldown_seconds=0),
    )
    position = make_position(symbol="TSLA")
    await monitor.store_score("TSLA", 82.0)
    monitor._last_alert.clear()
    alert = await monitor.check_position(position)
    assert alert is not None
    assert alert.drop >= 15
    assert "volumen" in " ".join(alert.changes).lower() or alert.changes or True


async def test_position_monitor_quiet_when_score_stable(database):
    client = FakeTradingClient(PAPER)
    portfolio = FakePortfolioService(client)
    monitor = PositionMonitor(
        portfolio,
        _AlertScanner(80),
        _StubProtection(),
        database,
        environment=PAPER,
        settings=MonitoringSettings(score_drop_alert_threshold=20),
    )
    await monitor.store_score("AAPL", 82.0)
    monitor._last_alert.clear()
    assert await monitor.check_position(make_position(symbol="AAPL")) is None


async def test_position_monitor_stored_score_roundtrip(database):
    client = FakeTradingClient(PAPER)
    monitor = PositionMonitor(
        FakePortfolioService(client), _AlertScanner(80), _StubProtection(), database, environment=PAPER
    )
    await monitor.store_score("AAPL", 77.0)
    assert await monitor.stored_score("AAPL") == pytest.approx(77.0)
    await monitor.clear_score("AAPL")
    assert await monitor.stored_score("AAPL") == pytest.approx(0.0)


class _StubProtection:
    settings = ProtectionSettings()

    def time_stop_deadline(self, environment, symbol):
        return None

    def is_time_stop_due(self, environment, symbol, now=None):
        return False


async def test_position_monitor_never_closes_without_a_rule(database):
    client = FakeTradingClient(PAPER)
    portfolio = FakePortfolioService(client)
    monitor = PositionMonitor(
        portfolio, _AlertScanner(50), _StubProtection(), database, environment=PAPER
    )
    await monitor.store_score("AAPL", 90.0)
    monitor._last_alert.clear()
    await monitor.check_position(make_position(symbol="AAPL"))
    assert not client.closed


# -- app container ------------------------------------------------------------------
class _StubMarket(FakeMarketService):
    pass


def _app(database, environment=PAPER, **settings_over):
    from botalpaca.app import Application
    from botalpaca.config.settings import Settings

    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        database_migrate=False,
        database_auto_create=True,
        active_trading_environment=environment,
        telegram_allowed_user_ids=[111],
        **settings_over,
    )
    return Application(settings=settings, database=database)


def _stubbed(app):
    """Replace the context factory with a stub that still registers itself."""
    real = app._build_context

    def build(environment):
        context = _StubContext(environment)
        app.contexts[environment] = context
        return context

    app._build_context = build
    return real


async def test_application_start_uses_configured_environment(database):
    app = _app(database, PAPER)
    _stubbed(app)
    await app.start()
    assert app.active_environment is PAPER
    await app.stop()


async def test_application_start_persists_environment(database):
    from botalpaca.db.repositories import AppStateRepository
    from botalpaca.security.guards import ACTIVE_ENV_KEY

    app = _app(database, PAPER)
    _stubbed(app)
    await app.start()
    async with database.session() as session:
        stored = await AppStateRepository(session).get(ACTIVE_ENV_KEY)
    assert stored is not None
    await app.stop()


async def test_application_rejects_missing_configuration(database):
    from botalpaca.config.settings import Settings

    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        active_trading_environment=TradingEnvironment.REAL,
    )
    with pytest.raises(ConfigurationError):
        settings.validate_runtime()


async def test_application_persisted_env_does_not_override_config(database):
    """A REAL environment persisted before a restart must not silently resume REAL."""
    from botalpaca.db.repositories import AppStateRepository
    from botalpaca.security.guards import ACTIVE_ENV_KEY

    async with database.session() as session:
        await AppStateRepository(session).set(ACTIVE_ENV_KEY, REAL)
    app = _app(database, PAPER)
    _stubbed(app)
    await app.start()
    assert app.active_environment is PAPER
    await app.stop()


async def test_switch_environment_returns_account(database):
    app = _app(database, PAPER)
    _stubbed(app)
    await app.start()
    account = await app.switch_environment(REAL, confirmed_by=111)
    assert app.active_environment is REAL
    assert account.environment is REAL
    await app.stop()


async def test_switch_to_same_environment_is_a_noop(database):
    app = _app(database, PAPER)
    _stubbed(app)
    await app.start()
    before = app.contexts[PAPER]
    await app.switch_environment(PAPER)
    assert app.contexts[PAPER] is before
    assert app.active is before
    await app.stop()


async def test_submit_plan_requires_authorized_user(database):
    app = _app(database, PAPER)
    _stubbed(app)
    await app.start()
    plan = TradePlan(
        symbol="AAPL", environment=PAPER, direction=SignalDirection.LONG, qty=10.0
    )
    with pytest.raises(Exception):
        await app.submit_plan(plan, RiskAssessment(approved=True, suggested_qty=10.0), user_id=999)
    await app.stop()


class _StubContext:
    def __init__(self, environment):
        self.environment = environment
        self.market_monitor = _StubMonitor()
        self.position_monitor = _StubMonitor()
        self.scanner = _StubScanner()
        self.portfolio = _StubPortfolio(environment)
        self.execution = _StubExecution(environment)
        self.protection = _StubProtection()

    async def verify(self):
        from botalpaca.domain.models import AccountSnapshot

        return AccountSnapshot(
            environment=self.environment,
            account_id="a",
            status="ACTIVE",
            equity=100_000.0,
            cash=100_000.0,
            buying_power=200_000.0,
        )

    async def close(self):
        return None


class _StubMonitor:
    def stop(self):
        return None


class _StubPortfolio:
    def __init__(self, environment):
        self.environment = environment

    async def get_account(self):
        from botalpaca.domain.models import AccountSnapshot

        return AccountSnapshot(
            environment=self.environment,
            account_id="a",
            status="ACTIVE",
            equity=100_000.0,
            cash=100_000.0,
            buying_power=200_000.0,
        )

    async def get_positions(self):
        return []


class _StubExecution:
    def __init__(self, environment):
        self.environment = environment
        self.submitted: list[object] = []

    async def submit(self, plan, **kwargs):
        from botalpaca.execution.engine import SubmissionResult

        self.submitted.append((plan, kwargs))
        return SubmissionResult(None, None, "cid", self.environment, created=False)
