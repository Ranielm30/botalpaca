"""Automatic execution: the bot operates alone only when it is switched on.

Two things have to hold. The executor must run through the same gates a manual
entry does, and the switch itself must be an explicit, persistent decision --
not something inferred from a flag that was already in the config file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from botalpaca.domain import TradingEnvironment
from botalpaca.monitoring.service import MarketMonitor
from botalpaca.telegram.service import CommandError
from tests.conftest import make_opportunity


def _monitor(database, *, executor=None, enabled=False):
    scanner = SimpleNamespace(scan=lambda **kw: None)
    market = SimpleNamespace(is_market_open=lambda: False)
    return MarketMonitor(
        scanner,
        market,
        database,
        environment=TradingEnvironment.PAPER,
        notify=None,
        executor=executor,
        settings=SimpleNamespace(
            opportunity_alert_min_score=60.0,
            signal_alert_cooldown_seconds=0,
            auto_trading_enabled=enabled,
        ),
    )


async def test_the_executor_does_not_run_when_the_mode_is_off(database):
    """Off by default: the flag in the config file is only the starting value."""
    seen: list[str] = []

    async def executor(opportunity):
        seen.append(opportunity.symbol)

    monitor = _monitor(database, executor=executor, enabled=False)
    await monitor._maybe_alert(make_opportunity())
    assert seen == []


async def test_the_executor_runs_when_the_mode_is_on(database):
    seen: list[str] = []

    async def executor(opportunity):
        seen.append(opportunity.symbol)

    monitor = _monitor(database, executor=executor, enabled=True)
    await monitor._maybe_alert(make_opportunity())
    assert seen == ["AAPL"]


async def test_the_executor_is_not_called_without_one(database):
    """A monitor with no executor cannot execute anything, even when switched on."""
    monitor = _monitor(database, executor=None, enabled=True)
    # Must not raise: the flag being on is not enough on its own.
    await monitor._maybe_alert(make_opportunity())


async def test_a_failure_in_the_executor_does_not_stop_the_scan(database):
    """One rejected opportunity must not take the monitor down with it."""

    async def executor(opportunity):
        raise RuntimeError("the broker said no")

    monitor = _monitor(database, executor=executor, enabled=True)
    await monitor._maybe_alert(make_opportunity())  # must not raise


async def test_the_switch_survives_a_restart(database):
    """The decision is persisted, so it outlives the process that made it."""
    monitor = _monitor(database, executor=None, enabled=False)

    assert await monitor.auto_trading_on() is False
    assert await monitor.set_auto_trading(True) is True
    assert await monitor.auto_trading_on() is True

    # A brand new monitor over the same database sees the same answer.
    reborn = _monitor(database, executor=None, enabled=False)
    assert await reborn.auto_trading_on() is True


async def test_the_config_default_is_used_until_the_operator_chooses(database):
    monitor = _monitor(database, executor=None, enabled=True)
    assert await monitor.auto_trading_on() is True


def test_the_command_is_registered_and_described():
    from botalpaca.telegram.bot import COMMAND_DESCRIPTIONS

    assert COMMAND_DESCRIPTIONS["auto"].strip()
    assert len(COMMAND_DESCRIPTIONS["auto"]) <= 256


def test_the_confirmation_kind_exists():
    from botalpaca.security.confirmation import ConfirmationKind

    assert ConfirmationKind.AUTO.value == "auto"


# --------------------------------------------------------------------- /auto


class _StubMonitor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    async def auto_trading_on(self) -> bool:
        return self.enabled

    async def set_auto_trading(self, flag: bool) -> bool:
        self.enabled = bool(flag)
        return self.enabled


async def _ok():
    return None


class _Conf:
    def __init__(self) -> None:
        self.pending = None

    async def request(self, **kwargs):
        self.pending = kwargs

    async def consume(self, *, user_id, kind):
        if self.pending is None:
            return None
        return SimpleNamespace(
            kind=kind,
            user_id=user_id,
            environment=TradingEnvironment.PAPER,
            details=self.pending.get("details", {}),
        )


class _AutoApp:
    def __init__(self, *, operator=None, enabled=False):
        self.environment = TradingEnvironment.PAPER
        self.active_environment = TradingEnvironment.PAPER
        self.security = SimpleNamespace(
            allowlist=SimpleNamespace(require=lambda uid: uid),
            rate_limiter=SimpleNamespace(
                check=lambda uid: _ok()
            ),
        )
        self.confirmations = _Conf()
        self.active = SimpleNamespace(
            market_monitor=_StubMonitor(enabled),
        )
        self.settings = SimpleNamespace(
            monitoring=SimpleNamespace(auto_trading_user_id=operator)
        )
        self.toggled: list[bool] = []

    async def _set(self, flag):
        self.toggled.append(flag)
        return flag


def _update(user_id=12345):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=user_id),
    )


async def test_auto_shows_the_current_state_without_an_argument():
    from botalpaca.telegram.service import TelegramFacade

    app = _AutoApp(enabled=True)
    facade = TelegramFacade(app)
    result = await facade.auto(_update())
    assert "ACTIVADO" in result.text
    assert app.toggled == []


async def test_auto_on_without_an_operator_is_refused():
    from botalpaca.telegram.service import TelegramFacade

    facade = TelegramFacade(_AutoApp(operator=None))
    with pytest.raises(CommandError, match="AUTO_TRADING_USER_ID"):
        await facade.auto(_update(), "on")


async def test_auto_on_stages_a_confirmation_and_the_button_applies_it():
    from botalpaca.telegram.service import TelegramFacade

    app = _AutoApp(operator=12345)
    facade = TelegramFacade(app)
    result = await facade.auto(_update(), "on")
    assert "modo automático" in result.text.lower()
    assert app.confirmations.pending is not None

    applied = await facade.confirm_auto(_update(), True)
    assert "ACTIVADO" in applied.text


async def test_auto_off_stages_a_confirmation_and_the_button_applies_it():
    from botalpaca.telegram.service import TelegramFacade

    app = _AutoApp(operator=12345)
    facade = TelegramFacade(app)
    await facade.auto(_update(), "off")
    applied = await facade.confirm_auto(_update(), False)
    assert "DESACTIVADO" in applied.text


async def test_a_confirmation_that_never_existed_is_refused():
    from botalpaca.telegram.service import TelegramFacade

    facade = TelegramFacade(_AutoApp(operator=12345))
    with pytest.raises(CommandError, match="confirmación"):
        await facade.confirm_auto(_update(), True)
