"""Tests for the autonomous market panel and unattended position protection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from botalpaca.config import ProtectionSettings
from botalpaca.domain.enums import TradingEnvironment
from botalpaca.monitoring.autonomy import (
    ACTION_LABELS,
    AutonomousProtector,
    AutonomyAction,
    AutonomyReport,
)

PAPER = TradingEnvironment.PAPER
REAL = TradingEnvironment.REAL


class _Position:
    def __init__(self, symbol="AAPL", qty=10.0, entry=100.0, current=104.0, pl=40.0):
        self.symbol = symbol
        self.qty = qty
        self.side = "buy"
        self.avg_entry_price = entry
        self.current_price = current
        self.market_value = current * abs(qty)
        self.unrealized_pl = pl
        self.unrealized_plpc = 0.04


def _state(**kw):
    base = dict(
        has_stop=True,
        stop_price=97.0,
        has_take_profit=False,
        take_profit_price=None,
        has_trailing=False,
        trail_percent=None,
        break_even_active=False,
        time_stop_at=None,
        notes=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Protection:
    """Records every call the protector makes."""

    def __init__(self, state=None, atr=2.0):
        self.state = state if state is not None else _state()
        self.calls: list[tuple[str, dict]] = []
        self.atr = atr

    async def state_for_position(self, environment, position):
        return self.state

    async def is_time_stop_due(self, environment, symbol):
        return False

    async def ensure_stop(self, *, environment, position, reason, atr=None):
        self.calls.append(("ensure_stop", {"reason": reason}))
        return _state(has_stop=True, stop_price=position.current_price * 0.98)

    async def move_to_break_even(self, *, environment, position):
        self.calls.append(("move_to_break_even", {}))
        self.state = _state(has_stop=True, stop_price=position.avg_entry_price * 1.0015)
        return self.state

    async def progressive_step(self, *, environment, position, step_pct=1.0):
        self.calls.append(("progressive_step", {}))
        self.state = _state(has_stop=True, stop_price=position.current_price * 0.99)
        return self.state

    async def enable_trailing_stop(self, *, environment, position, trail_percent=None, atr=None):
        self.calls.append(("enable_trailing_stop", {}))
        self.state = _state(has_stop=False, has_trailing=True, trail_percent=2.5)
        return self.state


# -- AutonomyReport -------------------------------------------------------------------
def test_every_action_has_a_label():
    assert set(ACTION_LABELS) == set(AutonomyAction)


def test_report_labels_itself():
    report = AutonomyReport(
        action=AutonomyAction.BREAK_EVEN,
        symbol="AAPL",
        environment=PAPER,
        position=None,
        detail="supero 1R",
        r_multiple=1.4,
        new_stop=100.15,
    )
    assert report.label == ACTION_LABELS[AutonomyAction.BREAK_EVEN]
    assert report.is_profitable is True
    assert "AAPL" in report.render()
    assert "1.40R" in report.render()


def test_report_knows_a_losing_trade():
    report = AutonomyReport(
        action=AutonomyAction.MOMENTUM_EXIT,
        symbol="TSLA",
        environment=REAL,
        position=None,
        detail="momentum roto",
        r_multiple=-1.2,
    )
    assert report.is_profitable is False


# -- rules ----------------------------------------------------------------------------
async def test_naked_position_gets_a_stop_first():
    protection = _Protection(state=_state(has_stop=False, stop_price=None))
    protector = AutonomousProtector(protection, environment=PAPER)
    reports = await protector.evaluate(_Position())
    assert [r.action for r in reports] == [AutonomyAction.STOP_CREATED]
    assert protection.calls[0][0] == "ensure_stop"


async def test_break_even_fires_at_the_trigger():
    protection = _Protection()
    protector = AutonomousProtector(protection, environment=PAPER)
    # entry 100, stop 97 => risk 3/share; current 104 => +1.33R
    reports = await protector.evaluate(_Position())
    assert [r.action for r in reports] == [AutonomyAction.BREAK_EVEN]
    assert protection.calls[0][0] == "move_to_break_even"


async def test_break_even_does_not_repeat_once_applied():
    protection = _Protection(
        state=_state(has_stop=True, stop_price=100.2, break_even_active=True)
    )
    protector = AutonomousProtector(protection, environment=PAPER)
    reports = await protector.evaluate(_Position())
    assert all(r.action != AutonomyAction.BREAK_EVEN for r in reports)
    assert all(call[0] != "move_to_break_even" for call in protection.calls)


async def test_trailing_arms_after_its_trigger():
    protection = _Protection(state=_state(has_stop=True, stop_price=98.0))
    settings = ProtectionSettings(auto_break_even=False, auto_progressive=False)
    protector = AutonomousProtector(protection, environment=PAPER, settings=settings)
    # entry 100, stop 98 => risk 2; current 104 => +2R, above trailing_trigger_r 1.5
    reports = await protector.evaluate(_Position())
    assert [r.action for r in reports] == [AutonomyAction.TRAILING]
    assert reports[0].trail_percent == 2.5


async def test_trailing_does_not_arm_below_its_trigger():
    protection = _Protection(state=_state(has_stop=True, stop_price=97.0))
    settings = ProtectionSettings(auto_break_even=False, auto_progressive=False)
    protector = AutonomousProtector(protection, environment=PAPER, settings=settings)
    # entry 100, stop 97 => risk 3/share; current 101 => +0.33R, below the 1.5R
    # trigger, so the trailing stop must not be armed yet.
    reports = await protector.evaluate(_Position(current=101.0, pl=10.0))
    assert reports == []


async def test_autonomy_can_be_switched_off():
    protection = _Protection(state=_state(has_stop=False, stop_price=None))
    settings = ProtectionSettings(auto_break_even=False)
    protector = AutonomousProtector(protection, environment=PAPER, settings=settings)
    # Even with break-even off, the naked-position rule is unconditional: leaving
    # a position without a stop is never acceptable.
    reports = await protector.evaluate(_Position())
    assert [r.action for r in reports] == [AutonomyAction.STOP_CREATED]


async def test_autonomy_never_fires_on_a_flat_position():
    protection = _Protection()
    protector = AutonomousProtector(protection, environment=PAPER)
    assert await protector.evaluate(_Position(qty=0.0)) == []


async def test_broker_failure_is_swallowed_and_logged():
    class _Broken(_Protection):
        async def move_to_break_even(self, *, environment, position):
            raise RuntimeError("Alpaca caido")

    protector = AutonomousProtector(_Broken(), environment=PAPER)
    reports = await protector.evaluate(_Position())
    assert reports == []


async def test_every_firing_notifies_the_operator():
    sent: list[str] = []

    async def notify(text: str) -> None:
        sent.append(text)

    protection = _Protection()
    protector = AutonomousProtector(protection, environment=PAPER, notify=notify)
    await protector.evaluate(_Position())
    assert sent, "an autonomous action must always be reported"
    assert "AAPL" in sent[0]


async def test_notify_cooldown_prevents_spam():
    sent: list[str] = []

    async def notify(text: str) -> None:
        sent.append(text)

    settings = ProtectionSettings(autonomy_notify_cooldown_minutes=30)
    protection = _Protection(state=_state(has_stop=False, stop_price=None))
    protector = AutonomousProtector(protection, environment=PAPER, settings=settings, notify=notify)
    await protector.evaluate(_Position())
    await protector.evaluate(_Position())
    assert len(sent) == 1


async def test_notify_failure_does_not_stop_protection():
    async def notify(text: str) -> None:
        raise RuntimeError("Telegram caido")

    protection = _Protection()
    protector = AutonomousProtector(protection, environment=PAPER, notify=notify)
    reports = await protector.evaluate(_Position())
    assert [r.action for r in reports] == [AutonomyAction.BREAK_EVEN]


# -- settings defaults ----------------------------------------------------------------
def test_autonomy_is_on_by_default():
    settings = ProtectionSettings()
    assert settings.auto_break_even is True
    assert settings.auto_progressive is True
    assert settings.auto_trailing is True
    assert settings.auto_time_stop is True


def test_autonomous_exit_is_off_by_default():
    """Closing on a momentum collapse realises a loss, so it stays opt-in."""
    assert ProtectionSettings().auto_momentum_exit is False


# -- the panel keyboard ---------------------------------------------------------------
def test_signal_keyboard_is_compact():
    """One row of decisions, not a wall of buttons."""
    from botalpaca.telegram import keyboards as kb

    markup = kb.signal_keyboard("AAPL")
    payloads = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert f"{kb.ACCEPT_SIGNAL}:AAPL" in payloads
    assert f"{kb.REJECT_SIGNAL}:AAPL" in payloads
    # Accept and cancel sit on the same row: they are the decision.
    first_row = [b.callback_data for b in markup.inline_keyboard[0]]
    assert any(p.startswith(kb.ACCEPT_SIGNAL) for p in first_row)
    assert any(p.startswith(kb.REJECT_SIGNAL) for p in first_row)


@pytest.mark.parametrize("command", ["analizar"])
def test_panel_command_is_registered(command):
    from botalpaca.telegram.bot import COMMAND_DESCRIPTIONS

    assert command in COMMAND_DESCRIPTIONS


def test_oportunidades_command_was_removed():
    from botalpaca.telegram.bot import COMMAND_DESCRIPTIONS

    assert "oportunidades" not in COMMAND_DESCRIPTIONS


def test_accept_button_routes_to_the_direct_handler():
    """Accepting must not stage a second confirmation in PAPER."""
    import inspect

    from botalpaca.telegram import service

    source = inspect.getsource(service.TelegramFacade.aceptar)
    assert "submit_plan" in source
    assert "self.environment.is_real" in source, "REAL must still confirm"
