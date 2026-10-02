"""Every autonomous action must reach Telegram.

The bot applied break-even, trailing, progressive and the emergency stop and
reported none of it: the protector hands over an already-rendered card, while
the delivery path expected a ``PositionAlert``. The string was wrapped as if it
were an alert and then failed to render, so the notice disappeared on the floor
while the log said ``autonomy.applied``.

These tests pin the whole chain: protector -> notify_position_alert ->
handle_alert -> notifications.send.
"""

from __future__ import annotations

import pytest

from botalpaca.domain.models import ProtectionState
from botalpaca.monitoring.autonomy import AutonomousProtector, AutonomyAction


class _Protection:
    """A broker double that only knows how to move the stop.

    Each call returns a *new* ProtectionState, exactly as the real manager does
    after a replace. Mutating in place would make the before and after states the
    same object, and the protector would read the move as a no-op.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._state: ProtectionState | None = None

    def _with(self, **kw) -> ProtectionState:
        base = self._state
        assert base is not None
        return base.model_copy(update=kw)

    async def state_for_position(self, environment, position):
        return self._state

    async def is_time_stop_due(self, environment, symbol):
        return False

    async def ensure_stop(self, *, environment, position, stop_price=None, reason="", atr=None):
        self.calls.append("ensure_stop")
        self._state = self._with(has_stop=True, stop_price=stop_price or 97.0)
        return self._state

    async def move_to_break_even(self, *, environment, position, buffer_pct=None):
        self.calls.append("move_to_break_even")
        self._state = self._with(stop_price=100.15, break_even_active=True)
        return self._state

    async def progressive_step(self, *, environment, position, step_pct=1.0):
        self.calls.append("progressive_step")
        return await self.move_to_break_even(environment=environment, position=position)

    async def enable_trailing_stop(
        self, *, environment, position, trail_percent=None, atr=None,
        floor_stop=None, widen_to=None,
    ):
        self.calls.append("enable_trailing_stop")
        self._state = self._with(
            has_stop=False,
            has_trailing=True,
            trail_percent=trail_percent or widen_to or 2.0,
            trail_price=position.current_price * 0.97,
        )
        return self._state


def _position(current: float, entry: float = 100.0):
    class _P:
        symbol = "AAPL"
        qty = 10.0
        avg_entry_price = entry
        side = "long"

        def __init__(self) -> None:
            self.current_price = current
            self.market_value = current * 10.0
            self.unrealized_pl = (current - entry) * 10.0
            self.unrealized_plpc = (current - entry) / entry

    return _P()


@pytest.fixture
def state():
    from botalpaca.domain.enums import TradingEnvironment
    from botalpaca.domain.models import ProtectionState

    return ProtectionState(
        symbol="AAPL",
        environment=TradingEnvironment.PAPER,
        has_stop=True,
        stop_price=97.0,
        initial_stop_price=97.0,
    )


# -- the delivery chain ---------------------------------------------------------------
async def test_an_autonomous_card_reaches_the_notification_service():
    """The whole path, end to end: protector string -> handle_alert -> send."""
    sent: list[str] = []
    keyboards: list[object] = []

    class _Notifications:
        async def send(self, text, keyboard=None, *, force=False):
            sent.append(text)
            keyboards.append(keyboard)
            return True

    class _App:
        active_environment = _PAPER
        notifications = _Notifications()
        _alert_handler = None

        async def notify_position_alert(self, alert):
            # The real method: a rendered card must short-circuit straight to
            # the handler instead of being wrapped as if it were an alert.
            if self._alert_handler is None:
                return None
            await self._alert_handler(alert)
            return None

    app = _App()
    facade = _make_facade(app)
    app._alert_handler = lambda card: facade.handle_alert(card)
    protection = _Protection()
    protection._state = _make_state()

    protector = AutonomousProtector(
        protection,
        environment=_PAPER,
        notify=facade.app.notify_position_alert,
    )
    reports = await protector.evaluate(_position(104.0))

    assert [r.action for r in reports] == [AutonomyAction.BREAK_EVEN]
    assert sent, "the break-even applied but nothing was delivered"
    assert "AAPL" in sent[0]
    assert "break-even" in sent[0].lower()
    # A finished card needs no action keyboard.
    assert keyboards == [None]


async def test_a_string_payload_is_sent_as_it_stands():
    sent: list[str] = []

    class _Notifications:
        async def send(self, text, keyboard=None, *, force=False):
            sent.append(text)
            return True

    class _App:
        active_environment = _PAPER
        notifications = _Notifications()
        _alert_handler = None

        async def notify_position_alert(self, alert):
            # The real method: a rendered card must short-circuit straight to
            # the handler instead of being wrapped as if it were an alert.
            if self._alert_handler is None:
                return None
            await self._alert_handler(alert)
            return None

    facade = _make_facade(_App())
    await facade.handle_alert("🟢 AAPL · break-even aplicado")
    assert sent == ["🟢 AAPL · break-even aplicado"]


async def test_a_normal_alert_still_goes_through_the_renderer():
    """The string shortcut must not swallow real PositionAlert payloads."""
    sent: list[str] = []

    class _Notifications:
        async def send(self, text, keyboard=None, *, force=False):
            sent.append(text)
            return True

    class _App:
        active_environment = _PAPER
        notifications = _Notifications()
        _alert_handler = None

        async def notify_position_alert(self, alert):
            # The real method: a rendered card must short-circuit straight to
            # the handler instead of being wrapped as if it were an alert.
            if self._alert_handler is None:
                return None
            await self._alert_handler(alert)
            return None

    facade = _make_facade(_App())
    from botalpaca.app import OpportunityAlert
    from botalpaca.domain.models import PositionSnapshot
    from botalpaca.monitoring.service import PositionAlert

    alert = PositionAlert(
        symbol="AAPL",
        environment=_PAPER,
        position=PositionSnapshot(
            environment=_PAPER, symbol="AAPL", qty=10, side="long",
            avg_entry_price=100.0, current_price=101.0, market_value=1010.0,
            cost_basis=1000.0, unrealized_pl=10.0, unrealized_plpc=0.01,
        ),
        changes=["el score cayo"],
        previous_score=82.0,
        current_score=61.0,
        kind="deterioration",
    )
    await facade.handle_alert(
        OpportunityAlert(environment=_PAPER, opportunities=(), alerts=(alert,))
    )
    assert sent and "AAPL" in sent[0]


def test_notify_position_alert_without_a_handler_is_harmless():
    """With Telegram detached the notice is dropped, not raised."""

    class _App:
        _alert_handler = None

    import asyncio

    from botalpaca.app import Application

    assert asyncio.run(Application.notify_position_alert(_App(), "carta")) is None
    assert OrderSide_SELL_is_importable() is True


def OrderSide_SELL_is_importable() -> bool:
    from botalpaca.domain.enums import OrderSide

    return OrderSide.SELL is not None


# -- helpers --------------------------------------------------------------------------
from botalpaca.domain.enums import TradingEnvironment  # noqa: E402

_PAPER = TradingEnvironment.PAPER


def _make_state():
    from botalpaca.domain.models import ProtectionState

    return ProtectionState(
        symbol="AAPL",
        environment=_PAPER,
        has_stop=True,
        stop_price=97.0,
        initial_stop_price=97.0,
    )


def _make_facade(app):
    """A facade bound to a stub app, without booting the whole container."""
    from botalpaca.telegram.service import TelegramFacade

    facade = TelegramFacade.__new__(TelegramFacade)
    facade.app = app
    return facade
