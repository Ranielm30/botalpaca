"""The R baseline, and the ways break-even and trailing used to collide.

Every case here came from a live PAPER observation, not from theory:

* A trailing stop replaced the fixed stop, so ``stop_price`` became None, the
  engine fell back to a percent move, and a +1.4R trade scored 0.44 against a
  threshold of 1.0 -- so nothing fired again, ever.
* The trailing was armed at 2% from price, landing at 327.43 on a position whose
  break-even had already been secured at 332.45.
* Break-even's replace left the old stop NEW, so one position carried two stops
  and the looser one could still close it at a loss.
* Progressive kept calling move_to_break_even on every pass once a trailing stop
  existed, cancelling the very protection that was running.
"""

from __future__ import annotations

import pytest

from botalpaca.config import ProtectionSettings
from botalpaca.domain.errors import ValidationError
from botalpaca.domain.models import ProtectionState
from botalpaca.execution.mapping import to_order_state
from botalpaca.monitoring.autonomy import AutonomousProtector, AutonomyAction
from tests.test_autonomy import _state
from tests.test_protection import PAPER, _setup, _stop_order, make_position


class _RecordingManager:
    """Stands in for the protection manager and records what it is asked to do.

    The trailing maths is modelled the way Alpaca does it -- a percentage of the
    price -- so a floor that is too loose can actually be seen to fail rather
    than being asserted in the abstract.
    """

    def __init__(self, state, *, current_price: float, entry: float):
        self.state = state
        self.current_price = current_price
        self.entry = entry
        self.trail_calls: list[dict] = []
        self.break_even_calls = 0
        self.progressive_calls = 0
        self.rejected: list[float] = []

    async def state_for_position(self, environment, position):
        return self.state

    async def is_time_stop_due(self, environment, symbol):
        return False

    async def ensure_stop(self, *, environment, position, stop_price=None, reason="", atr=None):
        self.state = _state(
            has_stop=True,
            stop_price=stop_price,
            initial_stop_price=self.state.initial_stop_price,
        )
        return self.state

    async def move_to_break_even(self, *, environment, position, buffer_pct=None):
        self.break_even_calls += 1
        self.state = _state(
            has_stop=True,
            stop_price=round(self.entry * 1.0015, 2),
            initial_stop_price=self.state.initial_stop_price,
            break_even_active=True,
        )
        return self.state

    async def progressive_step(self, *, environment, position, step_pct=1.0):
        self.progressive_calls += 1
        return await self.move_to_break_even(environment=environment, position=position)

    async def enable_trailing_stop(
        self, *, environment, position, trail_percent=None, atr=None,
        floor_stop=None, widen_to=None,
    ):
        self.trail_calls.append({"floor_stop": floor_stop, "widen_to": widen_to})
        if widen_to is not None:
            percent = max(float(widen_to), 2.0)
        else:
            percent = max(trail_percent or 2.0, 2.0)
        # Reproduce the real refusal: no trail can be tighter than the floor.
        if floor_stop is not None:
            trail_price = round(self.current_price * (1 - percent / 100.0), 2)
            if trail_price <= floor_stop:
                self.rejected.append(trail_price)
                raise ValidationError(
                    f"el trailing en {trail_price} queda por debajo del "
                    f"break-even {floor_stop}"
                )
        else:
            trail_price = round(self.current_price * (1 - percent / 100.0), 2)
        self.state = _state(
            has_stop=False,
            has_trailing=True,
            trail_percent=percent,
            trail_price=trail_price,
            initial_stop_price=self.state.initial_stop_price,
        )
        return self.state


def _manager(
    *, initial_stop: float, live_stop: float | None, entry: float, current: float,
    **state_kw,
):
    state = _state(
        has_stop=live_stop is not None,
        stop_price=live_stop,
        initial_stop_price=initial_stop,
        **state_kw,
    )
    return _RecordingManager(state, current_price=current, entry=entry)


@pytest.fixture
def notifying():
    sent: list[str] = []

    async def notify(text: str) -> None:
        sent.append(text)

    return sent, notify


# -- the frozen baseline -------------------------------------------------------------
async def test_r_uses_the_entry_risk_not_the_moved_stop():
    """+4 on a risk of 3 is +1.33R, which must not arm a 1.5R trailing stop.

    Measured against the moved stop (0.5 points from entry) it would read as
    +8R and fire on a trade that has barely moved.
    """
    manager = _manager(initial_stop=97.0, live_stop=99.5, entry=100.0, current=104.0)
    protector = AutonomousProtector(
        manager,
        environment=PAPER,
        settings=ProtectionSettings(auto_break_even=False, auto_progressive=False),
    )
    reports = await protector.evaluate(make_position(entry=100.0, current=104.0))
    assert reports == []
    assert manager.trail_calls == []


async def test_trailing_still_arms_when_the_baseline_survives_the_stop_moving(notifying):
    manager = _manager(
        initial_stop=98.0,
        live_stop=98.5,
        entry=100.0,
        current=104.0,
        break_even_active=True,
    )
    sent, notify = notifying
    protector = AutonomousProtector(
        manager,
        environment=PAPER,
        settings=ProtectionSettings(auto_break_even=False, auto_progressive=False),
        notify=notify,
    )
    reports = await protector.evaluate(make_position(entry=100.0, current=104.0))
    assert [r.action for r in reports] == [AutonomyAction.TRAILING]
    assert sent, "arming a trailing stop must be reported"


# -- the floor -----------------------------------------------------------------------
async def test_trailing_is_never_armed_below_break_even(notifying):
    """A 2% trail from 340 is 333.20; the break-even floor is 332.45."""
    manager = _manager(
        initial_stop=327.14, live_stop=332.45, entry=331.948, current=340.0,
        break_even_active=True,
    )
    sent, notify = notifying
    protector = AutonomousProtector(
        manager,
        environment=PAPER,
        settings=ProtectionSettings(auto_break_even=False, auto_progressive=False),
        notify=notify,
    )
    reports = await protector.evaluate(make_position(entry=331.948, current=340.0))
    floor = manager.trail_calls[0]["floor_stop"]
    assert floor == pytest.approx(332.45, abs=0.01)
    assert manager.state.trail_price > floor
    assert manager.rejected == []
    assert [r.action for r in reports] == [AutonomyAction.TRAILING]


async def test_a_losing_trade_does_not_arm_a_trailing_stop():
    manager = _manager(
        initial_stop=327.14, live_stop=327.14, entry=331.948, current=331.4,
        break_even_active=True,
    )
    protector = AutonomousProtector(manager, environment=PAPER)
    reports = await protector.evaluate(make_position(entry=331.948, current=331.4))
    assert manager.trail_calls == []
    assert reports == []


# -- trailing owns the position ------------------------------------------------------
async def test_progressive_does_nothing_once_the_trailing_owns_it():
    """Otherwise each pass cancels the trailing stop and rebuilds a fixed one."""
    manager = _manager(
        initial_stop=327.14, live_stop=None, entry=331.948, current=340.0,
        has_trailing=True, trail_percent=2.0, trail_price=330.0,
    )
    protector = AutonomousProtector(manager, environment=PAPER)
    await protector.evaluate(make_position(entry=331.948, current=340.0))
    assert manager.break_even_calls == 0
    assert manager.progressive_calls == 0
    assert manager.trail_calls == []


async def test_progressive_step_is_inert_under_a_trailing_stop(database):
    """The manager itself must refuse, not only the caller."""
    manager, _engine, _client = await _setup(database)
    protection = manager

    async def fake_state(environment, pos):
        # The real model, so the guard is exercised against a fully populated
        # state rather than a stub that can drift from it.
        return ProtectionState(
            symbol=pos.symbol,
            environment=PAPER,
            has_trailing=True,
            trailing_order_id="t1",
            trail_percent=2.0,
            trail_price=330.0,
            initial_stop_price=327.14,
        )

    protection.state_for_position = fake_state  # type: ignore[method-assign]

    position = make_position(entry=331.948, current=340.0)
    # Neither call may build a fixed stop underneath the trailing one.
    assert await protection.progressive_step(environment=PAPER, position=position) is None
    result = await protection.move_to_break_even(environment=PAPER, position=position)
    assert result.has_stop is False
    assert result.stop_price is None
    assert any("Trailing activo" in note for note in result.notes)


async def test_a_trailing_squeezed_by_a_gap_gets_widened(notifying):
    """A trail 0.2% under the price would exit on noise, not on a real move."""
    manager = _manager(
        initial_stop=327.14, live_stop=None, entry=331.948, current=331.4,
        has_trailing=True, trail_percent=2.0, trail_price=330.74,
    )
    sent, notify = notifying
    protector = AutonomousProtector(manager, environment=PAPER, notify=notify)
    reports = await protector.evaluate(make_position(entry=331.948, current=331.4))
    assert [r.action for r in reports] == [AutonomyAction.TRAILING_WIDENED]
    assert manager.trail_calls[0]["widen_to"] == 4.0
    assert manager.state.trail_percent == 4.0
    assert sent


async def test_a_healthy_trailing_is_left_alone():
    manager = _manager(
        initial_stop=327.14, live_stop=None, entry=331.948, current=331.4,
        has_trailing=True, trail_percent=2.0, trail_price=327.43,
    )
    protector = AutonomousProtector(manager, environment=PAPER)
    reports = await protector.evaluate(make_position(entry=331.948, current=331.4))
    assert manager.trail_calls == []
    assert reports == []


# -- the stale stop ------------------------------------------------------------------
def _raw(order, status: str):
    order.status = status
    return order


def _state_of(raw):
    return to_order_state(raw, PAPER)


async def test_break_even_cancels_a_stop_that_survived_the_replace(database):
    """Alpaca returned a new id and left the old one NEW: two stops, one live."""
    from botalpaca.domain.models import OrderState

    manager, engine, _ = await _setup(database)
    protection, client = manager, engine._client
    position = make_position(entry=100.0, current=105.0)
    await protection.register_entry_protection(
        environment=PAPER,
        symbol=position.symbol,
        qty=10,
        entry_price=100.0,
        initial_stop_price=97.0,
    )

    replaced_id = "s2"

    async def fake_replace(order_id, request, *, environment):
        return type("Order", (), {"id": replaced_id})()

    async def fake_live(symbol):
        # The old order is still there, still NEW: the broker did not retire it.
        return [
            _state_of(_raw(_stop_order("s2", stop=100.15), "new")),
            _state_of(_raw(_stop_order("s1", stop=97.0), "new")),
        ]

    async def fake_cancel(order_id, *, environment):
        client.cancelled.append(order_id)

    engine.replace_order = fake_replace  # type: ignore[method-assign]
    engine.get_live_orders_for_symbol = fake_live  # type: ignore[method-assign]
    engine.cancel_order = fake_cancel  # type: ignore[method-assign]

    updated = await protection.move_to_break_even(environment=PAPER, position=position)

    assert updated.break_even_active is True
    assert updated.stop_price == 100.15
    assert "s1" in client.cancelled, "the superseded stop must be cancelled"
    assert OrderState is not None


async def test_a_retired_old_stop_is_left_alone(database):
    manager, engine, _ = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=105.0)
    await protection.register_entry_protection(
        environment=PAPER, symbol=position.symbol, qty=10,
        entry_price=100.0, initial_stop_price=97.0,
    )

    async def fake_replace(order_id, request, *, environment):
        return type("Order", (), {"id": "s2"})()

    async def fake_live(symbol):
        return [_state_of(_raw(_stop_order("s2", stop=100.15), "new"))]

    engine.replace_order = fake_replace  # type: ignore[method-assign]
    engine.get_live_orders_for_symbol = fake_live  # type: ignore[method-assign]

    await protection.move_to_break_even(environment=PAPER, position=position)
    # Nothing stale was found, so nothing was cancelled.
    assert not protection._engine._client.cancelled


# -- the baseline is frozen ----------------------------------------------------------
async def test_the_entry_risk_is_never_overwritten(database):
    manager, engine, _ = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)
    await protection.register_entry_protection(
        environment=PAPER, symbol=position.symbol, qty=10,
        entry_price=100.0, initial_stop_price=97.0,
    )
    # The stop is ratcheted twice. The baseline must not follow it.
    for moved in (100.15, 101.50):
        async def fake_replace(order_id, request, *, environment, _m=moved):
            return type("Order", (), {"id": f"s-{_m}"})()

        async def fake_live(symbol, _m=moved):
            return [_state_of(_raw(_stop_order(f"s-{_m}", stop=_m), "new"))]

        async def fake_cancel(order_id, *, environment):
            pass

        engine.replace_order = fake_replace  # type: ignore[method-assign]
        engine.get_live_orders_for_symbol = fake_live  # type: ignore[method-assign]
        engine.cancel_order = fake_cancel  # type: ignore[method-assign]
        await protection.move_to_break_even(environment=PAPER, position=position)

    state = await protection.state_for_position(PAPER, position)
    assert state.initial_stop_price == 97.0
