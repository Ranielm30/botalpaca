"""An inert order must be swept even when the position looks protected.

Alpaca left a bracket child in HELD: it reserves the shares, cannot execute,
and never appears in an OPEN query. The position still had a live stop, so
``ensure_stop`` returned early and the inert order sat there for the rest of the
trade -- which means every future exit on that position is rejected with
``insufficient qty available`` naming the order nobody can see.

That is how a protected position silently became unprotectable.
"""

from __future__ import annotations

import pytest

from botalpaca.execution.mapping import to_order_state
from tests.conftest import fake_order_raw
from tests.test_protection import PAPER, _setup, _stop_order, make_position


def _raw(order, status: str):
    order.status = status
    return order


def _held_stop(order_id: str):
    return _raw(_stop_order(order_id, stop=327.14), "held")


def _live_stop(order_id: str, stop: float):
    return _raw(_stop_order(order_id, stop=stop), "new")


async def test_a_protected_position_still_sheds_its_inert_orders(database):
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)

    client.orders["live-1"] = _live_stop("live-1", 99.0)
    client.orders["stale-1"] = _held_stop("stale-1")

    state = await protection.state_for_position(PAPER, position)
    assert state.has_stop is True
    assert "stale-1" in state.inert_order_ids

    returned = await protection.ensure_stop(environment=PAPER, position=position)

    # The live stop is still the protection; the inert one is gone either way.
    assert returned.has_stop is True
    assert "stale-1" in client.cancelled, (
        "an inert order holds the shares and must be released even when the "
        "position already has a stop"
    )


async def test_reconcile_sweeps_inert_orders_on_a_protected_position(database):
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)

    client.orders["live-1"] = _live_stop("live-1", 99.0)
    client.orders["stale-2"] = _held_stop("stale-2")

    notes = await protection.reconcile(environment=PAPER, positions=[position])

    assert "stale-2" in client.cancelled
    assert any("inertes" in note for note in notes), notes


async def test_a_held_order_is_never_counted_as_protection(database):
    """The core of it: HELD is invisible to an OPEN query, so it is not a stop."""
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)

    client.orders["held-only"] = _held_stop("held-only")
    state = await protection.state_for_position(PAPER, position)

    assert state.has_stop is False
    assert state.stop_price is None
    assert state.inert_order_ids == ["held-only"]


async def test_nothing_inert_means_no_cancel_chatter(database):
    """The sweep must be a no-op on a clean position, not an API call per pass."""
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)
    client.orders["live-1"] = _live_stop("live-1", 99.0)

    await protection.ensure_stop(environment=PAPER, position=position)

    assert client.cancelled == []


async def test_reconcile_leaves_a_trailing_stop_alone(database):
    """A trailing stop IS the protection.

    Adding a fixed stop on top of it is rejected for insufficient qty, because
    Alpaca reserves the shares for the trailing order. That used to surface as
    a failure to protect a position that was already protected.
    """
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=110.0)

    trail = fake_order_raw(
        "t1",
        order_type="trailing_stop",
        side="sell",
        status="new",
        qty=10.0,
    )
    trail.trail_percent = 2.0
    client.orders["t1"] = trail

    state = await protection.state_for_position(PAPER, position)
    assert state.has_trailing is True
    assert state.has_stop is False

    before = list(client.submitted)
    notes = await protection.reconcile(environment=PAPER, positions=[position])

    assert client.submitted == before, "no exit may be added above a trailing stop"
    assert not any("stop" in note.lower() for note in notes), notes


@pytest.mark.parametrize("status", ["pending_cancel", "pending_replace"])
async def test_other_stuck_states_are_inert_too(database, status):
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)

    client.orders["stuck"] = _raw(_stop_order("stuck", stop=99.0), status)
    state = await protection.state_for_position(PAPER, position)

    assert "stuck" in state.inert_order_ids
    assert state.has_stop is False
    assert to_order_state(client.orders["stuck"], PAPER).id == "stuck"
