from __future__ import annotations

import pytest

from botalpaca.domain.enums import TradingEnvironment
from botalpaca.domain.errors import ValidationError

from .test_protection import _setup, _stop_order, _tp_order, make_position

PAPER = TradingEnvironment.PAPER


async def test_a_live_take_profit_is_merged_into_an_oco_not_dropped(database):
    """The target used to be cancelled to make room for the stop.

    That was written when the only known shape was "one independent exit per
    position", so the manager sacrificed the target on purpose. Alpaca's OCO
    holds both at once -- parent take-profit limit, held stop child -- which was
    live-verified on the paper account, so releasing the target is no longer
    necessary and quietly threw away the promised profit.
    """
    manager, engine, client = await _setup(database, orders=[_tp_order("t1")])
    position = make_position(entry=100.0, current=101.0)

    state = await manager.ensure_stop(
        environment=PAPER, position=position, stop_price=97.0, reason="test"
    )

    # The OCO replaces the standalone target, so that order is released, but the
    # target price survives as the group's parent limit.
    assert "t1" in client.cancelled, "the standalone target must be folded into the group"
    assert state.has_stop is True
    assert state.has_take_profit is True
    assert state.take_profit_price == 106.0
    assert any("OCO" in n for n in state.notes), state.notes
    submitted = [r for r in client.submitted if getattr(r, "order_class", None)]
    assert submitted, "an OCO group must have been submitted"


async def test_a_failed_merge_still_refuses_to_leave_the_position_bare(database):
    """A merge that cannot happen must not become a reason to stay unprotected.

    The old code raised as soon as the standalone target resisted the cancel,
    because a stop was the only exit it believed in. Now the group is the goal
    but a bare stop is still better than nothing, so the failure falls through
    to one and only raises if that fails too.
    """
    manager, engine, client = await _setup(database, orders=[_tp_order("t1")])
    position = make_position(entry=100.0, current=101.0)

    async def refuse(order_id, environment=None):
        raise RuntimeError("Alpaca rejected the request")

    engine.cancel_order = refuse

    state = await manager.ensure_stop(
        environment=PAPER, position=position, stop_price=97.0, reason="test"
    )
    assert state.has_stop is True, "a failed merge must still leave a stop"
    assert state.stop_price == 97.0


async def test_a_failed_merge_and_a_failed_stop_is_reported_loudly(database):
    """Both routes failing is the only case that may raise."""
    manager, engine, client = await _setup(database, orders=[_tp_order("t1")])
    position = make_position(entry=100.0, current=101.0)

    async def refuse_cancel(order_id, environment=None):
        raise RuntimeError("Alpaca rejected the request")

    async def refuse_submit(*, environment, request, symbol=None):
        raise RuntimeError("insufficient qty available")

    engine.cancel_order = refuse_cancel
    engine.submit_protective = refuse_submit

    with pytest.raises(ValidationError) as excinfo:
        await manager.ensure_stop(
            environment=PAPER, position=position, stop_price=97.0, reason="test"
        )
    assert "SIN STOP" in str(excinfo.value)


async def test_the_stop_is_left_alone_when_one_already_exists(database):
    """A target must never displace a working stop."""
    manager, _, client = await _setup(
        database, orders=[_stop_order("s1"), _tp_order("t1")]
    )
    position = make_position(entry=100.0, current=101.0)

    state = await manager.ensure_stop(
        environment=PAPER, position=position, stop_price=97.0, reason="test"
    )

    assert state.has_stop is True
    assert client.cancelled == [], "nothing should be cancelled when protected"
