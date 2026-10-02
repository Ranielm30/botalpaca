from __future__ import annotations

import pytest

from botalpaca.domain.enums import TradingEnvironment
from botalpaca.domain.errors import ValidationError

from .test_protection import _setup, _stop_order, _tp_order, make_position

PAPER = TradingEnvironment.PAPER


async def test_a_live_take_profit_is_released_so_a_stop_can_exist(database):
    """Alpaca reserves shares per open exit order, so a target blocks a stop.

    Measured against the PAPER account: with a limit resting on a 4-share long,
    submitting a stop for those same 4 shares is rejected with
    ``insufficient qty available ... held_for_orders: 4``. The stop is the exit
    that has to survive, so the target is the one that gets released.
    """
    manager, _, client = await _setup(
        database, orders=[_tp_order("t1", limit=106.0)]
    )
    position = make_position(entry=100.0, current=101.0)

    state = await manager.ensure_stop(
        environment=PAPER, position=position, stop_price=97.0, reason="test"
    )

    assert "t1" in client.cancelled, "the target must be released before the stop"
    assert state.has_stop is True
    assert state.has_take_profit is False
    assert any("take-profit liberado" in n.lower() for n in state.notes)


async def test_releasing_the_target_fails_loudly_when_the_cancel_is_refused(database):
    """If the shares cannot be released, the operator must be told, not ignored."""
    manager, engine, _ = await _setup(database, orders=[_tp_order("t1")])
    position = make_position(entry=100.0, current=101.0)

    async def refuse(order_id, environment=None):
        raise RuntimeError("Alpaca rejected the request")

    engine.cancel_order = refuse

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
