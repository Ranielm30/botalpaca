from __future__ import annotations

import asyncio

from botalpaca.domain.enums import TradingEnvironment
from botalpaca.execution.mapping import to_order_state

from .test_protection import _setup, _tp_order, make_position

PAPER = TradingEnvironment.PAPER


def _to_state(raw):
    return to_order_state(raw, PAPER)



async def test_the_create_waits_for_a_cancel_to_actually_release_the_shares(database, monkeypatch):
    """Alpaca keeps counting a cancelled order as held for a short while.

    Measured live: cancelling the stop and creating its replacement in the same
    breath is rejected with ``held_for_orders: 4`` naming the order that was
    just cancelled, and the replacement never lands. The create has to wait for
    the old order to go terminal.
    """
    manager, engine, _ = await _setup(database, orders=[_tp_order("t1")])
    position = make_position(entry=100.0, current=101.0)
    waits: list[int] = []

    calls = {"n": 0}
    real_live = engine.get_live_orders_for_symbol

    async def still_held_a_while(symbol):
        calls["n"] += 1
        if calls["n"] <= 2:
            return [_to_state(_tp_order("t1"))]
        return []

    monkeypatch.setattr(engine, "get_live_orders_for_symbol", still_held_a_while)
    monkeypatch.setattr(asyncio, "sleep", lambda _d: _noop())

    state = await manager.ensure_stop(
        environment=PAPER, position=position, stop_price=97.0, reason="test"
    )
    assert state.has_stop is True
    assert calls["n"] == 3, "it must keep asking until the shares are free"


async def _noop():
    return None


async def test_the_wait_gives_up_instead_of_hanging_forever(database, monkeypatch):
    """A position whose shares never free must fail fast, not block the monitor."""
    manager, engine, _ = await _setup(database, orders=[_tp_order("t1")])
    position = make_position(entry=100.0, current=101.0)

    async def always_held(symbol):
        return [_to_state(_tp_order("t1"))]

    monkeypatch.setattr(engine, "get_live_orders_for_symbol", always_held)
    monkeypatch.setattr(asyncio, "sleep", lambda _d: _noop())

    freed = await manager._await_shares_free(PAPER, "AAPL", attempts=3)
    assert freed is False
