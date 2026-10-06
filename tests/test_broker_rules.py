"""The broker rules that decide whether protection can see an order.

Every rule here was confirmed against the live PAPER API rather than inferred.
The ``nested`` one cost a full debugging round on the running machine, so it is
pinned by a test that fails if anyone turns nesting back on.
"""

from __future__ import annotations

import asyncio
import inspect
import types
from types import SimpleNamespace

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, QueryOrderStatus

from botalpaca.domain.enums import OrderSide, OrderType, TradingEnvironment
from botalpaca.execution.client import order_status_value
from botalpaca.execution.engine import ExecutionEngine
from botalpaca.execution.mapping import to_order_state

PAPER = TradingEnvironment.PAPER


def _run(coro):
    return asyncio.run(coro)


def _raw(status, order_id, **kw):
    return SimpleNamespace(
        id=order_id,
        client_order_id=None,
        symbol=kw.pop("symbol", "AAPL"),
        qty=kw.pop("qty", 5.0),
        side=kw.pop("side", OrderSide.SELL),
        type=kw.pop("type", OrderType.MARKET),
        order_class=kw.pop("order_class", "bracket"),
        status=status,
        filled_qty=kw.pop("filled_qty", None),
        filled_avg_price=kw.pop("filled_avg_price", None),
        limit_price=kw.pop("limit_price", None),
        stop_price=kw.pop("stop_price", None),
        trail_percent=None,
        trail_price=None,
        legs=[],
        **kw,
    )


# --- what Alpaca can even ask for ------------------------------------------------
def test_query_order_status_has_no_held():
    """A HELD order can only be found through ALL, never through OPEN."""
    members = [str(m).split(".")[-1].lower() for m in QueryOrderStatus]
    assert "held" not in members
    assert "all" in members


def test_alpaca_cancel_orders_takes_no_symbol():
    """Forwarding a symbol raised TypeError and made /cancelar dead."""
    assert "symbol" not in inspect.signature(TradingClient.cancel_orders).parameters


def test_status_value_unwraps_alpaca_enums():
    from alpaca.trading.enums import OrderStatus

    assert order_status_value(_raw(OrderStatus.NEW, "a")) == "new"
    assert order_status_value(_raw(OrderStatus.FILLED, "a")) == "filled"
    assert order_status_value(_raw(OrderStatus.HELD, "a")) == "held"
    assert order_status_value(_raw("pending_new", "a")) == "pending_new"


# --- nesting is what hides HELD --------------------------------------------------
def test_live_orders_query_both_ways_and_union_them():
    """Nesting hides the HELD child; not nesting hides the group link.

    Measured on the PAPER account: the identical query returns 27 rows
    unnested -- the OCO stop child as its own sibling row -- and 20 rows
    nested, with children folded into the parent. Neither alone answers "is this
    HELD order a leg of a live group?", which is exactly what the protection
    manager needs to know before sweeping an order as debris.
    """
    seen: list[dict[str, object]] = []

    child = _raw("held", "h1", type=OrderType.STOP, stop_price=97.0)
    child.order_class = OrderClass.OCO
    parent = _raw("new", "t1", type=OrderType.LIMIT, limit_price=106.0)
    parent.order_class = OrderClass.OCO
    orphan = _raw("held", "o1", type=OrderType.STOP, stop_price=95.0)
    orphan.order_class = OrderClass.SIMPLE
    parent_nested = _raw("new", "t1", type=OrderType.LIMIT, limit_price=106.0)
    parent_nested.order_class = OrderClass.OCO
    parent_nested.legs = [child]

    class _RecordingClient:
        environment_name = PAPER
        environment = PAPER

        async def get_orders(self, **kwargs):
            seen.append(kwargs)
            if kwargs.get("nested"):
                return [parent_nested, orphan]
            return [child, parent, orphan, _raw("filled", "old")]

    engine = ExecutionEngine(_RecordingClient(), None, active_environment=PAPER)
    live = _run(engine.get_live_orders(symbols=["AAPL"]))

    assert len(seen) == 2, "both queries are needed"
    assert [call.get("nested") for call in seen] == [False, True]
    assert all(call["status"] == QueryOrderStatus.ALL for call in seen)

    by_id = {str(o.id): o for o in live}
    assert "old" not in by_id, "a filled order is not live"
    # The parent carries the group link even though the child is a separate row.
    assert len(by_id["t1"].legs) == 1
    assert str(by_id["t1"].legs[0].id) == "h1"
    # The HELD child survives, and the genuinely orphaned one is still visible.
    assert {"h1", "o1"} <= set(by_id)


async def test_a_merged_order_keeps_the_row_that_has_the_group():
    """The unnested row arrives first; the nested one must still win.

    This is the exact production failure: the take-profit parent came back from
    the unnested query with ``legs=0``, so the merge has to prefer the nested
    row that carries the child id, otherwise the stop it protects is invisible.
    """
    child = _raw("held", "h1", type=OrderType.STOP, stop_price=97.0)
    # ``_raw`` pins ``legs`` to an empty tuple, so build the two parent shapes by
    # hand: that is the whole difference the merge has to notice.
    parent_kwargs = dict(
        type=OrderType.LIMIT,
        limit_price=110.0,
        order_class=OrderClass.OCO,
        status="new",
        side=OrderSide.SELL,
        qty=10.0,
        filled_qty=None,
        filled_avg_price=None,
        stop_price=None,
        trail_percent=None,
        trail_price=None,
        time_in_force=None,
        client_order_id=None,
        created_at=None,
        updated_at=None,
    )
    bare_parent = types.SimpleNamespace(id="t1", **parent_kwargs, legs=())
    nested_parent = types.SimpleNamespace(id="t1", **parent_kwargs, legs=(child,))

    class _Client:
        environment_name = PAPER
        environment = PAPER

        async def get_orders(self, status=None, limit=500, nested=False, symbols=None):
            return [nested_parent] if nested else [bare_parent]

    engine = ExecutionEngine(_Client(), None, active_environment=PAPER, require_confirmation=False)
    merged = await engine.get_live_orders()
    by_id = {str(o.id): o for o in merged}

    assert len(by_id["t1"].legs) == 1
    assert str(by_id["t1"].legs[0].id) == "h1"


def test_get_live_orders_takes_no_nested_argument():
    """The footgun is removed, not merely unused."""
    params = inspect.signature(ExecutionEngine.get_live_orders).parameters
    assert "nested" not in params


# --- which orders count as still able to act -------------------------------------
def test_terminal_states_are_final():
    from botalpaca.execution.engine import _TERMINAL_STATUSES

    assert _TERMINAL_STATUSES == frozenset(
        {"filled", "canceled", "expired", "replaced", "rejected"}
    )


def test_live_states_are_not_terminal():
    from botalpaca.execution.engine import _TERMINAL_STATUSES

    for state in ("new", "accepted", "held", "pending_new", "partially_filled"):
        assert state not in _TERMINAL_STATUSES


def test_a_held_stop_is_not_counted_as_protection():
    """It reserves the shares and cannot execute, so it is not a stop."""
    from botalpaca.protection.manager import PositionProtectionManager

    manager = PositionProtectionManager.__new__(PositionProtectionManager)
    held = to_order_state(
        _raw("held", "h1", type=OrderType.STOP, stop_price=97.0), PAPER
    )
    state = manager._state_from_orders(PAPER, "AAPL", [held], OrderSide.SELL)
    assert state.has_stop is False
    assert state.inert_order_ids == ["h1"]


def test_a_new_stop_is_counted_as_protection():
    from botalpaca.protection.manager import PositionProtectionManager

    manager = PositionProtectionManager.__new__(PositionProtectionManager)
    live = to_order_state(
        _raw("new", "s1", type=OrderType.STOP, stop_price=97.0), PAPER
    )
    state = manager._state_from_orders(PAPER, "AAPL", [live], OrderSide.SELL)
    assert state.has_stop is True
    assert state.stop_price == 97.0
    assert state.inert_order_ids == []


# --- inert states ----------------------------------------------------------------
def test_held_and_pending_cancel_are_inert():
    from botalpaca.protection.manager import _is_inert

    assert _is_inert(to_order_state(_raw("held", "a"), PAPER)) is True
    assert _is_inert(to_order_state(_raw("pending_cancel", "a"), PAPER)) is True
    assert _is_inert(to_order_state(_raw("pending_replace", "a"), PAPER)) is True


def test_new_order_is_not_inert():
    from botalpaca.protection.manager import _is_inert

    assert _is_inert(to_order_state(_raw("new", "a"), PAPER)) is False
