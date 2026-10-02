"""The broker rules that decide whether protection can see an order.

Every rule here was confirmed against the live PAPER API rather than inferred.
The ``nested`` one cost a full debugging round on the running machine, so it is
pinned by a test that fails if anyone turns nesting back on.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus

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
def test_live_orders_are_queried_without_nesting():
    """Nesting folds children into the parent and drops the non-open ones.

    Measured on the PAPER account: the identical query returns 38 rows including
    the HELD order unnested, and 18 rows with zero HELD when nested. This asserts
    the arguments the engine really sends, not the text of its docstring.
    """
    seen: dict[str, object] = {}

    class _RecordingClient:
        environment_name = PAPER
        environment = PAPER

        async def get_orders(self, **kwargs):
            seen.update(kwargs)
            return [
                _raw("held", "h1", type=OrderType.STOP, stop_price=97.0),
                _raw("new", "t1", type=OrderType.LIMIT, limit_price=106.0),
                _raw("filled", "old"),
            ]

    engine = ExecutionEngine(_RecordingClient(), None, active_environment=PAPER)
    live = _run(engine.get_live_orders(symbols=["AAPL"]))

    assert seen.get("nested") is None, "nesting hides the HELD order protection needs"
    assert seen["status"] == QueryOrderStatus.ALL
    # The HELD order survives the terminal filter; the filled one does not.
    assert [o.id for o in live] == ["h1", "t1"]


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
