"""Tests for the broker rules that broke protection against real Alpaca.

Every rule here was confirmed live against the PAPER API, not inferred:

* a bracket child can be left in ``HELD``, which reserves the shares without
  protecting anything and never appears in an OPEN query;
* Alpaca reserves shares per open exit order, so a replacement exit submitted
  before the old one is cancelled is always rejected;
* ``replace_order`` is refused while the order is accepted or pending_new, which
  is exactly the state a child is in right after its entry fills.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from botalpaca.domain import TradingEnvironment
from botalpaca.execution.client import order_status_value
from botalpaca.execution.engine import _TERMINAL_STATUSES
from botalpaca.execution.mapping import to_order_state
from botalpaca.protection.manager import _is_inert

PAPER = TradingEnvironment.PAPER


def _raw(status: str, order_id: str = "o1", **kw) -> object:
    base = dict(
        id=order_id,
        client_order_id=None,
        symbol="AAPL",
        qty=5.0,
        side="sell",
        type="stop",
        order_class="simple",
        status=status,
        filled_qty=None,
        filled_avg_price=None,
        limit_price=None,
        stop_price=97.0,
        trail_percent=None,
        legs=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


# -- status normalisation -------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("OrderStatus.NEW", "new"),
        ("OrderStatus.FILLED", "filled"),
        ("new", "new"),
        ("HELD", "held"),
        ("pending_new", "pending_new"),
    ],
)
def test_status_value_unwraps_alpaca_enums(raw: str, expected: str):
    assert order_status_value(_raw(raw)) == expected


@pytest.mark.parametrize("status", sorted(_TERMINAL_STATUSES))
def test_terminal_states_are_final(status: str):
    assert status in _TERMINAL_STATUSES


@pytest.mark.parametrize(
    "status", ["new", "accepted", "held", "pending_new", "partially_filled", "pending_cancel"]
)
def test_live_states_are_not_terminal(status: str):
    """Anything not final may still fill, so it must stay visible."""
    assert status not in _TERMINAL_STATUSES


# -- inert orders ---------------------------------------------------------------------
def test_held_order_is_inert():
    """HELD reserves the shares but can never fire."""
    assert _is_inert(to_order_state(_raw("held"), PAPER)) is True


def test_pending_cancel_order_is_inert():
    assert _is_inert(to_order_state(_raw("pending_cancel"), PAPER)) is True


def test_new_order_is_not_inert():
    assert _is_inert(to_order_state(_raw("new"), PAPER)) is False


def test_a_held_stop_is_not_counted_as_protection():
    """The bug: a HELD stop looked like protection while protecting nothing."""
    from botalpaca.protection.manager import PositionProtectionManager

    manager = PositionProtectionManager.__new__(PositionProtectionManager)
    state = manager._state_from_orders(PAPER, "AAPL", [to_order_state(_raw("held", "h1"), PAPER)], None)
    assert state.has_stop is False
    assert state.inert_order_ids == ["h1"]


# -- the call signatures the broker actually has --------------------------------------
def test_alpaca_cancel_orders_takes_no_symbol():
    """Guards the bug that made /cancelar raise TypeError.

    ``TradingClient.cancel_orders`` accepts no arguments, so our wrapper must
    never forward a symbol to it.
    """
    import inspect

    from alpaca.trading.client import TradingClient

    params = inspect.signature(TradingClient.cancel_orders).parameters
    assert "symbol" not in params


def test_alpaca_query_order_status_has_no_held():
    """HELD is unreachable by status, which is why live queries use ALL."""
    from alpaca.trading.enums import QueryOrderStatus

    members = {m.name.lower() for m in QueryOrderStatus}
    assert "held" not in members
    assert "all" in members


# -- resolution ------------------------------------------------------------------------
async def test_live_orders_exclude_terminal_states_and_keep_held(database):
    from botalpaca.execution.engine import ExecutionEngine
    from tests.conftest import FakeTradingClient

    client = FakeTradingClient(PAPER)
    for raw in (
        _raw("new", "live-1"),
        _raw("filled", "dead-1"),
        _raw("canceled", "dead-2"),
        _raw("held", "held-1"),
    ):
        client.orders[raw.id] = raw

    engine = ExecutionEngine(client, database, active_environment=PAPER)
    ids = {o.id for o in await engine.get_live_orders_for_symbol("AAPL")}
    assert "live-1" in ids
    assert "held-1" in ids, "a HELD child still reserves the shares and must be visible"
    assert "dead-1" not in ids
    assert "dead-2" not in ids


async def test_ensure_stop_clears_the_inert_order_before_creating(database):
    """A HELD child holds the shares; creating a stop without clearing it fails."""
    from tests.test_protection import PAPER as P
    from tests.test_protection import _setup, _stop_order, make_position

    manager, engine, client = await _setup(database, [_stop_order("held-1", stop=97.0)])
    client.orders["held-1"].status = "held"

    state = await manager.ensure_stop(
        environment=P, position=make_position(entry=100.0, current=102.0), stop_price=95.0
    )
    assert "held-1" in client.cancelled
    assert state.has_stop is True


async def test_break_even_recreates_when_replace_is_refused(database):
    """Right after an entry fills the stop is pending_new and cannot be replaced."""
    from tests.test_protection import PAPER as P
    from tests.test_protection import _setup, _stop_order, make_position

    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])

    async def refuse(order_id, request):
        raise RuntimeError("order cannot be replaced when status is pending_new")

    client.replace_order_by_id = refuse

    state = await manager.move_to_break_even(
        environment=P, position=make_position(entry=100.0, current=104.0)
    )
    assert state.break_even_active is True
    assert state.has_stop is True
    assert "s1" in client.cancelled, "the unreplaceable stop must be cancelled"
    assert any("recreado" in n for n in state.notes)
