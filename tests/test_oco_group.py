"""A live OCO group is working protection, never debris.

Alpaca accepts an OCO order on an open position and parks its stop child in
``HELD`` so the shares are not reserved twice, while the take-profit sits ``NEW``
as the parent. Proven live on PAPER: both legs accepted, both working.

The protection manager used to classify *any* ``HELD`` order as inert debris and
cancel it. Cancelling one leg of an OCO cancels the whole group, so within about
twenty-five seconds the bot had destroyed the take-profit it had just been given
and replaced it with a plain stop that consumed the shares. The operator was
left with a stop, no target, and an advertised R:R that could never be reached.
"""

from __future__ import annotations

import pytest

from botalpaca.execution.mapping import to_order_state
from botalpaca.protection.manager import _group_child_ids, _is_inert
from tests.conftest import fake_order_raw
from tests.test_protection import PAPER, _setup, make_position

PARENT_ID = "oco-parent"
CHILD_ID = "oco-stop"


def _oco_group(*, parent_status: str = "new", child_status: str = "held"):
    """The shape Alpaca actually returns for a live OCO exit pair."""
    child = fake_order_raw(
        CHILD_ID,
        symbol="AAPL",
        order_type="stop",
        side="sell",
        qty=17.0,
        stop_price=1110.16,
        status=child_status,
    )
    parent = fake_order_raw(
        PARENT_ID,
        symbol="AAPL",
        order_type="limit",
        side="sell",
        qty=17.0,
        limit_price=1201.73,
        status=parent_status,
        order_class="oco",
        legs=[child],
    )
    return to_order_state(parent, PAPER), to_order_state(child, PAPER)


# -- the classification itself --------------------------------------------------------
def test_a_held_stop_is_inert_when_nothing_claims_it():
    _parent, child = _oco_group()
    assert _is_inert(child) is True, "an orphan that holds shares is inert"


def test_a_held_stop_inside_a_live_group_is_not_inert():
    parent, child = _oco_group()
    protected = _group_child_ids([parent, child])
    assert CHILD_ID in protected
    assert _is_inert(child, protected_ids=protected) is False


def test_the_parent_is_never_inert():
    parent, _child = _oco_group()
    assert _is_inert(parent) is False


def test_a_bracket_parent_also_claims_its_children():
    child = fake_order_raw(
        "br-child",
        symbol="AAPL",
        order_type="stop",
        side="sell",
        qty=10.0,
        stop_price=95.0,
        status="held",
    )
    parent = fake_order_raw(
        "br-parent",
        symbol="AAPL",
        order_type="market",
        side="buy",
        qty=10.0,
        status="new",
        order_class="bracket",
        legs=[child],
    )
    assert "br-child" in _group_child_ids(
        [to_order_state(parent, PAPER), to_order_state(child, PAPER)]
    )


def test_a_plain_stop_is_not_mistaken_for_a_group_child():
    plain = to_order_state(
        fake_order_raw(
            "plain", symbol="AAPL", order_type="stop", side="sell",
            qty=10.0, stop_price=95.0, status="held",
        ),
        PAPER,
    )
    assert _group_child_ids([plain]) == set()


# -- what the monitor believes about a real OCO position ------------------------------
async def test_a_live_oco_reads_as_stop_plus_take_profit(database):
    manager, engine, client = await _setup(database)
    protection = manager
    parent, child = _oco_group()
    client.orders[parent.id] = parent
    client.orders[child.id] = child

    position = make_position(entry=1156.0, current=1142.0, qty=17.0)
    state = await protection.state_for_position(PAPER, position)

    assert state.has_stop is True
    assert state.stop_price == pytest.approx(1110.16)
    assert state.has_take_profit is True
    assert state.take_profit_price == pytest.approx(1201.73)
    assert state.inert_order_ids == [], "a working group must never look like debris"


async def test_reconciling_an_oco_leaves_the_group_intact(database):
    """The regression itself: reconcile used to cancel the group away."""
    manager, engine, client = await _setup(database)
    protection = manager
    parent, child = _oco_group()
    client.orders[parent.id] = parent
    client.orders[child.id] = child

    position = make_position(entry=1156.0, current=1142.0, qty=17.0)
    notes = await protection.reconcile(environment=PAPER, positions=[position])

    assert client.cancelled == [], "the bot must not cancel a working group"
    assert not any("ancladas" in n or "SIN stop" in n for n in notes), notes

    state = await protection.state_for_position(PAPER, position)
    assert state.has_stop is True
    assert state.has_take_profit is True
