"""Regression tests for the bracket-child id that protection registration needs.

Alpaca returns each bracket/OCO child as a leg, and ``Application.submit_plan``
reads ``stop_leg.id`` to record which order carries the stop. When ``OrderLeg``
had no ``id`` field, that attribute access raised ``AttributeError`` and took
down the whole buy path -- the exact thing the operator taps in Telegram.
"""

from __future__ import annotations

import types

from botalpaca.domain.enums import OrderSide, OrderType, TradingEnvironment
from botalpaca.execution.mapping import to_order_state

PAPER = TradingEnvironment.PAPER


def _raw_bracket() -> object:
    stop_leg = types.SimpleNamespace(
        id="stop-777",
        symbol="AAPL",
        qty=5.0,
        side=OrderSide.SELL,
        type=OrderType.STOP,
    )
    tp_leg = types.SimpleNamespace(
        id="tp-888",
        symbol="AAPL",
        qty=5.0,
        side=OrderSide.SELL,
        type=OrderType.LIMIT,
    )
    return types.SimpleNamespace(
        id="parent-111",
        client_order_id="abc",
        symbol="AAPL",
        qty=5.0,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        order_class="bracket",
        status="accepted",
        filled_qty=None,
        filled_avg_price=None,
        legs=[stop_leg, tp_leg],
    )


def test_legs_carry_the_alpaca_child_id():
    state = to_order_state(_raw_bracket(), PAPER)
    by_type = {leg.order_type.value: leg for leg in state.legs}
    assert by_type[OrderType.STOP.value].id == "stop-777"
    assert by_type[OrderType.LIMIT.value].id == "tp-888"


def test_leg_id_is_optional_when_alpaca_omits_it():
    """A leg without an id must not explode the mapper."""
    raw = _raw_bracket()
    raw.legs[0].id = None
    state = to_order_state(raw, PAPER)
    assert state.legs[0].id is None


def test_submit_plan_reads_the_child_ids():
    """The buy path indexes legs by type and then dereferences .id."""
    state = to_order_state(_raw_bracket(), PAPER)
    legs = {leg.order_type.value: leg for leg in state.legs}
    stop_leg = legs.get(OrderType.STOP.value)
    tp_leg = legs.get(OrderType.LIMIT.value)
    # This is the line that used to raise AttributeError.
    assert stop_leg.id == "stop-777"
    assert tp_leg.id == "tp-888"
