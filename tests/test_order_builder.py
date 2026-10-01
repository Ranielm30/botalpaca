"""Order builder: only shapes that Alpaca truly supports."""

from __future__ import annotations

import pytest

from botalpaca.domain.enums import (
    OrderClass,
    OrderSide,
    OrderType,
    SignalDirection,
    TimeInForce,
    TradingEnvironment,
)
from botalpaca.domain.errors import UnsupportedOrderShapeError, ValidationError
from botalpaca.domain.models import TradePlan
from botalpaca.execution.builder import OrderBuilder


def _plan(**over) -> TradePlan:
    base = dict(
        symbol="AAPL",
        environment=TradingEnvironment.PAPER,
        direction=SignalDirection.LONG,
        order_type=OrderType.MARKET,
        qty=10.0,
    )
    base.update(over)
    return TradePlan(**base)


# -- entry orders -------------------------------------------------------------------
def test_market_entry():
    req = OrderBuilder.build_entry(_plan(), client_order_id="cid-1")
    assert type(req).__name__ == "MarketOrderRequest"
    assert req.qty == 10.0
    assert req.time_in_force == TimeInForce.DAY


def test_limit_entry():
    req = OrderBuilder.build_entry(_plan(order_type=OrderType.LIMIT, limit_price=99.5))
    assert type(req).__name__ == "LimitOrderRequest"
    assert req.limit_price == 99.5


def test_stop_entry():
    req = OrderBuilder.build_entry(_plan(order_type=OrderType.STOP, stop_loss=101.0))
    assert type(req).__name__ == "StopOrderRequest"
    assert req.stop_price == 101.0


def test_stop_limit_entry():
    req = OrderBuilder.build_entry(
        _plan(order_type=OrderType.STOP_LIMIT, stop_loss=101.0, limit_price=100.5)
    )
    assert type(req).__name__ == "StopLimitOrderRequest"


def test_qty_and_notional_together_rejected():
    with pytest.raises(UnsupportedOrderShapeError):
        OrderBuilder.build_entry(_plan(notional=1000.0))


def test_notional_only_on_stock_market():
    req = OrderBuilder.build_entry(
        _plan(qty=0.0, notional=1000.0, order_type=OrderType.MARKET)
    )
    assert req.notional == 1000.0
    with pytest.raises(UnsupportedOrderShapeError):
        OrderBuilder.build_entry(
            _plan(qty=0.0, notional=1000.0, order_type=OrderType.LIMIT, limit_price=99.0)
        )


def test_fractional_qty_only_on_market():
    req = OrderBuilder.build_entry(_plan(qty=0.5))
    assert req.qty == 0.5
    with pytest.raises(UnsupportedOrderShapeError):
        OrderBuilder.build_entry(_plan(qty=0.5, order_type=OrderType.LIMIT, limit_price=99.0))


def test_missing_qty_and_notional_rejected():
    with pytest.raises(ValidationError):
        OrderBuilder.build_entry(_plan(qty=0.0))


def test_unknown_order_type_rejected():
    # TradePlan itself validates the enum, so an unsupported type can never
    # reach the builder through a plan.
    with pytest.raises(ValueError):  # pydantic ValidationError subclasses ValueError
        _plan(order_type="teleport")  # type: ignore[arg-type]


# -- protective legs ----------------------------------------------------------------
def test_take_profit_leg_is_limit_only():
    # Bracket legs are built as TakeProfitRequest: Alpaca gives a take-profit
    # leg no type or TIF of its own, only a limit price.
    req = OrderBuilder.build_take_profit(110.0)
    assert type(req).__name__ == "TakeProfitRequest"
    assert req.limit_price == 110.0
    assert OrderBuilder.build_take_profit(None) is None


def test_stop_loss_leg_is_stop_or_stop_limit():
    req = OrderBuilder.build_stop_loss(95.0)
    assert type(req).__name__ == "StopLossRequest"
    assert req.stop_price == 95.0
    assert req.limit_price is None

    req = OrderBuilder.build_stop_loss(95.0, limit_price=94.8)
    assert type(req).__name__ == "StopLossRequest"
    assert req.limit_price == 94.8
    assert OrderBuilder.build_stop_loss(None) is None


def test_standalone_protective_stop_is_a_stop_order():
    req = OrderBuilder.build_protective_stop(
        symbol="AAPL", qty=10, exit_side=SignalDirection.LONG, stop_price=95.0
    )
    assert type(req).__name__ == "StopOrderRequest"
    assert req.stop_price == 95.0


def test_standalone_protective_stop_can_be_stop_limit():
    req = OrderBuilder.build_protective_stop(
        symbol="AAPL", qty=10, exit_side=SignalDirection.LONG, stop_price=95.0, limit_price=94.8
    )
    assert type(req).__name__ == "StopLimitOrderRequest"


def test_bracket_promoted_automatically():
    req = OrderBuilder.build_entry(_plan(stop_loss=97.0, take_profit=106.0))
    assert req.order_class == OrderClass.BRACKET
    assert req.stop_loss is not None
    assert req.take_profit is not None


def test_bracket_respects_explicit_class():
    req = OrderBuilder.build_entry(
        _plan(order_class=OrderClass.BRACKET, stop_loss=97.0, take_profit=106.0)
    )
    assert req.order_class == OrderClass.BRACKET


def test_oco_order_class_allowed():
    req = OrderBuilder.build_entry(
        _plan(order_class=OrderClass.OCO, stop_loss=97.0, take_profit=106.0)
    )
    assert req.order_class == OrderClass.OCO


# -- trailing stop: never a bracket leg --------------------------------------------
def test_trailing_stop_as_bracket_leg_is_rejected():
    with pytest.raises(UnsupportedOrderShapeError):
        OrderBuilder.build_entry(
            _plan(order_type=OrderType.TRAILING_STOP, trail_percent=2.0, stop_loss=97.0)
        )


def test_trailing_stop_standalone_percent():
    req = OrderBuilder.build_trailing_stop(
        symbol="AAPL", qty=10, exit_side=SignalDirection.LONG, trail_percent=2.0
    )
    assert type(req).__name__ == "TrailingStopOrderRequest"
    assert req.trail_percent == 2.0
    assert req.trail_price is None


def test_trailing_stop_standalone_price():
    req = OrderBuilder.build_trailing_stop(
        symbol="AAPL", qty=10, exit_side=SignalDirection.LONG, trail_price=105.0
    )
    assert req.trail_price == 105.0


def test_trailing_stop_requires_exactly_one_mode():
    with pytest.raises(UnsupportedOrderShapeError):
        OrderBuilder.build_trailing_stop(symbol="AAPL", qty=10, exit_side=SignalDirection.LONG)
    with pytest.raises(UnsupportedOrderShapeError):
        OrderBuilder.build_trailing_stop(
            symbol="AAPL", qty=10, exit_side=SignalDirection.LONG, trail_percent=2.0, trail_price=105.0
        )


def test_standalone_trailing_entry_is_allowed():
    req = OrderBuilder.build_entry(
        _plan(
            order_type=OrderType.TRAILING_STOP,
            trail_percent=2.0,
            direction=SignalDirection.SHORT,
        )
    )
    assert type(req).__name__ == "TrailingStopOrderRequest"


# -- protective orders --------------------------------------------------------------
def test_build_protective_stop_exit_side():
    req = OrderBuilder.build_protective_stop(
        symbol="AAPL", qty=10, exit_side=SignalDirection.LONG, stop_price=95.0
    )
    assert req.side == OrderSide.SELL
    assert req.time_in_force == TimeInForce.GTC


def test_build_protective_limit():
    req = OrderBuilder.build_protective_limit(
        symbol="AAPL", qty=10, exit_side=SignalDirection.LONG, limit_price=110.0
    )
    assert type(req).__name__ == "LimitOrderRequest"
    assert req.side == OrderSide.SELL


# -- replace ------------------------------------------------------------------------
def test_build_replace_requires_a_change():
    with pytest.raises(ValidationError):
        OrderBuilder.build_replace()


def test_build_replace_with_stop():
    req = OrderBuilder.build_replace(stop_price=98.0)
    assert req.stop_price == 98.0
    assert req.qty is None


def test_build_replace_coerces_qty_to_int():
    req = OrderBuilder.build_replace(qty=10.0)
    assert req.qty == 10
    assert isinstance(req.qty, int)


def test_build_orders_query():
    query = OrderBuilder.build_orders_query(status="open", limit=50, nested=True)
    assert query.status.value == "open"
    assert query.limit == 50
    assert query.nested is True
