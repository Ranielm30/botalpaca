"""Translate a :class:`TradePlan` into Alpaca request objects.

This module is the single place where domain intent becomes a broker request.
It encodes the *verified* Alpaca constraints and refuses anything it cannot
represent faithfully, raising
:class:`~botalpaca.domain.UnsupportedOrderShapeError` rather than silently
dropping a protection:

* ``qty`` and ``notional`` are mutually exclusive (and at least one is required).
* ``notional`` is only accepted for stock market orders.
* Fractional quantities are only accepted for stock market orders.
* Take-profit legs are limit orders (``TakeProfitRequest`` has no stop field).
* Stop-loss legs may be stop or stop-limit orders.
* A trailing stop cannot be a bracket/OCO child leg, so a plan that combines
  them is rejected here and handled by the Position Protection Manager as an
  explicit transition instead.
* Brackets are supported for market, limit and stop entries only; OCO is
  exit-only and OTO needs a primary order, so neither is built implicitly.
"""

from __future__ import annotations

from typing import Any

from alpaca.trading.enums import OrderClass as AlpacaOrderClass
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import OrderType as AlpacaOrderType
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)

from botalpaca.domain import (
    OrderClass,
    OrderType,
    SignalDirection,
    TimeInForce,
    TradePlan,
    TradingEnvironment,
    UnsupportedOrderShapeError,
    ValidationError,
)

__all__ = [
    "OrderBuilder",
    "alpaca_order_class",
    "alpaca_side",
    "alpaca_tif",
    "alpaca_type",
]

# Alpaca TIF values that Alpaca actually accepts.
_SUPPORTED_TIF = {t.value for t in AlpacaTIF}

# Entry order types Alpaca permits to carry bracket legs.
_BRACKET_CAPABLE_TYPES = {
    OrderType.MARKET,
    OrderType.LIMIT,
    OrderType.STOP,
    OrderType.STOP_LIMIT,
}


def alpaca_side(side: str | Any) -> AlpacaOrderSide:
    text = str(getattr(side, "value", side)).lower()
    try:
        return AlpacaOrderSide(text)
    except ValueError as exc:
        raise ValidationError(f"unsupported order side: {side!r}") from exc


def alpaca_type(order_type: OrderType | str) -> AlpacaOrderType:
    text = str(getattr(order_type, "value", order_type)).lower()
    try:
        return AlpacaOrderType(text)
    except ValueError as exc:
        raise ValidationError(f"unsupported order type: {order_type!r}") from exc


def alpaca_tif(tif: TimeInForce | str) -> AlpacaTIF:
    text = str(getattr(tif, "value", tif)).lower()
    if text not in _SUPPORTED_TIF:
        raise ValidationError(f"time in force not supported by Alpaca: {tif!r}")
    return AlpacaTIF(text)


def alpaca_order_class(order_class: OrderClass | str) -> AlpacaOrderClass:
    text = str(getattr(order_class, "value", order_class)).lower()
    try:
        return AlpacaOrderClass(text)
    except ValueError as exc:
        raise ValidationError(f"unsupported order class: {order_class!r}") from exc


def _is_fractional(value: float) -> bool:
    return abs(value - round(value)) > 1e-9


class OrderBuilder:
    """Builds broker requests and the values needed to audit them."""

    # -- entry orders --------------------------------------------------------

    @staticmethod
    def build_entry(
        plan: TradePlan,
        *,
        client_order_id: str | None = None,
    ) -> Any:
        """Build the entry order request for ``plan``.

        Raises :class:`UnsupportedOrderShapeError` when the requested shape
        cannot be expressed in a single Alpaca order.
        """
        _validate_base(plan)

        qty: float | None = None
        notional: float | None = None
        if plan.notional is not None:
            if plan.order_type is not OrderType.MARKET:
                raise UnsupportedOrderShapeError(
                    "Alpaca only accepts notional sizing on stock market orders."
                )
            notional = float(plan.notional)
        else:
            qty = float(plan.qty)
            if qty <= 0:
                raise ValidationError("qty must be positive")
            if _is_fractional(qty) and plan.order_type is not OrderType.MARKET:
                raise UnsupportedOrderShapeError(
                    "Alpaca only accepts fractional quantities on stock market orders."
                )

        side = alpaca_side(plan.direction.entry_side)
        tif = alpaca_tif(plan.time_in_force)
        # For a stop / stop-limit *entry*, ``stop_loss`` is the trigger price of
        # the entry itself, not a protective leg, so it must not be turned into
        # a bracket child order.
        is_stop_entry = plan.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
        has_protection = plan.take_profit is not None or (
            plan.stop_loss is not None and not is_stop_entry
        )

        if has_protection and plan.order_type not in _BRACKET_CAPABLE_TYPES:
            raise UnsupportedOrderShapeError(
                f"Alpaca cannot attach bracket legs to a {plan.order_type.value} order."
            )

        order_class = (
            AlpacaOrderClass.BRACKET
            if (has_protection and plan.order_class in (OrderClass.BRACKET, OrderClass.SIMPLE))
            else alpaca_order_class(plan.order_class)
        )
        wants_trailing = plan.trail_percent is not None or plan.trail_price is not None
        if wants_trailing and (
            has_protection or plan.order_class in (OrderClass.BRACKET, OrderClass.OCO)
        ):
            # A trailing stop is never a bracket/OCO child leg in Alpaca, but it
            # *is* valid as a standalone entry order.
            raise UnsupportedOrderShapeError(
                "A trailing stop cannot be submitted as a bracket leg. "
                "Submit the entry first, then use the Position Protection Manager "
                "to transition to a standalone trailing stop."
            )

        common: dict[str, Any] = {
            "symbol": plan.symbol.upper(),
            "side": side,
            "time_in_force": tif,
            "order_class": order_class,
        }
        if client_order_id:
            common["client_order_id"] = client_order_id
        if qty is not None:
            common["qty"] = qty
        if notional is not None:
            common["notional"] = notional
        if has_protection:
            common["take_profit"] = OrderBuilder.build_take_profit(plan.take_profit)
            common["stop_loss"] = OrderBuilder.build_stop_loss(plan.stop_loss)

        if plan.order_type is OrderType.MARKET:
            return MarketOrderRequest(**common)
        if plan.order_type is OrderType.LIMIT:
            return LimitOrderRequest(limit_price=_require(plan.limit_price, "limit_price"), **common)
        if plan.order_type is OrderType.STOP:
            # A stop *entry* reuses the ``stop_loss`` field as its trigger
            # price; TradePlan has no separate stop_price field.
            return StopOrderRequest(stop_price=_require(plan.stop_loss, "stop_loss"), **common)
        if plan.order_type is OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                stop_price=_require(plan.stop_loss, "stop_loss"),
                limit_price=_require(plan.limit_price, "limit_price"),
                **common,
            )
        if plan.order_type is OrderType.TRAILING_STOP:
            # Alpaca accepts exactly one of trail_percent / trail_price.
            if (plan.trail_percent is None) == (plan.trail_price is None):
                raise UnsupportedOrderShapeError(
                    "a trailing stop entry needs exactly one of trail_percent or trail_price"
                )
            return TrailingStopOrderRequest(
                trail_percent=plan.trail_percent,
                trail_price=plan.trail_price,
                **common,
            )
        raise UnsupportedOrderShapeError(
            f"order type {plan.order_type} cannot open a position"
        )

    # -- protective orders ---------------------------------------------------

    @staticmethod
    def build_take_profit(price: float | None) -> TakeProfitRequest | None:
        """Take profit legs are limit-only in Alpaca."""
        if price is None:
            return None
        return TakeProfitRequest(limit_price=float(price))

    @staticmethod
    def build_stop_loss(
        price: float | None, *, limit_price: float | None = None
    ) -> StopLossRequest | None:
        """Stop loss may be a stop or a stop-limit (``limit_price`` optional)."""
        if price is None:
            return None
        if limit_price is not None:
            return StopLossRequest(stop_price=float(price), limit_price=float(limit_price))
        return StopLossRequest(stop_price=float(price))

    @staticmethod
    def build_protective_stop(
        *,
        symbol: str,
        qty: float,
        exit_side: SignalDirection,
        stop_price: float,
        limit_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: str | None = None,
    ) -> StopOrderRequest | StopLimitOrderRequest:
        """A standalone stop order to protect an already-open position."""
        _require_positive(stop_price, "stop_price")
        if qty <= 0:
            raise ValidationError("protective qty must be positive")
        common: dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": abs(float(qty)),
            "side": alpaca_side(exit_side.exit_side),
            "time_in_force": alpaca_tif(time_in_force),
            "order_class": AlpacaOrderClass.SIMPLE,
        }
        if client_order_id:
            common["client_order_id"] = client_order_id
        if limit_price is not None:
            _require_positive(limit_price, "limit_price")
            return StopLimitOrderRequest(
                stop_price=float(stop_price), limit_price=float(limit_price), **common
            )
        return StopOrderRequest(stop_price=float(stop_price), **common)

    @staticmethod
    def build_protective_limit(
        *,
        symbol: str,
        qty: float,
        exit_side: SignalDirection,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: str | None = None,
    ) -> LimitOrderRequest:
        _require_positive(limit_price, "limit_price")
        if qty <= 0:
            raise ValidationError("protective qty must be positive")
        common: dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": abs(float(qty)),
            "side": alpaca_side(exit_side.exit_side),
            "time_in_force": alpaca_tif(time_in_force),
            "order_class": AlpacaOrderClass.SIMPLE,
        }
        if client_order_id:
            common["client_order_id"] = client_order_id
        return LimitOrderRequest(limit_price=float(limit_price), **common)

    @staticmethod
    def build_oco_exits(
        *,
        symbol: str,
        qty: float,
        exit_side: SignalDirection,
        stop_price: float,
        take_profit_price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: str | None = None,
    ) -> LimitOrderRequest:
        """Both exits for an open position, as the single group Alpaca allows.

        Alpaca reserves the shares behind every open exit order, so two
        independent orders for one position is not merely discouraged, it is
        rejected with "insufficient qty available". The documented way to hold a
        stop and a take-profit together on a position that is already open is an
        OCO order: the parent is the take-profit limit and the stop is its
        child. The parent type is always "limit" for an OCO, whichever side the
        position is on.

        Cancelling either leg cancels the group, so the manager must never treat
        the held child as debris.
        """
        _require_positive(stop_price, "stop_price")
        _require_positive(take_profit_price, "take_profit_price")
        if qty <= 0:
            raise ValidationError("oco qty must be positive")
        # The stop must sit on the losing side of the target, which is what
        # makes one of them a profit exit and the other a loss exit.
        #
        # Decided from the exit side, never from enum identity: the manager passes
        # a PositionDirection while the strategies pass a SignalDirection, and
        # two enums whose members share a name are still different objects, so
        # `exit_side is SignalDirection.LONG` was silently always False and every
        # long exit was validated with the short rule.
        closing_long = alpaca_side(exit_side.exit_side) == AlpacaOrderSide.SELL
        if closing_long and stop_price >= take_profit_price:
            raise ValidationError(
                "un OCO de salida necesita stop por debajo del objetivo"
            )
        if not closing_long and take_profit_price >= stop_price:
            raise ValidationError(
                "un OCO de salida necesita stop por encima del objetivo"
            )
        common: dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": abs(float(qty)),
            "side": alpaca_side(exit_side.exit_side),
            "time_in_force": alpaca_tif(time_in_force),
            "order_class": AlpacaOrderClass.OCO,
        }
        if client_order_id:
            common["client_order_id"] = client_order_id
        return LimitOrderRequest(
            limit_price=float(take_profit_price),
            take_profit=TakeProfitRequest(limit_price=float(take_profit_price)),
            stop_loss=StopLossRequest(stop_price=float(stop_price)),
            **common,
        )

    @staticmethod
    def build_trailing_stop(
        *,
        symbol: str,
        qty: float,
        exit_side: SignalDirection,
        trail_percent: float | None = None,
        trail_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: str | None = None,
    ) -> TrailingStopOrderRequest:
        """A standalone trailing stop.

        Alpaca supports ``trail_percent`` or ``trail_price`` on a
        ``trailing_stop`` order, never both.
        """
        if qty <= 0:
            raise ValidationError("trailing qty must be positive")
        has_pct = trail_percent is not None
        has_price = trail_price is not None
        if has_pct == has_price:
            raise UnsupportedOrderShapeError(
                "A trailing stop requires exactly one of trail_percent or trail_price."
            )
        common: dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": abs(float(qty)),
            "side": alpaca_side(exit_side.exit_side),
            "time_in_force": alpaca_tif(time_in_force),
            "order_class": AlpacaOrderClass.SIMPLE,
        }
        if client_order_id:
            common["client_order_id"] = client_order_id
        if has_pct:
            if float(trail_percent) <= 0:
                raise ValidationError("trail_percent must be positive")
            return TrailingStopOrderRequest(trail_percent=float(trail_percent), **common)
        _require_positive(trail_price, "trail_price")  # type: ignore[arg-type]
        return TrailingStopOrderRequest(trail_price=float(trail_price), **common)  # type: ignore[arg-type]

    # -- replacements --------------------------------------------------------

    @staticmethod
    def build_replace(
        *,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail: float | None = None,
        time_in_force: TimeInForce | None = None,
        client_order_id: str | None = None,
    ) -> Any:
        """Build a ``ReplaceOrderRequest`` with only the fields being changed."""
        from alpaca.trading.requests import ReplaceOrderRequest

        if qty is None and limit_price is None and stop_price is None and trail is None:
            raise ValidationError("a replacement must change at least one field")
        if trail is not None and trail <= 0:
            raise ValidationError("trail must be positive")
        return ReplaceOrderRequest(
            qty=int(qty) if qty is not None else None,
            limit_price=limit_price,
            stop_price=stop_price,
            trail=trail,
            time_in_force=alpaca_tif(time_in_force) if time_in_force else None,
            client_order_id=client_order_id,
        )

    # -- queries -------------------------------------------------------------

    @staticmethod
    def build_orders_query(
        *,
        status: str = "open",
        limit: int = 100,
        nested: bool = False,
        symbols: list[str] | None = None,
    ) -> GetOrdersRequest:
        return GetOrdersRequest(
            status=QueryOrderStatus(status), limit=limit, nested=nested, symbols=symbols
        )


def _validate_base(plan: TradePlan) -> None:
    if not plan.symbol or not plan.symbol.strip():
        raise ValidationError("symbol is required")
    if (plan.qty is None or plan.qty <= 0) and not plan.notional:
        raise ValidationError("an order needs either a positive qty or a notional amount")
    if plan.qty and plan.qty > 0 and plan.notional:
        raise UnsupportedOrderShapeError(
            "Alpaca rejects orders that set both qty and notional."
        )
    if plan.environment is not TradingEnvironment.PAPER and plan.environment is not TradingEnvironment.REAL:
        raise ValidationError(f"unknown environment: {plan.environment!r}")


def _require(value: float | None, field: str) -> float:
    if value is None:
        raise ValidationError(f"{field} is required for this order type")
    _require_positive(value, field)
    return float(value)


def _require_positive(value: float, field: str) -> None:
    if value is None or value <= 0:
        raise ValidationError(f"{field} must be positive")
