"""Mapping between Alpaca response objects and domain models.

Kept in its own module so the execution engine never leaks Alpaca types and
the tests can build domain objects without an Alpaca dependency.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from botalpaca.domain import (
    AccountSnapshot,
    FillRecord,
    OrderLeg,
    OrderState,
    PositionSnapshot,
    TradingEnvironment,
)

__all__ = [
    "to_account",
    "to_fill",
    "to_order_state",
    "to_position",
]


def _dt(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    return None


def _enum_value(value: Any, default: str) -> str:
    if value is None:
        return default
    raw = getattr(value, "value", value)
    return str(raw).lower() if isinstance(raw, str) else str(raw)


def to_account(raw: Any, environment: TradingEnvironment) -> AccountSnapshot:
    """Convert a ``TradeAccount`` into an :class:`AccountSnapshot`."""
    return AccountSnapshot(
        environment=environment,
        account_id=str(getattr(raw, "id", "") or ""),
        account_number=str(getattr(raw, "account_number", "") or ""),
        status=str(getattr(raw, "status", "unknown") or "unknown"),
        currency=str(getattr(raw, "currency", "USD") or "USD"),
        equity=float(getattr(raw, "equity", 0.0) or 0.0),
        last_equity=float(getattr(raw, "last_equity", 0.0) or 0.0),
        cash=float(getattr(raw, "cash", 0.0) or 0.0),
        buying_power=float(getattr(raw, "buying_power", 0.0) or 0.0),
        non_marginable_buying_power=float(
            getattr(raw, "non_marginable_buying_power", 0.0) or 0.0
        ),
        daytrading_buying_power=float(getattr(raw, "daytrading_buying_power", 0.0) or 0.0),
        regt_buying_power=float(getattr(raw, "regt_buying_power", 0.0) or 0.0),
        long_market_value=float(getattr(raw, "long_market_value", 0.0) or 0.0),
        short_market_value=float(getattr(raw, "short_market_value", 0.0) or 0.0),
        portfolio_value=float(getattr(raw, "portfolio_value", 0.0) or 0.0),
        daytrade_count=int(getattr(raw, "daytrade_count", 0) or 0),
        pattern_day_trader=bool(getattr(raw, "pattern_day_trader", False)),
        trading_blocked=bool(getattr(raw, "trading_blocked", False)),
        account_blocked=bool(getattr(raw, "account_blocked", False)),
        trade_suspended=bool(getattr(raw, "trade_suspended_by_user", False)),
        shorting_enabled=bool(getattr(raw, "shorting_enabled", False)),
        as_of=_dt(getattr(raw, "created_at", None)) or dt.datetime.now(dt.UTC),
    )


def to_position(raw: Any, environment: TradingEnvironment) -> PositionSnapshot:
    """Convert an Alpaca ``Position`` into a :class:`PositionSnapshot`."""
    qty = float(getattr(raw, "qty", 0.0) or 0.0)
    return PositionSnapshot(
        environment=environment,
        symbol=str(getattr(raw, "symbol", "")).upper(),
        qty=qty,
        side=str(getattr(raw, "side", "long") or "long").lower(),
        avg_entry_price=float(getattr(raw, "avg_entry_price", 0.0) or 0.0),
        current_price=float(getattr(raw, "current_price", 0.0) or 0.0),
        market_value=float(getattr(raw, "market_value", 0.0) or 0.0),
        cost_basis=float(getattr(raw, "cost_basis", 0.0) or 0.0),
        unrealized_pl=float(getattr(raw, "unrealized_pl", 0.0) or 0.0),
        unrealized_plpc=float(getattr(raw, "unrealized_plpc", 0.0) or 0.0),
        unrealized_intraday_pl=float(getattr(raw, "unrealized_intraday_pl", 0.0) or 0.0),
        unrealized_intraday_plpc=float(getattr(raw, "unrealized_intraday_plpc", 0.0) or 0.0),
        change_today=float(getattr(raw, "change_today", 0.0) or 0.0),
        qty_available=float(getattr(raw, "qty_available", qty) or 0.0),
        exchange=str(getattr(raw, "exchange", "") or ""),
        asset_class=_enum_value(getattr(raw, "asset_class", "us_equity"), "us_equity"),
    )


def to_order_state(raw: Any, environment: TradingEnvironment) -> OrderState:
    """Convert an Alpaca ``Order`` into an :class:`OrderState`.

    ``legs`` is populated from the order's child orders, which is how bracket
    and OCO children are obtained from Alpaca.
    """
    legs: list[OrderLeg] = []
    for leg in getattr(raw, "legs", None) or []:
        legs.append(
            OrderLeg(
                symbol=str(getattr(leg, "symbol", "")).upper(),
                qty=float(getattr(leg, "qty", 0.0) or 0.0),
                side=_enum_value(getattr(leg, "side", "buy"), "buy"),  # type: ignore[arg-type]
                order_type=_enum_value(getattr(leg, "type", "market"), "market"),  # type: ignore[arg-type]
            )
        )

    raw_dict = _as_dict(raw)
    return OrderState(
        id=str(getattr(raw, "id", "") or ""),
        client_order_id=getattr(raw, "client_order_id", None),
        environment=environment,
        symbol=str(getattr(raw, "symbol", "")).upper(),
        qty=_float_or_none(getattr(raw, "qty", None)),
        notional=_float_or_none(getattr(raw, "notional", None)),
        filled_qty=float(getattr(raw, "filled_qty", 0.0) or 0.0),
        filled_avg_price=_float_or_none(getattr(raw, "filled_avg_price", None)),
        side=_enum_value(getattr(raw, "side", "buy"), "buy"),  # type: ignore[arg-type]
        order_type=_enum_value(getattr(raw, "order_type", None) or getattr(raw, "type", "market"), "market"),  # type: ignore[arg-type]
        time_in_force=_enum_value(getattr(raw, "time_in_force", "day"), "day"),  # type: ignore[arg-type]
        order_class=_enum_value(getattr(raw, "order_class", "simple"), "simple"),  # type: ignore[arg-type]
        status=str(getattr(raw, "status", "unknown") or "unknown").lower(),
        limit_price=_float_or_none(getattr(raw, "limit_price", None)),
        stop_price=_float_or_none(getattr(raw, "stop_price", None)),
        trail_percent=_float_or_none(getattr(raw, "trail_percent", None)),
        trail_price=_float_or_none(getattr(raw, "trail_price", None)),
        extended_hours=bool(getattr(raw, "extended_hours", False)),
        created_at=_dt(getattr(raw, "created_at", None)),
        updated_at=_dt(getattr(raw, "updated_at", None)),
        legs=legs,
        raw=raw_dict,
    )


def to_fill(raw: Any, environment: TradingEnvironment) -> FillRecord:
    """Convert an account activity (``FILL``) into a :class:`FillRecord`."""
    price = getattr(raw, "price", None)
    qty = getattr(raw, "qty", None)
    if price is None and qty is None:
        # Partial-fill activity shape: (order_id, executed_qty, executed_avg_price)
        order_id, executed_qty, executed_price = raw
        return FillRecord(
            environment=environment,
            order_id=str(order_id),
            symbol="",
            side="buy",  # type: ignore[arg-type]
            qty=float(executed_qty or 0.0),
            price=float(executed_price or 0.0),
        )
    return FillRecord(
        environment=environment,
        order_id=str(getattr(raw, "order_id", "") or ""),
        symbol=str(getattr(raw, "symbol", "") or "").upper(),
        side=_enum_value(getattr(raw, "side", "buy"), "buy"),  # type: ignore[arg-type]
        qty=float(qty or 0.0),
        price=float(price or 0.0),
        timestamp=_dt(getattr(raw, "transaction_time", None) or getattr(raw, "timestamp", None)),
        fee=float(getattr(raw, "fee", 0.0) or 0.0),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dict(raw: Any) -> dict[str, Any]:
    for method in ("model_dump", "dict"):
        fn = getattr(raw, method, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:  # pragma: no cover - defensive
                continue
            if isinstance(data, dict):
                return data
    if isinstance(raw, dict):
        return dict(raw)
    return {}
