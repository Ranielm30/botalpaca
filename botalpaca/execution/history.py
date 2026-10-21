"""What the broker says really happened.

The journal keeps the bot's book: one row per operation it decided to open
and later close. The broker keeps the truth: one fill per action that traded.
The two drift apart, and they drift in both directions -- the journal holds a
row for a position the broker closed in several lots, it misses the fills of
an exit that only went through in part, and it knows nothing at all about a
trade that happened without the bot ever writing it down.

When the operator asks how much they are up, only the broker's ledger adds up.
This module reads that ledger and turns it into operations, so the history
command can answer with the real numbers and with the bot's own notes hanging
on them where a journal row describes the same operation.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_LOOKBACK_DAYS = 180
MAX_PAGES = 40
PAGE_SIZE = 100
BUY = "buy"
SELL = "sell"
LONG = "LONG"
SHORT = "SHORT"
# Two journal rows for the same operation can be minutes apart in their
# closing time -- one row per lot -- so the pairing cannot demand that the
# clocks agree exactly.
MATCH_WINDOW = dt.timedelta(days=2)


@dataclass(frozen=True)
class BrokerFill:
    """One action that traded."""

    symbol: str
    side: str
    qty: float
    price: float
    at: dt.datetime
    order_id: str = ""
    order_status: str = ""


@dataclass
class BrokerOperation:
    """A position that was opened and closed again, as the broker saw it."""

    symbol: str
    direction: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    opened_at: dt.datetime
    closed_at: dt.datetime
    entry_order_id: str = ""
    exit_order_id: str = ""
    # The bot's own account of the operation, when a journal row describes it.
    strategy: str = ""
    setup: str = ""
    exit_reason: str = ""
    r_multiple: float | None = None
    registered: bool = False


@dataclass
class _Lot:
    """An open piece of a position."""

    side: str
    qty: float
    price: float
    at: dt.datetime
    order_id: str


def _parse_time(value: Any) -> dt.datetime | None:
    """Read a broker timestamp, which arrives as an ISO string."""
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _iso(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def parse_activities(activities: Iterable[dict[str, Any]]) -> list[BrokerFill]:
    """Turn Alpaca's FILL activities into fills.

    Alpaca names the time of a fill ``transaction_time`` and the state of its
    order ``order_status``; a fill that only went through in part is a real
    trade and is kept like any other.
    """
    fills: list[BrokerFill] = []
    for item in activities or []:
        if not isinstance(item, dict):
            continue
        # Only a fill is a trade. A cash dividend or a corporate action also
        # carries a symbol, a side and a price, and reading one of those as a
        # buy would invent a position that never existed.
        activity_type = str(item.get("activity_type") or "").upper()
        if activity_type and activity_type != "FILL":
            continue
        symbol = str(item.get("symbol") or "").upper()
        side = str(item.get("side") or "").lower()
        try:
            qty = float(item.get("qty") or 0.0)
            price = float(item.get("price") or 0.0)
        except (TypeError, ValueError):
            continue
        if not symbol or side not in (BUY, SELL) or qty <= 0 or price <= 0:
            continue
        at = _parse_time(item.get("transaction_time") or item.get("at"))
        if at is None:
            continue
        fills.append(
            BrokerFill(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                at=at,
                order_id=str(item.get("order_id") or ""),
                order_status=str(item.get("order_status") or ""),
            )
        )
    return fills


async def fetch_fills(
    client: Any,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list[BrokerFill]:
    """Read every fill the broker recorded, oldest first.

    The window matters: the activities endpoint answers a single day when it
    is given a date, so the range is asked for with two instants instead.
    """
    until = until or dt.datetime.now(dt.UTC)
    since = since or until - dt.timedelta(days=lookback_days)
    fills: list[BrokerFill] = []
    token: str | None = None
    for _ in range(MAX_PAGES):
        page = await client.get_activities(
            activity_types="FILL",
            direction="asc",
            page_size=PAGE_SIZE,
            after=_iso(since),
            until=_iso(until),
            page_token=token,
        )
        if not page:
            break
        fills.extend(parse_activities(page))
        last = page[-1] if isinstance(page[-1], dict) else {}
        token = str(last.get("id") or "")
        if not token or len(page) < PAGE_SIZE:
            break
    return sorted(fills, key=lambda fill: (fill.at, fill.order_id))


def match_operations(
    fills: Sequence[BrokerFill],
) -> tuple[list[BrokerOperation], list[BrokerFill]]:
    """Match fills into closed operations, oldest first.

    The matching is two-sided: a buy closes the short lots that are open and
    opens a long one when there is nothing to close, and a sell does the
    mirror. A fill bigger than the lot it closes does both -- the part that
    closes is an operation, the rest opens a new position.

    The lots that are still open come back with the operations: they are the
    positions the account is holding right now.
    """
    lots: dict[str, list[_Lot]] = {}
    operations: list[BrokerOperation] = []
    for fill in sorted(fills, key=lambda item: (item.at, item.order_id)):
        open_lots = lots.setdefault(fill.symbol, [])
        remaining = fill.qty
        while remaining > 1e-9:
            index = next(
                (i for i, lot in enumerate(open_lots) if lot.side != fill.side),
                None,
            )
            if index is None:
                break
            lot = open_lots[index]
            take = min(remaining, lot.qty)
            direction = LONG if lot.side == BUY else SHORT
            if direction == LONG:
                pnl = (fill.price - lot.price) * take
            else:
                pnl = (lot.price - fill.price) * take
            operations.append(
                BrokerOperation(
                    symbol=fill.symbol,
                    direction=direction,
                    qty=take,
                    entry_price=lot.price,
                    exit_price=fill.price,
                    pnl=pnl,
                    pnl_pct=(pnl / (lot.price * take) * 100.0) if lot.price and take else 0.0,
                    opened_at=lot.at,
                    closed_at=fill.at,
                    entry_order_id=lot.order_id,
                    exit_order_id=fill.order_id,
                )
            )
            remaining -= take
            lot.qty -= take
            if lot.qty <= 1e-9:
                open_lots.pop(index)
        if remaining > 1e-9:
            open_lots.append(
                _Lot(
                    side=fill.side,
                    qty=remaining,
                    price=fill.price,
                    at=fill.at,
                    order_id=fill.order_id,
                )
            )
    still_open = [
        BrokerFill(
            symbol=symbol,
            side=lot.side,
            qty=lot.qty,
            price=lot.price,
            at=lot.at,
            order_id=lot.order_id,
        )
        for symbol, symbol_lots in lots.items()
        for lot in symbol_lots
    ]
    return operations, still_open


def match_rows(
    operations: Sequence[BrokerOperation], rows: Sequence[Any]
) -> list[BrokerOperation]:
    """Hang the bot's own notes on the operations the broker recorded.

    A journal row is the bot's account of one lot; the broker's operation is
    what really happened. When the two describe the same thing, the row gives
    the operation its strategy, its exit reason and its R multiple. An
    operation with no row behind it is an orphan: the bot never wrote it down,
    and the card has to say so instead of hiding it.
    """
    pending = list(rows)
    for operation in operations:
        best = None
        best_gap: dt.timedelta | None = None
        for row in pending:
            if str(getattr(row, "symbol", "") or "").upper() != operation.symbol:
                continue
            if str(getattr(row, "direction", "") or "").upper() != operation.direction:
                continue
            closed_at = getattr(row, "closed_at", None)
            if closed_at is None:
                gap = dt.timedelta(0)
            else:
                closed_at = _parse_time(closed_at) or operation.closed_at
                gap = abs(closed_at - operation.closed_at)
            if gap > MATCH_WINDOW:
                continue
            if best_gap is None or gap < best_gap:
                best = row
                best_gap = gap
        if best is None:
            continue
        pending.remove(best)
        operation.strategy = str(getattr(best, "strategy", "") or "")
        operation.setup = str(getattr(best, "setup", "") or "")
        operation.exit_reason = str(getattr(best, "exit_reason", "") or "")
        r_multiple = getattr(best, "r_multiple", None)
        operation.r_multiple = float(r_multiple) if r_multiple is not None else None
        operation.registered = True
    return list(operations)
