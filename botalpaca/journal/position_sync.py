"""Bring the open positions of the journal in line with the broker.

The journal keeps a row for every trade it planned, and that row stays open
until the broker says the position is gone. Two things break that. A row the
broker closed while the journal never noticed: the position was sold and
entered again under the same symbol, so the symbol never left the account and
the reconciler kept the old row open for good. And a position the broker holds
that the journal never wrote down at all.

``plan_position_sync`` works out what would change and how much moves;
``apply_position_sync`` is the only thing that writes, and it is only ever
called on purpose.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from botalpaca.db.models import TradeModel
from botalpaca.db.repositories import TradeRepository
from botalpaca.domain.enums import TradeStatus
from botalpaca.journal.broker_repair import (
    BROKER_EXIT_REASON,
    BROKER_IMPORT,
    BROKER_SOURCE,
    NEVER_EXECUTED,
    _as_datetime,
    _as_float,
    _rebuild_daily_pnl,
    is_broker_import,
)

_EPOCH = dt.datetime.min.replace(tzinfo=dt.UTC)

_LONG = "LONG"
_SHORT = "SHORT"


def _direction_of(side: str) -> str:
    """The side a broker fill leaves on when it opens a position."""
    return _LONG if str(side or "").lower() == "buy" else _SHORT


def _weighted_price(lots: Sequence[Any]) -> float:
    """The average price of a set of lots, weighted by their size."""
    total = 0.0
    weight = 0.0
    for lot in lots:
        qty = _as_float(getattr(lot, "qty", 0.0))
        total += _as_float(getattr(lot, "price", 0.0)) * qty
        weight += qty
    if weight <= 0:
        return 0.0
    return total / weight


@dataclass
class RowClose:
    """An open row the broker says was already sold."""

    row: Any
    operation: Any
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_order_id: str
    closed_at: dt.datetime | None
    exit_reason: str
    r_multiple: float | None


@dataclass
class RowUpdate:
    """An open row whose size or entry price the broker knows better."""

    row: Any
    qty: float
    entry_price: float


@dataclass
class RowOpen:
    """A position the broker holds and the journal never wrote down."""

    symbol: str
    direction: str
    qty: float
    entry_price: float
    opened_at: dt.datetime | None
    entry_order_id: str
    stop_price: float | None
    target_price: float | None
    risk_amount: float


@dataclass
class DuplicateRow:
    """An imported row that says the same thing as a row the journal owns."""

    row: Any
    reason: str


@dataclass
class PositionSyncPlan:
    """What the synchronisation would do, before it does anything."""

    closes: list[RowClose] = field(default_factory=list)
    updates: list[RowUpdate] = field(default_factory=list)
    inserts: list[RowOpen] = field(default_factory=list)
    duplicates: list[DuplicateRow] = field(default_factory=list)
    unmatched_rows: list[Any] = field(default_factory=list)
    broker_qty: float = 0.0
    journal_qty: float = 0.0

    @property
    def pnl_moved(self) -> float:
        """How much profit the closed rows pick up from the broker."""
        return sum(close.pnl for close in self.closes)

    @property
    def pnl_removed(self) -> float:
        """How much profit the duplicated imported rows take away."""
        return sum(_as_float(getattr(item.row, "pnl", 0.0)) for item in self.duplicates)


@dataclass
class PositionSyncReport:
    """What the synchronisation actually did."""

    closed: int = 0
    updated: int = 0
    inserted: int = 0
    removed: int = 0
    unmatched: int = 0
    days_rebuilt: int = 0
    pnl_moved: float = 0.0
    pnl_removed: float = 0.0


def _build_close(row: Any, operation: Any) -> RowClose:
    qty = _as_float(getattr(operation, "qty", 0.0))
    entry_price = _as_float(getattr(operation, "entry_price", 0.0))
    pnl = _as_float(getattr(operation, "pnl", 0.0))
    reason = str(getattr(row, "exit_reason", "") or "")
    if not reason or reason == NEVER_EXECUTED:
        reason = BROKER_EXIT_REASON
    r_multiple = None
    stop = getattr(row, "stop_price", None)
    if stop is not None and entry_price and qty:
        risk = abs(entry_price - _as_float(stop))
        if risk > 0:
            r_multiple = pnl / (risk * qty)
    return RowClose(
        row=row,
        operation=operation,
        qty=qty,
        entry_price=entry_price,
        exit_price=_as_float(getattr(operation, "exit_price", 0.0)),
        pnl=pnl,
        pnl_pct=_as_float(getattr(operation, "pnl_pct", 0.0)),
        exit_order_id=str(getattr(operation, "exit_order_id", "") or ""),
        closed_at=_as_datetime(getattr(operation, "closed_at", None)),
        exit_reason=reason,
        r_multiple=r_multiple,
    )


def _protection_for(
    protection: Mapping[str, Mapping[str, Any]] | None, symbol: str
) -> tuple[float | None, float | None]:
    if not protection:
        return None, None
    entry = protection.get(symbol.upper()) or {}
    stop = entry.get("stop_price")
    target = entry.get("target_price")
    return (
        _as_float(stop) if stop is not None else None,
        _as_float(target) if target is not None else None,
    )


def plan_position_sync(
    operations: Sequence[Any],
    open_rows: Sequence[Any],
    open_lots: Sequence[Any],
    protection: Mapping[str, Mapping[str, Any]] | None = None,
    all_rows: Sequence[Any] | None = None,
) -> PositionSyncPlan:
    """Work out how the open rows have to change to match the broker.

    The broker's own ledger is the account of record: a lot it still holds is
    open, a lot it sold is closed. Rows are paired with lots through the entry
    order, because that is the one thing both sides recorded.

    ``all_rows`` is every row of the journal, and it is where the imported
    rows live: the ones a previous repair brought back from the broker.
    """
    lots_by_order: dict[str, list[Any]] = {}
    for lot in open_lots:
        key = str(getattr(lot, "order_id", "") or "")
        if key:
            lots_by_order.setdefault(key, []).append(lot)
    operations_by_order: dict[str, list[Any]] = {}
    for operation in operations:
        key = str(getattr(operation, "entry_order_id", "") or "")
        if key:
            operations_by_order.setdefault(key, []).append(operation)

    plan = PositionSyncPlan()
    consumed: set[str] = set()
    for row in open_rows:
        key = str(getattr(row, "entry_order_id", "") or "")
        lots = [lot for lot in lots_by_order.get(key, []) if key not in consumed]
        if lots:
            consumed.add(key)
            qty = sum(_as_float(getattr(lot, "qty", 0.0)) for lot in lots)
            price = _weighted_price(lots)
            if (
                abs(qty - _as_float(getattr(row, "qty", 0.0))) > 1e-6
                or abs(price - _as_float(getattr(row, "entry_price", 0.0))) > 1e-4
            ):
                plan.updates.append(RowUpdate(row=row, qty=qty, entry_price=price))
            continue
        candidates = operations_by_order.get(key, [])
        if candidates:
            closed_at = _as_datetime(getattr(row, "closed_at", None)) or _EPOCH
            operation = min(
                candidates,
                key=lambda item: abs(
                    (_as_datetime(getattr(item, "closed_at", None)) or _EPOCH) - closed_at
                ),
            )
            plan.closes.append(_build_close(row, operation))
            continue
        plan.unmatched_rows.append(row)

    # Whatever the broker still holds and no row accounts for is a position the
    # journal never wrote down. It is written now, or the risk engine goes on
    # believing the account is lighter than it is.
    for key, lots in lots_by_order.items():
        if key in consumed:
            continue
        first = lots[0]
        symbol = str(getattr(first, "symbol", "") or "").upper()
        qty = sum(_as_float(getattr(lot, "qty", 0.0)) for lot in lots)
        stop_price, target_price = _protection_for(protection, symbol)
        entry_price = _weighted_price(lots)
        risk = 0.0
        if stop_price is not None and entry_price:
            risk = abs(entry_price - stop_price) * qty
        plan.inserts.append(
            RowOpen(
                symbol=symbol,
                direction=_direction_of(str(getattr(first, "side", "") or "")),
                qty=qty,
                entry_price=entry_price,
                opened_at=_as_datetime(getattr(first, "at", None)),
                entry_order_id=key,
                stop_price=stop_price,
                target_price=target_price,
                risk_amount=risk,
            )
        )

    # A row the journal owns and a row the broker brought back can describe the
    # same trade. When the first is closed with the broker's numbers, the
    # second says it again, and the profit would be counted twice.
    for close in plan.closes:
        for row in all_rows if all_rows is not None else open_rows:
            if not is_broker_import(row):
                continue
            if str(getattr(row, "entry_order_id", "") or "") != str(
                getattr(close.operation, "entry_order_id", "") or ""
            ):
                continue
            if abs(_as_float(getattr(row, "qty", 0.0)) - close.qty) > 1e-6:
                continue
            if abs(_as_float(getattr(row, "pnl", 0.0)) - close.pnl) > 0.01:
                continue
            plan.duplicates.append(
                DuplicateRow(row=row, reason="la fila del diario ya cuenta esa operación")
            )
            break

    plan.broker_qty = sum(_as_float(getattr(lot, "qty", 0.0)) for lot in open_lots)
    plan.journal_qty = sum(_as_float(getattr(row, "qty", 0.0)) for row in open_rows)
    return plan


def describe(plan: PositionSyncPlan) -> list[str]:
    """A short, plain report of what the synchronisation would do."""
    return [
        f"Filas abiertas que el broker ya cerró: {len(plan.closes)}",
        f"Filas abiertas corregidas: {len(plan.updates)}",
        f"Posiciones del broker sin fila: {len(plan.inserts)}",
        f"Filas importadas duplicadas a borrar: {len(plan.duplicates)}",
        f"Filas abiertas sin pareja en el broker: {len(plan.unmatched_rows)}",
        f"Acciones abiertas del diario: {plan.journal_qty:,.4g}",
        f"Acciones abiertas del broker: {plan.broker_qty:,.4g}",
        f"P&L que cambia de sitio: {plan.pnl_moved:,.2f} $",
        f"P&L que se retira de las duplicadas: {plan.pnl_removed:,.2f} $",
        f"Diferencia neta: {plan.pnl_moved - plan.pnl_removed:+,.2f} $",
    ]


async def apply_position_sync(
    session: Any, plan: PositionSyncPlan, environment: Any
) -> PositionSyncReport:
    """Write the synchronisation into the journal.

    The rows the broker closed are closed with the broker's numbers, the rows
    that were wrong are corrected, the positions the journal never wrote are
    added, and the imported rows that said the same thing are removed so the
    profit is only counted once.
    """
    trades = TradeRepository(session)
    days: set[dt.date] = set()
    report = PositionSyncReport()

    for close in plan.closes:
        row = close.row
        row.qty = close.qty
        row.entry_price = close.entry_price
        row.exit_price = close.exit_price
        row.pnl = close.pnl
        row.pnl_pct = close.pnl_pct
        row.r_multiple = close.r_multiple if close.r_multiple is not None else 0.0
        row.exit_order_id = close.exit_order_id or None
        row.exit_reason = close.exit_reason
        row.status = TradeStatus.CLOSED.value
        if close.closed_at is not None:
            row.closed_at = close.closed_at
            days.add(close.closed_at.date())
            opened = _as_datetime(getattr(row, "opened_at", None))
            if opened is not None:
                row.duration_seconds = (close.closed_at - opened).total_seconds()
        report.closed += 1
        report.pnl_moved += close.pnl

    for update in plan.updates:
        update.row.qty = update.qty
        update.row.entry_price = update.entry_price
        report.updated += 1

    for insert in plan.inserts:
        row = TradeModel(
            environment=environment,
            symbol=insert.symbol,
            direction=insert.direction,
            strategy=BROKER_IMPORT,
            setup=BROKER_IMPORT,
            status=TradeStatus.OPEN.value,
            qty=insert.qty,
            entry_price=insert.entry_price,
            stop_price=insert.stop_price,
            target_price=insert.target_price,
            risk_amount=insert.risk_amount,
            entry_order_id=insert.entry_order_id or None,
            metadata_json={"source": BROKER_SOURCE, "imported": True},
        )
        if insert.opened_at is not None:
            row.opened_at = insert.opened_at
            row.filled_at = insert.opened_at
        await trades.create(row)
        report.inserted += 1

    for duplicate in plan.duplicates:
        report.pnl_removed += _as_float(getattr(duplicate.row, "pnl", 0.0))
        await session.delete(duplicate.row)
        report.removed += 1

    report.unmatched = len(plan.unmatched_rows)
    report.days_rebuilt = await _rebuild_daily_pnl(session, environment, days)
    return report
