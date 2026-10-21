"""Bring the journal back in line with the broker's own ledger.

The journal is the bot's account of what it traded. The broker's fills are
what really happened. When the two disagree the broker is right: a row that
was closed with the whole position while the broker only ever sold part of it
is simply wrong, and a trade the broker recorded while the journal never wrote
it down is missing.

The repair is planned before it is applied. ``plan_repairs`` says what would
change and how much money moves; ``apply_plan`` is the only thing that writes,
and it is only ever called on purpose.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Sequence

from botalpaca.db.models import TradeModel
from botalpaca.db.repositories import DailyPnlRepository, TradeRepository
from botalpaca.domain.enums import TradeStatus

#: The strategy and setup written on a row the broker brought back.
BROKER_IMPORT = "BROKER"
#: The marker kept in the row metadata so the row is never mistaken for one
#: the bot planned itself.
BROKER_SOURCE = "broker"
#: The exit reason used when the journal had none to give.
BROKER_EXIT_REASON = "CIERRE_BROKER"
#: The reason the bot writes when an entry never went through.
NEVER_EXECUTED = "ENTRADA_NO_EJECUTADA"

_EPOCH = dt.datetime.min.replace(tzinfo=dt.UTC)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_datetime(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    return None


def _row_closed_at(row: Any) -> dt.datetime | None:
    return _as_datetime(getattr(row, "closed_at", None))


def _operation_closed_at(operation: Any) -> dt.datetime | None:
    return _as_datetime(getattr(operation, "closed_at", None))


def is_broker_import(row: Any) -> bool:
    """True when the row was imported from the broker rather than planned."""
    metadata = getattr(row, "metadata_json", None)
    if not isinstance(metadata, dict):
        return False
    return str(metadata.get("source") or "") == BROKER_SOURCE


@dataclass
class RowFix:
    """What one journal row has to become to match the broker."""

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
    previous_pnl: float
    previous_qty: float


@dataclass
class RowInsert:
    """A trade the broker recorded and the journal never wrote down."""

    operation: Any
    symbol: str
    direction: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    opened_at: dt.datetime | None
    closed_at: dt.datetime | None
    entry_order_id: str
    exit_order_id: str


@dataclass
class RepairPlan:
    """What the repair would do, before it does anything."""

    fixes: list[RowFix] = field(default_factory=list)
    inserts: list[RowInsert] = field(default_factory=list)
    unmatched_rows: list[Any] = field(default_factory=list)
    book_pnl: float = 0.0
    broker_pnl: float = 0.0

    @property
    def pnl_delta(self) -> float:
        """How much the journal's profit changes once the repair lands."""
        return self.broker_pnl - self.book_pnl

    @property
    def fixed_pnl(self) -> float:
        return sum(fix.pnl for fix in self.fixes)

    @property
    def inserted_pnl(self) -> float:
        return sum(insert.pnl for insert in self.inserts)


@dataclass
class RepairReport:
    """What the repair actually did."""

    fixed: int = 0
    inserted: int = 0
    unmatched: int = 0
    days_rebuilt: int = 0
    book_pnl: float = 0.0
    broker_pnl: float = 0.0


def _match_operations(
    operations: Sequence[Any], rows: Sequence[Any]
) -> tuple[list[tuple[Any, Any]], list[Any]]:
    """Pair every journal row with the broker operation that closed its lot.

    The entry order is the key: the journal wrote down the order that opened
    the lot and the broker knows the same order, so the two describe the same
    trade even when the exit was split in pieces. When one entry produced
    several exits the closest close in time wins, and the rest stay free to be
    imported on their own.
    """
    by_entry: dict[str, list[Any]] = {}
    for operation in operations:
        key = str(getattr(operation, "entry_order_id", "") or "")
        if key:
            by_entry.setdefault(key, []).append(operation)
    for group in by_entry.values():
        group.sort(key=lambda op: _operation_closed_at(op) or _EPOCH)

    used: set[int] = set()
    pairs: list[tuple[Any, Any]] = []
    unmatched: list[Any] = []

    def sort_key(row: Any) -> tuple[dt.datetime, int]:
        return (
            _row_closed_at(row) or _EPOCH,
            int(getattr(row, "id", 0) or 0),
        )

    for row in sorted(rows, key=sort_key):
        key = str(getattr(row, "entry_order_id", "") or "")
        candidates = [
            op
            for op in by_entry.get(key, [])
            if id(op) not in used
        ]
        if not candidates:
            unmatched.append(row)
            continue
        row_at = _row_closed_at(row)
        if row_at is None:
            chosen = candidates[0]
        else:
            chosen = min(
                candidates,
                key=lambda op: abs((_operation_closed_at(op) or row_at) - row_at),
            )
        used.add(id(chosen))
        pairs.append((row, chosen))
    return pairs, unmatched


def _build_fix(row: Any, operation: Any) -> RowFix:
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
    return RowFix(
        row=row,
        operation=operation,
        qty=qty,
        entry_price=entry_price,
        exit_price=_as_float(getattr(operation, "exit_price", 0.0)),
        pnl=pnl,
        pnl_pct=_as_float(getattr(operation, "pnl_pct", 0.0)),
        exit_order_id=str(getattr(operation, "exit_order_id", "") or ""),
        closed_at=_operation_closed_at(operation),
        exit_reason=reason,
        r_multiple=r_multiple,
        previous_pnl=_as_float(getattr(row, "pnl", 0.0)),
        previous_qty=_as_float(getattr(row, "qty", 0.0)),
    )


def _build_insert(operation: Any) -> RowInsert:
    return RowInsert(
        operation=operation,
        symbol=str(getattr(operation, "symbol", "") or "").upper(),
        direction=str(getattr(operation, "direction", "") or "").upper(),
        qty=_as_float(getattr(operation, "qty", 0.0)),
        entry_price=_as_float(getattr(operation, "entry_price", 0.0)),
        exit_price=_as_float(getattr(operation, "exit_price", 0.0)),
        pnl=_as_float(getattr(operation, "pnl", 0.0)),
        pnl_pct=_as_float(getattr(operation, "pnl_pct", 0.0)),
        opened_at=_as_datetime(getattr(operation, "opened_at", None)),
        closed_at=_as_datetime(getattr(operation, "closed_at", None)),
        entry_order_id=str(getattr(operation, "entry_order_id", "") or ""),
        exit_order_id=str(getattr(operation, "exit_order_id", "") or ""),
    )


def plan_repairs(operations: Sequence[Any], rows: Sequence[Any]) -> RepairPlan:
    """Work out what has to change for the journal to tell the truth."""
    pairs, unmatched = _match_operations(operations, rows)
    plan = RepairPlan(unmatched_rows=list(unmatched))
    for row, operation in pairs:
        plan.fixes.append(_build_fix(row, operation))
    matched = {id(operation) for _, operation in pairs}
    for operation in operations:
        if id(operation) not in matched:
            plan.inserts.append(_build_insert(operation))
    plan.book_pnl = sum(_as_float(getattr(row, "pnl", 0.0)) for row in rows)
    plan.broker_pnl = sum(_as_float(getattr(op, "pnl", 0.0)) for op in operations)
    return plan


def describe(plan: RepairPlan) -> list[str]:
    """A short, plain report of what the repair would do."""
    return [
        f"Filas corregidas: {len(plan.fixes)}",
        f"Operaciones importadas del broker: {len(plan.inserts)}",
        f"Filas sin pareja en el broker: {len(plan.unmatched_rows)}",
        f"P&L del diario: {plan.book_pnl:,.2f} $",
        f"P&L del broker: {plan.broker_pnl:,.2f} $",
        f"Diferencia: {plan.pnl_delta:+,.2f} $",
    ]


async def _rebuild_daily_pnl(
    session: Any, environment: Any, days: set[dt.date]
) -> int:
    """Recompute the daily totals from the trades that are now in the book.

    The loss limits read these numbers, so a day whose rows were rewritten
    has to be counted again: otherwise the limits keep believing the old,
    wrong story.
    """
    if not days:
        return 0
    trades = TradeRepository(session)
    daily = DailyPnlRepository(session)
    closed = list(await trades.get_closed(environment))
    rebuilt = 0
    for day in sorted(days):
        rows = [
            row
            for row in closed
            if (_row_closed_at(row) or _EPOCH).date() == day
        ]
        if not rows:
            continue
        entry = await daily.upsert_day(environment, day)
        pnl = sum(_as_float(getattr(row, "pnl", 0.0)) for row in rows)
        wins = [row for row in rows if _as_float(getattr(row, "pnl", 0.0)) > 0]
        losses = [row for row in rows if _as_float(getattr(row, "pnl", 0.0)) < 0]
        entry.realized_pnl = pnl
        entry.trade_count = len(rows)
        entry.win_count = len(wins)
        entry.loss_count = len(losses)
        entry.gross_profit = sum(
            _as_float(getattr(row, "pnl", 0.0)) for row in wins
        )
        entry.gross_loss = abs(
            sum(_as_float(getattr(row, "pnl", 0.0)) for row in losses)
        )
        rebuilt += 1
    return rebuilt


async def apply_plan(session: Any, plan: RepairPlan, environment: Any) -> RepairReport:
    """Write the plan into the journal.

    The rows that had a trade behind them are rewritten with the broker's
    numbers; the ones the journal never had are added as closed rows, marked
    as imported so nothing mistakes them for trades the bot planned.
    """
    trades = TradeRepository(session)
    days: set[dt.date] = set()
    fixed = 0
    for fix in plan.fixes:
        row = fix.row
        row.qty = fix.qty
        row.entry_price = fix.entry_price
        row.exit_price = fix.exit_price
        row.pnl = fix.pnl
        row.pnl_pct = fix.pnl_pct
        row.r_multiple = fix.r_multiple if fix.r_multiple is not None else 0.0
        row.exit_order_id = fix.exit_order_id or None
        row.exit_reason = fix.exit_reason
        row.status = TradeStatus.CLOSED.value
        if fix.closed_at is not None:
            row.closed_at = fix.closed_at
            days.add(fix.closed_at.date())
            opened = _as_datetime(getattr(row, "opened_at", None))
            if opened is not None:
                row.duration_seconds = (fix.closed_at - opened).total_seconds()
        fixed += 1

    inserted = 0
    for insert in plan.inserts:
        row = TradeModel(
            environment=environment,
            symbol=insert.symbol,
            direction=insert.direction,
            strategy=BROKER_IMPORT,
            setup=BROKER_IMPORT,
            status=TradeStatus.CLOSED.value,
            qty=insert.qty,
            entry_price=insert.entry_price,
            exit_price=insert.exit_price,
            pnl=insert.pnl,
            pnl_pct=insert.pnl_pct,
            exit_reason=BROKER_EXIT_REASON,
            entry_order_id=insert.entry_order_id or None,
            exit_order_id=insert.exit_order_id or None,
            risk_amount=0.0,
            metadata_json={"source": BROKER_SOURCE, "imported": True},
        )
        if insert.opened_at is not None:
            row.opened_at = insert.opened_at
        if insert.closed_at is not None:
            row.closed_at = insert.closed_at
            days.add(insert.closed_at.date())
            if insert.opened_at is not None:
                row.duration_seconds = (
                    insert.closed_at - insert.opened_at
                ).total_seconds()
        await trades.create(row)
        inserted += 1

    rebuilt = await _rebuild_daily_pnl(session, environment, days)
    return RepairReport(
        fixed=fixed,
        inserted=inserted,
        unmatched=len(plan.unmatched_rows),
        days_rebuilt=rebuilt,
        book_pnl=plan.book_pnl,
        broker_pnl=plan.broker_pnl,
    )
