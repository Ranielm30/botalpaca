"""Bringing the open positions of the journal in line with the broker."""

import datetime as dt

import pytest

from botalpaca.db import TradeRepository
from botalpaca.db.models import TradeModel
from botalpaca.db.repositories import DailyPnlRepository
from botalpaca.domain import TradingEnvironment
from botalpaca.execution.history import BrokerFill, BrokerOperation
from botalpaca.journal.broker_repair import BROKER_IMPORT
from botalpaca.journal.position_sync import (
    apply_position_sync,
    describe,
    plan_position_sync,
)

PAPER = TradingEnvironment.PAPER
UTC = dt.UTC


def _operation(
    symbol: str = "AAPL",
    direction: str = "LONG",
    qty: float = 10.0,
    entry: float = 100.0,
    exit_price: float = 90.0,
    closed: dt.datetime | None = None,
    entry_order: str = "e-1",
    exit_order: str = "x-1",
) -> BrokerOperation:
    pnl = (exit_price - entry) * qty
    if direction == "SHORT":
        pnl = (entry - exit_price) * qty
    return BrokerOperation(
        symbol=symbol,
        direction=direction,
        qty=qty,
        entry_price=entry,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl / (entry * qty) * 100.0,
        opened_at=dt.datetime(2024, 1, 1, 15, 0, tzinfo=UTC),
        closed_at=closed or dt.datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
        entry_order_id=entry_order,
        exit_order_id=exit_order,
    )


def _lot(
    symbol: str = "AAPL",
    side: str = "buy",
    qty: float = 10.0,
    price: float = 100.0,
    at: dt.datetime | None = None,
    order_id: str = "e-1",
) -> BrokerFill:
    return BrokerFill(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        at=at or dt.datetime(2024, 1, 1, 15, 0, tzinfo=UTC),
        order_id=order_id,
        order_status="filled",
    )


async def _open_row(database, **over):
    trade = TradeModel(
        environment=PAPER,
        symbol="AAPL",
        direction="LONG",
        strategy="momentum",
        setup="breakout",
        status="OPEN",
        qty=10.0,
        entry_price=100.0,
        risk_amount=1000.0,
        opened_at=dt.datetime(2024, 1, 1, 15, 0, tzinfo=UTC),
        entry_order_id="e-1",
    )
    for key, value in over.items():
        setattr(trade, key, value)
    async with database.session() as session:
        await TradeRepository(session).create(trade)
    return trade


async def _imported_row(database, **over):
    """A row a previous repair brought back from the broker."""
    trade = TradeModel(
        environment=PAPER,
        symbol="AAPL",
        direction="LONG",
        strategy=BROKER_IMPORT,
        setup=BROKER_IMPORT,
        status="CLOSED",
        qty=10.0,
        entry_price=100.0,
        exit_price=90.0,
        pnl=-100.0,
        pnl_pct=-10.0,
        exit_reason="CIERRE_BROKER",
        opened_at=dt.datetime(2024, 1, 1, 15, 0, tzinfo=UTC),
        closed_at=dt.datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
        entry_order_id="e-1",
        exit_order_id="x-1",
        metadata_json={"source": "broker", "imported": True},
    )
    for key, value in over.items():
        setattr(trade, key, value)
    async with database.session() as session:
        await TradeRepository(session).create(trade)
    return trade


async def test_a_row_the_broker_sold_is_closed_with_the_brokers_numbers(database):
    await _open_row(database)

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_open_positions(PAPER))
        plan = plan_position_sync([_operation(qty=4.0, exit_price=90.0)], rows, [])
        assert len(plan.closes) == 1
        assert plan.closes[0].qty == 4.0
        assert plan.closes[0].pnl == pytest.approx(-40.0)
        report = await apply_position_sync(session, plan, PAPER)
        assert report.closed == 1

    async with database.session() as session:
        saved = (await TradeRepository(session).get_open_positions(PAPER))
        assert saved == []
        closed = (await TradeRepository(session).get_closed(PAPER))[0]
        assert closed.status == "CLOSED"
        assert closed.qty == 4.0
        assert closed.pnl == pytest.approx(-40.0)


async def test_an_open_row_the_broker_still_holds_is_corrected(database):
    await _open_row(database, qty=10.0, entry_price=100.0)

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_open_positions(PAPER))
        plan = plan_position_sync([], rows, [_lot(qty=8.0, price=101.0)])
        assert len(plan.updates) == 1
        assert plan.updates[0].qty == 8.0
        assert plan.updates[0].entry_price == pytest.approx(101.0)
        report = await apply_position_sync(session, plan, PAPER)
        assert report.updated == 1

    async with database.session() as session:
        saved = (await TradeRepository(session).get_open_positions(PAPER))[0]
        assert saved.qty == 8.0
        assert saved.entry_price == pytest.approx(101.0)


async def test_a_position_the_journal_never_wrote_is_added(database):
    async with database.session() as session:
        rows = list(await TradeRepository(session).get_open_positions(PAPER))
        plan = plan_position_sync(
            [],
            rows,
            [_lot(symbol="TSLA", qty=7.0, price=380.0, order_id="e-9")],
            {"TSLA": {"stop_price": 370.0, "target_price": 400.0}},
        )
        assert len(plan.inserts) == 1
        assert plan.inserts[0].symbol == "TSLA"
        assert plan.inserts[0].direction == "LONG"
        assert plan.inserts[0].qty == 7.0
        assert plan.inserts[0].risk_amount == pytest.approx(70.0)
        report = await apply_position_sync(session, plan, PAPER)
        assert report.inserted == 1

    async with database.session() as session:
        saved = list(await TradeRepository(session).get_open_positions(PAPER))
        assert len(saved) == 1
        assert saved[0].symbol == "TSLA"
        assert saved[0].status == "OPEN"
        assert saved[0].stop_price == pytest.approx(370.0)


async def test_the_imported_row_that_says_the_same_thing_is_removed(database):
    await _open_row(database)
    await _imported_row(database)

    async with database.session() as session:
        trades = TradeRepository(session)
        open_rows = list(await trades.get_open_positions(PAPER))
        all_rows = list(await trades.get_all(PAPER, limit=5000))
        plan = plan_position_sync(
            [_operation(qty=10.0, entry=100.0, exit_price=90.0)],
            open_rows,
            [],
            None,
            all_rows,
        )
        assert len(plan.closes) == 1
        assert len(plan.duplicates) == 1
        report = await apply_position_sync(session, plan, PAPER)
        assert report.closed == 1
        assert report.removed == 1

    async with database.session() as session:
        closed = list(await TradeRepository(session).get_closed(PAPER))
        assert len(closed) == 1
        assert closed[0].strategy == "momentum"


async def test_the_profit_is_only_counted_once(database):
    await _open_row(database)
    await _imported_row(database)

    async with database.session() as session:
        trades = TradeRepository(session)
        before = sum(row.pnl for row in await trades.get_closed(PAPER))
        open_rows = list(await trades.get_open_positions(PAPER))
        all_rows = list(await trades.get_all(PAPER, limit=5000))
        plan = plan_position_sync(
            [_operation(qty=10.0, entry=100.0, exit_price=90.0)],
            open_rows,
            [],
            None,
            all_rows,
        )
        await apply_position_sync(session, plan, PAPER)

    async with database.session() as session:
        after = sum(row.pnl for row in await TradeRepository(session).get_closed(PAPER))
        assert after == pytest.approx(before)


async def test_the_daily_totals_are_counted_again(database):
    await _open_row(database)

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_open_positions(PAPER))
        plan = plan_position_sync([_operation(qty=10.0, exit_price=90.0)], rows, [])
        report = await apply_position_sync(session, plan, PAPER)
        assert report.days_rebuilt == 1

    async with database.session() as session:
        day = await DailyPnlRepository(session).get(PAPER, dt.date(2024, 1, 2))
        assert day is not None
        assert day.realized_pnl == pytest.approx(-100.0)
        assert day.trade_count == 1


async def test_a_row_the_broker_knows_nothing_about_is_left_alone(database):
    await _open_row(database, entry_order_id="e-unknown")

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_open_positions(PAPER))
        plan = plan_position_sync([_operation(entry_order="e-other")], rows, [])
        assert plan.closes == []
        assert plan.updates == []
        assert plan.unmatched_rows == [rows[0]]


async def test_the_plan_says_how_much_moves(database):
    await _open_row(database)

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_open_positions(PAPER))
        plan = plan_position_sync([_operation(qty=10.0, exit_price=90.0)], rows, [])
        lines = describe(plan)
        text = "\n".join(lines)
        assert "Filas abiertas que el broker ya cerró: 1" in text
        assert "-100.00" in text
