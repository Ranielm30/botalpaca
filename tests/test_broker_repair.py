"""Repairing the journal against the broker's own ledger."""

import datetime as dt

import pytest

from botalpaca.config import AnalyticsSettings
from botalpaca.db import TradeRepository
from botalpaca.db.models import TradeModel
from botalpaca.db.repositories import DailyPnlRepository
from botalpaca.domain import TradingEnvironment
from botalpaca.execution.history import BrokerOperation
from botalpaca.journal.broker_repair import (
    BROKER_EXIT_REASON,
    BROKER_IMPORT,
    apply_plan,
    describe,
    is_broker_import,
    plan_repairs,
)
from botalpaca.learning.engine import StatisticalEngine

PAPER = TradingEnvironment.PAPER
UTC = dt.UTC


def _operation(
    symbol: str = "AAPL",
    direction: str = "LONG",
    qty: float = 10.0,
    entry: float = 100.0,
    exit_price: float = 110.0,
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


async def _row(database, **over):
    trade = TradeModel(
        environment=PAPER,
        symbol="AAPL",
        direction="LONG",
        strategy="momentum",
        setup="breakout",
        status="CLOSED",
        qty=10.0,
        entry_price=100.0,
        exit_price=99.0,
        pnl=-10.0,
        pnl_pct=-1.0,
        exit_reason="STOP",
        opened_at=dt.datetime(2024, 1, 1, 15, 0, tzinfo=UTC),
        closed_at=dt.datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
        entry_order_id="e-1",
        exit_order_id="x-old",
    )
    for key, value in over.items():
        setattr(trade, key, value)
    async with database.session() as session:
        await TradeRepository(session).create(trade)
    return trade


async def test_the_row_is_matched_with_the_operation_of_its_entry_order(database):
    row = await _row(database)
    operation = _operation()

    plan = plan_repairs([operation], [row])

    assert len(plan.fixes) == 1
    assert plan.fixes[0].row is row
    assert plan.fixes[0].operation is operation
    assert plan.inserts == []
    assert plan.unmatched_rows == []


async def test_a_row_without_an_entry_order_is_left_alone(database):
    row = await _row(database, entry_order_id="")

    plan = plan_repairs([_operation()], [row])

    assert plan.fixes == []
    assert plan.unmatched_rows == [row]
    assert len(plan.inserts) == 1


async def test_the_row_is_rewritten_with_the_numbers_the_broker_gives(database):
    await _row(database)

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_closed(PAPER))
        plan = plan_repairs([_operation(qty=4.0, exit_price=120.0)], rows)
        report = await apply_plan(session, plan, PAPER)
        assert report.fixed == 1

    async with database.session() as session:
        saved = (await TradeRepository(session).get_closed(PAPER))[0]
    assert saved.qty == pytest.approx(4.0)
    assert saved.exit_price == pytest.approx(120.0)
    assert saved.pnl == pytest.approx(80.0)
    assert saved.exit_order_id == "x-1"
    assert saved.status == "CLOSED"


async def test_an_entry_that_never_filled_but_did_trade_loses_the_wrong_reason(database):
    await _row(database, exit_reason="ENTRADA_NO_EJECUTADA", pnl=0.0, pnl_pct=0.0)

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_closed(PAPER))
        plan = plan_repairs([_operation()], rows)
        await apply_plan(session, plan, PAPER)

    async with database.session() as session:
        saved = (await TradeRepository(session).get_closed(PAPER))[0]
    assert saved.exit_reason == BROKER_EXIT_REASON
    assert saved.pnl == pytest.approx(100.0)


async def test_the_trades_the_journal_never_wrote_are_added(database):
    await _row(database)
    orphan = _operation(symbol="TSLA", entry_order="e-2", exit_order="x-2")

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_closed(PAPER))
        plan = plan_repairs([_operation(), orphan], rows)
        assert len(plan.inserts) == 1
        assert plan.inserts[0].symbol == "TSLA"
        report = await apply_plan(session, plan, PAPER)

    assert report.inserted == 1
    async with database.session() as session:
        rows = list(await TradeRepository(session).get_all(PAPER))
    imported = [row for row in rows if row.symbol == "TSLA"]
    assert len(imported) == 1
    assert imported[0].strategy == BROKER_IMPORT
    assert imported[0].status == "CLOSED"
    assert imported[0].pnl == pytest.approx(orphan.pnl)
    assert is_broker_import(imported[0])


async def test_the_daily_totals_are_counted_again(database):
    await _row(database)

    async with database.session() as session:
        rows = list(await TradeRepository(session).get_closed(PAPER))
        plan = plan_repairs([_operation()], rows)
        report = await apply_plan(session, plan, PAPER)

    assert report.days_rebuilt == 1
    async with database.session() as session:
        day = await DailyPnlRepository(session).get(PAPER, dt.date(2024, 1, 2))
    assert day is not None
    assert day.realized_pnl == pytest.approx(100.0)
    assert day.trade_count == 1
    assert day.win_count == 1


async def test_the_learning_engine_ignores_the_trades_imported_from_the_broker(database):
    await _row(database)
    await _row(
        database,
        symbol="TSLA",
        strategy=BROKER_IMPORT,
        setup=BROKER_IMPORT,
        metadata_json={"source": "broker", "imported": True},
    )
    engine = StatisticalEngine(database, settings=AnalyticsSettings())

    summary = await engine.overall(PAPER)
    by_strategy = await engine.by_strategy(PAPER)

    assert summary.sample_size == 1
    assert BROKER_IMPORT not in by_strategy


async def test_the_plan_says_how_much_money_moves(database):
    row = await _row(database, pnl=-10.0)

    plan = plan_repairs([_operation()], [row])
    lines = describe(plan)

    assert plan.book_pnl == pytest.approx(-10.0)
    assert plan.broker_pnl == pytest.approx(100.0)
    assert plan.pnl_delta == pytest.approx(110.0)
    assert any("Filas corregidas: 1" in line for line in lines)
