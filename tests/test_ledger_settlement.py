"""The ledger has to close trades the broker already finished.

Alpaca fills a protective stop or a take-profit without being asked. When it
does, the position is gone but the row still says OPEN, so the trade kept
counting as open risk and never reached the statistics. Reconciliation is what
compares SQLite against Alpaca, so that is where the settlement belongs.

These tests drive ``_settle_finished_trades`` unbound against a real SQLite
journal and a stubbed broker.
"""

from __future__ import annotations

import datetime as dt
from functools import partial
from types import SimpleNamespace

import pytest

from botalpaca.app import Application
from botalpaca.db.repositories import TradeRepository
from botalpaca.domain.enums import MarketRegime, SetupType, SignalDirection, TradingEnvironment
from botalpaca.journal.service import TradeJournal

PAPER = TradingEnvironment.PAPER
UTC = dt.UTC


# --------------------------------------------------------------------------- doubles
def _order(
    order_id: str,
    *,
    side: str = "sell",
    status: str = "filled",
    order_type: str = "stop",
    price: float | None = 98.0,
    qty: float = 10.0,
    filled_at: dt.datetime | None = None,
    legs=None,
):
    """One Alpaca order, shaped like the entity the trading client returns."""
    return SimpleNamespace(
        id=order_id,
        side=SimpleNamespace(value=side),
        status=SimpleNamespace(value=status),
        type=SimpleNamespace(value=order_type),
        legs=legs,
        filled_at=filled_at,
        filled_avg_price=price,
        filled_qty=qty,
    )


class _Client:
    def __init__(self, orders):
        self._orders = orders
        self.calls = []

    async def get_orders(self, *, status, symbols=None, limit=100, nested=True):
        self.calls.append({"status": status, "symbols": symbols, "nested": nested})
        return list(self._orders)


class _App:
    """Enough Application to settle trades: a real journal, a fake broker.

    ``_settle_trade`` is bound to the real implementation because the method
    under test calls it on ``self``; the stub stands in for the rest of the
    object graph, not for the code being tested.
    """

    def __init__(self, database, orders=(), last_price=100.0):
        self.database = database
        self.journal = TradeJournal(database)
        self.active = SimpleNamespace(
            trading_client=_Client(list(orders)),
            market=SimpleNamespace(get_last_price=self._last_price),
        )
        self.last_price = last_price
        self._settle_trade = partial(Application._settle_trade, self)
        self._exit_fills = partial(Application._exit_fills, self)

    async def _last_price(self, symbol: str) -> float:
        return self.last_price


def _position(symbol: str, qty: float):
    return SimpleNamespace(symbol=symbol, qty=qty)


async def _settle(app, positions):
    return await Application._settle_finished_trades(app, PAPER, positions)


async def _open(journal, *, symbol="AAPL", direction=SignalDirection.LONG, **over):
    long = direction == SignalDirection.LONG
    kwargs = dict(
        environment=PAPER,
        symbol=symbol,
        direction=direction,
        strategy=SetupType.PULLBACK,
        setup=SetupType.PULLBACK,
        timeframe="1D",
        qty=10.0,
        entry_price=100.0,
        stop_price=98.0 if long else 102.0,
        target_price=106.0 if long else 92.0,
        score=80.0,
        rr=2.0,
        risk_amount=20.0,
        atr=2.5,
        regime=MarketRegime.TRENDING_UP.value,
        sector="TECHNOLOGY",
        filled_at=dt.datetime.now(UTC),
    )
    kwargs.update(over)
    return await journal.open_trade(**kwargs)


async def _row(database, trade_id: int, environment=PAPER):
    async with database.session() as session:
        rows = await TradeRepository(session).get_all(environment, limit=200)
    return next(r for r in rows if r.id == trade_id)


# --------------------------------------------------------------------------- the stop
async def test_a_stop_alpaca_filled_closes_the_ledger_row(database):
    trade = await _open(TradeJournal(database), stop_order_id="sl-1")
    app = _App(
        database,
        orders=[
            _order(
                "sl-1",
                order_type="stop",
                price=98.0,
                filled_at=dt.datetime.now(UTC) + dt.timedelta(seconds=5),
            )
        ],
    )

    notes = await _settle(app, [])

    assert len(notes) == 1
    assert "STOP" in notes[0]
    row = await _row(database, trade.id)
    assert row.status == "CLOSED"
    assert row.exit_reason == "STOP"
    assert row.exit_price == pytest.approx(98.0)
    assert row.pnl == pytest.approx(-20.0)
    assert row.r_multiple == pytest.approx(-1.0)
    assert row.closed_at is not None
    assert row.exit_order_id == "sl-1"


async def test_a_take_profit_alpaca_filled_closes_the_ledger_row(database):
    trade = await _open(TradeJournal(database), take_profit_order_id="tp-1")
    app = _App(
        database,
        orders=[
            _order(
                "tp-1",
                order_type="limit",
                price=106.0,
                filled_at=dt.datetime.now(UTC) + dt.timedelta(seconds=5),
            )
        ],
    )

    notes = await _settle(app, [])

    assert "TARGET" in notes[0]
    row = await _row(database, trade.id)
    assert row.status == "CLOSED"
    assert row.exit_reason == "TARGET"
    assert row.pnl == pytest.approx(60.0)
    assert row.r_multiple == pytest.approx(3.0)


async def test_the_canceled_leg_of_the_pair_is_not_mistaken_for_the_exit(database):
    """In a bracket Alpaca cancels the losing leg; it must not settle anything."""
    trade = await _open(TradeJournal(database), stop_order_id="sl-1", take_profit_order_id="tp-1")
    app = _App(
        database,
        orders=[
            _order("tp-1", order_type="limit", price=106.0, status="filled",
                   filled_at=dt.datetime.now(UTC) + dt.timedelta(seconds=5)),
            _order("sl-1", order_type="stop", price=98.0, status="canceled", qty=None),
        ],
    )

    await _settle(app, [])

    row = await _row(database, trade.id)
    assert row.exit_reason == "TARGET"
    assert row.pnl == pytest.approx(60.0)


# --------------------------------------------------------------------------- direction
async def test_a_short_is_settled_on_the_buy_side(database):
    trade = await _open(
        TradeJournal(database), symbol="SPY", direction=SignalDirection.SHORT, stop_order_id="sl-2"
    )
    app = _App(
        database,
        orders=[
            _order(
                "sl-2",
                side="buy",
                order_type="stop",
                price=102.0,
                filled_at=dt.datetime.now(UTC) + dt.timedelta(seconds=5),
            )
        ],
    )

    await _settle(app, [])

    row = await _row(database, trade.id)
    assert row.status == "CLOSED"
    assert row.pnl == pytest.approx(-20.0)
    assert row.r_multiple == pytest.approx(-1.0)


async def test_the_entry_order_of_a_short_is_not_counted_as_its_exit(database):
    """A short is entered with a sell; counting that sell would invent a win."""
    trade = await _open(
        TradeJournal(database), symbol="SPY", direction=SignalDirection.SHORT, stop_order_id="sl-2"
    )
    app = _App(
        database,
        orders=[
            _order("entry-1", side="sell", order_type="market", price=100.0,
                   filled_at=dt.datetime.now(UTC) - dt.timedelta(seconds=1)),
            _order("sl-2", side="buy", order_type="stop", price=102.0,
                   filled_at=dt.datetime.now(UTC) + dt.timedelta(seconds=5)),
        ],
    )

    await _settle(app, [])

    row = await _row(database, trade.id)
    assert row.exit_price == pytest.approx(102.0)
    assert row.pnl == pytest.approx(-20.0)


# --------------------------------------------------------------------------- timing
async def test_a_fill_from_an_earlier_trade_on_the_same_symbol_is_ignored(database):
    """Same symbol, older trade. Counting its exit would invent a P&L."""
    trade = await _open(TradeJournal(database), stop_order_id="sl-3")
    app = _App(
        database,
        orders=[
            _order("old-exit", price=150.0,
                   filled_at=dt.datetime.now(UTC) - dt.timedelta(days=3)),
            _order("sl-3", price=98.0,
                   filled_at=dt.datetime.now(UTC) + dt.timedelta(seconds=5)),
        ],
        last_price=97.0,
    )

    await _settle(app, [])

    row = await _row(database, trade.id)
    assert row.exit_price == pytest.approx(98.0)
    assert row.pnl == pytest.approx(-20.0)


async def test_partial_exits_are_averaged_by_quantity(database):
    trade = await _open(TradeJournal(database), stop_order_id="sl-4", qty=10.0)
    when = dt.datetime.now(UTC) + dt.timedelta(seconds=5)
    app = _App(
        database,
        orders=[
            _order("manual-1", order_type="market", price=104.0, qty=4.0, filled_at=when),
            _order("sl-4", order_type="stop", price=98.0, qty=6.0, filled_at=when),
        ],
    )

    await _settle(app, [])

    row = await _row(database, trade.id)
    # (104*4 + 98*6) / 10 = 100.40
    assert row.exit_price == pytest.approx(100.4)
    assert row.pnl == pytest.approx(4.0)


# --------------------------------------------------------------------------- no exit
async def test_an_entry_that_never_filled_is_closed_without_pnl(database):
    trade = await _open(TradeJournal(database), filled_at=None, stop_order_id="sl-5")
    app = _App(database, orders=[], last_price=137.0)

    notes = await _settle(app, [])

    assert "ENTRADA_NO_EJECUTADA" in notes[0]
    row = await _row(database, trade.id)
    assert row.status == "CLOSED"
    assert row.pnl == pytest.approx(0.0)
    assert row.exit_price == pytest.approx(100.0)


async def test_a_vanished_position_with_no_fill_is_priced_as_an_estimate_and_says_so(database):
    trade = await _open(TradeJournal(database), stop_order_id="sl-6")
    app = _App(database, orders=[], last_price=103.0)

    notes = await _settle(app, [])

    assert "CIERRE_SIN_REGISTRO" in notes[0]
    row = await _row(database, trade.id)
    assert row.status == "CLOSED"
    assert row.exit_price == pytest.approx(103.0)
    assert row.pnl == pytest.approx(30.0)
    assert "estimado" in (row.exit_reason_note or "")


# --------------------------------------------------------------------------- restraint
async def test_a_live_position_is_left_alone(database):
    trade = await _open(TradeJournal(database))

    notes = await _settle(_App(database), [_position("AAPL", 10.0)])

    assert notes == []
    row = await _row(database, trade.id)
    assert row.status == "OPEN"
    assert row.closed_at is None


async def test_a_bracket_exit_is_found_among_the_legs(database):
    """Alpaca reports a bracket's exits as legs of the entry order."""
    when = dt.datetime.now(UTC) + dt.timedelta(seconds=5)
    tp_leg = _order("tp-leg", side="sell", status="filled", order_type="limit",
                    price=104.0, filled_at=when)
    sl_leg = _order("sl-leg", side="sell", status="canceled", order_type="stop",
                    price=None, filled_at=None)
    # The entry is the parent: filled, but on the side that opened the trade.
    entry = _order("entry-1", side="buy", status="filled", order_type="market",
                   price=100.0, filled_at=when - dt.timedelta(seconds=5),
                   legs=[tp_leg, sl_leg])
    row = await _open(
        TradeJournal(database), stop_order_id="sl-leg", take_profit_order_id="tp-leg"
    )
    app = _App(database, orders=[entry])

    await _settle(app, [])

    row = await _row(database, row.id)
    assert row.status == "CLOSED"
    assert row.exit_reason == "TARGET"
    assert row.exit_price == pytest.approx(104.0)
    assert row.pnl == pytest.approx(40.0)


async def test_the_same_leg_is_not_counted_twice(database):
    when = dt.datetime.now(UTC) + dt.timedelta(seconds=5)
    tp_leg = _order("tp-leg", side="sell", status="filled", order_type="limit",
                    price=104.0, filled_at=when)
    entry = _order("entry-1", side="buy", status="filled", order_type="market",
                   price=100.0, filled_at=when, legs=[tp_leg])
    # The same fill also shows up on its own at the top level.
    row = await _open(
        TradeJournal(database), stop_order_id="sl-x", take_profit_order_id="tp-leg"
    )
    app = _App(database, orders=[entry, tp_leg])

    await _settle(app, [])

    row = await _row(database, row.id)
    assert row.exit_price == pytest.approx(104.0)
    assert row.pnl == pytest.approx(40.0)


async def test_a_trade_in_another_environment_is_untouched(database):
    trade = await _open(TradeJournal(database), environment=TradingEnvironment.REAL)

    notes = await _settle(_App(database), [])

    assert notes == []
    assert (await _row(database, trade.id, TradingEnvironment.REAL)).status == "OPEN"


async def test_one_broken_row_does_not_stop_the_others(database, monkeypatch):
    journal = TradeJournal(database)
    good = await _open(journal, symbol="MSFT", stop_order_id="sl-7")
    await _open(journal, symbol="AAPL", stop_order_id="sl-8")
    app = _App(
        database,
        orders=[
            _order("sl-7", price=98.0,
                   filled_at=dt.datetime.now(UTC) + dt.timedelta(seconds=5))
        ],
    )

    # A row with data this process cannot make sense of must not abort the
    # cycle. It is handed back as a detached stand-in so the stored row, and
    # the session that read it, stay untouched.
    real = TradeRepository.get_open_positions

    async def broken_rows(self, environment):
        stand_ins = []
        for row in await real(self, environment):
            if row.symbol == "AAPL":
                stand_ins.append(
                    SimpleNamespace(
                        id=row.id,
                        symbol=row.symbol,
                        direction=row.direction,
                        entry_price=row.entry_price,
                        qty="not-a-number",
                        stop_price=row.stop_price,
                        stop_order_id=row.stop_order_id,
                        take_profit_order_id=row.take_profit_order_id,
                        filled_at=row.filled_at,
                        opened_at=row.opened_at,
                    )
                )
            else:
                stand_ins.append(row)
        return stand_ins

    monkeypatch.setattr(TradeRepository, "get_open_positions", broken_rows)

    notes = await _settle(app, [])

    assert (await _row(database, good.id)).status == "CLOSED"
    assert any("no se pudo cerrar" in n for n in notes)


async def test_settling_twice_does_not_double_count(database):
    await _open(TradeJournal(database), stop_order_id="sl-8")
    app = _App(
        database,
        orders=[
            _order("sl-8", price=98.0, filled_at=dt.datetime.now(UTC) + dt.timedelta(seconds=5))
        ],
    )

    first = await _settle(app, [])
    async with database.session() as session:
        after_first = await TradeRepository(session).get_all(PAPER)
    second = await _settle(app, [])

    assert len(first) == 1
    assert second == []
    assert len([r for r in after_first if r.status == "CLOSED"]) == 1
