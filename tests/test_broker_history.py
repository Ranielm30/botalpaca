"""The broker's own ledger of fills, and the history card built on it.

The journal can only report what the bot wrote down. The broker knows what
really happened, so the numbers of the history come from there and the journal
only adds the bot's own notes.
"""

import datetime as dt
from types import SimpleNamespace

import pytest

from botalpaca.domain import TradingEnvironment
from botalpaca.execution.history import (
    BrokerFill,
    fetch_fills,
    match_operations,
    match_rows,
    parse_activities,
)
from botalpaca.telegram.service import TelegramFacade

PAPER = TradingEnvironment.PAPER
UTC = dt.UTC


def _fill(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    at: dt.datetime,
    order_id: str = "ord-1",
) -> BrokerFill:
    return BrokerFill(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        at=at,
        order_id=order_id,
        order_status="filled",
    )


def _day(day: int) -> dt.datetime:
    return dt.datetime(2024, 1, day, 15, 0, tzinfo=UTC)


# -- the fills as Alpaca tells them ---------------------------------------------------


def test_a_fill_is_read_with_the_time_the_broker_gave_it():
    # Alpaca names the moment of a fill ``transaction_time``; reading any other
    # field loses the trade.
    fills = parse_activities(
        [
            {
                "activity_type": "FILL",
                "id": "act-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "3",
                "price": "101.25",
                "order_id": "ord-9",
                "order_status": "filled",
                "transaction_time": "2024-01-02T15:04:05.123456Z",
            }
        ]
    )

    assert len(fills) == 1
    assert fills[0].symbol == "AAPL"
    assert fills[0].side == "buy"
    assert fills[0].qty == pytest.approx(3.0)
    assert fills[0].price == pytest.approx(101.25)
    assert fills[0].at == dt.datetime(2024, 1, 2, 15, 4, 5, 123456, tzinfo=UTC)
    assert fills[0].order_id == "ord-9"


def test_an_activity_that_is_not_a_fill_is_left_out():
    fills = parse_activities(
        [
            {
                "activity_type": "FILL",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "0",
                "price": "100.00",
                "transaction_time": "2024-01-02T15:00:00Z",
            },
            {
                "activity_type": "CSD",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "5",
                "price": "100.00",
                "transaction_time": "2024-01-02T15:00:00Z",
            },
            {
                "activity_type": "FILL",
                "symbol": "",
                "side": "buy",
                "qty": "5",
                "price": "100.00",
                "transaction_time": "2024-01-02T15:00:00Z",
            },
            {
                "activity_type": "FILL",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "5",
                "price": "100.00",
                "transaction_time": "2024-01-02T15:00:00Z",
            },
            "not-a-dict",
        ]
    )

    assert len(fills) == 1


async def test_the_fills_are_read_page_by_page():
    class _Broker:
        def __init__(self) -> None:
            self.pages = 0

        async def get_activities(self, **kwargs):
            self.pages += 1
            if self.pages == 1:
                return [
                    {
                        "activity_type": "FILL",
                        "id": f"act-{index}",
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": "1",
                        "price": "100.00",
                        "order_id": f"ord-{index}",
                        "order_status": "filled",
                        "transaction_time": "2024-01-02T15:00:00Z",
                    }
                    for index in range(100)
                ]
            return [
                {
                    "activity_type": "FILL",
                    "id": "act-last",
                    "symbol": "AAPL",
                    "side": "sell",
                    "qty": "1",
                    "price": "101.00",
                    "order_id": "ord-last",
                    "order_status": "filled",
                    "transaction_time": "2024-01-02T16:00:00Z",
                }
            ]

    broker = _Broker()
    fills = await fetch_fills(broker)

    assert len(fills) == 101
    assert broker.pages == 2


# -- matching fills into operations ---------------------------------------------------


def test_a_buy_closes_the_short_that_was_open():
    operations, still_open = match_operations(
        [
            _fill("AAPL", "sell", 10, 100.0, _day(1), "entry-1"),
            _fill("AAPL", "buy", 10, 110.0, _day(2), "exit-1"),
        ]
    )

    assert still_open == []
    assert len(operations) == 1
    operation = operations[0]
    assert operation.direction == "SHORT"
    assert operation.qty == pytest.approx(10.0)
    assert operation.entry_price == pytest.approx(100.0)
    assert operation.exit_price == pytest.approx(110.0)
    # A short earns when the price falls: it was sold at 100 and bought at 110.
    assert operation.pnl == pytest.approx(-100.0)
    assert operation.pnl_pct == pytest.approx(-10.0)
    assert operation.entry_order_id == "entry-1"
    assert operation.exit_order_id == "exit-1"


def test_a_sell_closes_the_long_that_was_open():
    operations, still_open = match_operations(
        [
            _fill("AAPL", "buy", 10, 100.0, _day(1), "entry-1"),
            _fill("AAPL", "sell", 10, 110.0, _day(2), "exit-1"),
        ]
    )

    assert still_open == []
    assert len(operations) == 1
    operation = operations[0]
    assert operation.direction == "LONG"
    assert operation.pnl == pytest.approx(100.0)
    assert operation.pnl_pct == pytest.approx(10.0)


def test_a_fill_bigger_than_the_lot_closes_it_and_opens_the_rest():
    operations, still_open = match_operations(
        [
            _fill("AAPL", "buy", 5, 100.0, _day(1), "entry-1"),
            _fill("AAPL", "sell", 8, 110.0, _day(2), "exit-1"),
        ]
    )

    assert len(operations) == 1
    assert operations[0].qty == pytest.approx(5.0)
    assert operations[0].pnl == pytest.approx(50.0)
    # The three actions that were not needed to close the lot are a new short:
    # they were sold without owning them.
    assert len(still_open) == 1
    assert still_open[0].side == "sell"
    assert still_open[0].qty == pytest.approx(3.0)


def test_the_exit_is_shared_out_between_the_lots_it_closes():
    operations, still_open = match_operations(
        [
            _fill("AAPL", "buy", 5, 100.0, _day(1), "entry-1"),
            _fill("AAPL", "buy", 3, 120.0, _day(2), "entry-2"),
            _fill("AAPL", "sell", 8, 130.0, _day(3), "exit-1"),
        ]
    )

    assert still_open == []
    assert len(operations) == 2
    # Each lot is priced with its own entry: the first one was opened at 100.
    assert operations[0].qty == pytest.approx(5.0)
    assert operations[0].pnl == pytest.approx(150.0)
    # The second one was opened at 120.
    assert operations[1].qty == pytest.approx(3.0)
    assert operations[1].pnl == pytest.approx(30.0)


def test_the_symbols_do_not_mix_with_each_other():
    operations, still_open = match_operations(
        [
            _fill("AAPL", "buy", 10, 100.0, _day(1), "entry-1"),
            _fill("MSFT", "sell", 10, 100.0, _day(2), "entry-2"),
        ]
    )

    assert operations == []
    assert {lot.symbol for lot in still_open} == {"AAPL", "MSFT"}


# -- the journal hangs its notes on the operations -------------------------------------


def test_the_journal_notes_are_attached_to_the_operation():
    operations, _ = match_operations(
        [
            _fill("AAPL", "buy", 10, 100.0, _day(1), "entry-1"),
            _fill("AAPL", "sell", 10, 110.0, _day(2), "exit-1"),
        ]
    )
    row = SimpleNamespace(
        symbol="AAPL",
        direction="LONG",
        strategy="momentum",
        setup="breakout",
        exit_reason="target",
        r_multiple=1.5,
        closed_at=_day(2),
    )

    match_rows(operations, [row])

    assert operations[0].registered is True
    assert operations[0].strategy == "momentum"
    assert operations[0].setup == "breakout"
    assert operations[0].exit_reason == "target"
    assert operations[0].r_multiple == pytest.approx(1.5)


def test_an_operation_the_journal_never_recorded_is_still_shown():
    operations, _ = match_operations(
        [
            _fill("AAPL", "buy", 10, 100.0, _day(1), "entry-1"),
            _fill("AAPL", "sell", 10, 110.0, _day(2), "exit-1"),
        ]
    )

    match_rows(operations, [])

    assert operations[0].registered is False
    assert operations[0].exit_reason == ""


def test_a_row_of_another_symbol_is_not_hung_on_the_operation():
    operations, _ = match_operations(
        [
            _fill("AAPL", "buy", 10, 100.0, _day(1), "entry-1"),
            _fill("AAPL", "sell", 10, 110.0, _day(2), "exit-1"),
        ]
    )
    row = SimpleNamespace(
        symbol="MSFT",
        direction="LONG",
        strategy="momentum",
        setup="breakout",
        exit_reason="target",
        r_multiple=1.5,
        closed_at=_day(2),
    )

    match_rows(operations, [row])

    assert operations[0].registered is False


# -- the card the operator reads ------------------------------------------------------


class _Broker:
    """A broker that only knows its fills."""

    def __init__(self, activities: list[dict]) -> None:
        self.activities = list(activities)

    async def get_activities(self, **kwargs):
        return self.activities


class _Result:
    def scalars(self):
        return self

    def all(self):
        return []

    def scalar_one(self):
        return None


class _Session:
    async def execute(self, *args, **kwargs):
        return _Result()


class _Database:
    def session(self):
        database = self

        class _Ctx:
            async def __aenter__(self_inner):
                return _Session()

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


class _App:
    def __init__(self, activities: list[dict]) -> None:
        self.active_environment = PAPER
        self.security = SimpleNamespace(
            allowlist=SimpleNamespace(require=lambda user_id: user_id),
            rate_limiter=SimpleNamespace(check=self._check),
        )
        self.database = _Database()
        self.active = SimpleNamespace(trading_client=_Broker(activities))

    async def _check(self, user_id):
        return None


def _update(user_id: int = 111):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=user_id),
    )


async def test_the_card_shows_the_total_the_broker_reports():
    activities = [
        {
            "activity_type": "FILL",
            "id": "1",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10",
            "price": "100.00",
            "order_id": "entry-1",
            "order_status": "filled",
            "transaction_time": "2024-01-02T15:00:00Z",
        },
        {
            "activity_type": "FILL",
            "id": "2",
            "symbol": "AAPL",
            "side": "sell",
            "qty": "10",
            "price": "110.00",
            "order_id": "exit-1",
            "order_status": "filled",
            "transaction_time": "2024-01-03T15:00:00Z",
        },
    ]
    facade = TelegramFacade(_App(activities))

    result = await facade.historial(_update())

    assert "Total cerrado (Alpaca)" in result.text
    assert "+$100.00" in result.text
    assert "1 operaciones · 0 abiertas" in result.text
    assert "acierto 100%" in result.text
    # Nothing in the journal describes it, and the card says so.
    assert "sin registrar" in result.text


async def test_the_card_keeps_the_journal_as_the_fallback():
    facade = TelegramFacade(_App([]))

    result = await facade.historial(_update())

    assert "Total cerrado" in result.text
    assert "Alpaca" not in result.text


# -- the short sale Alpaca names apart -------------------------------------------------


def test_a_short_sale_is_read_as_the_sale_it_is():
    # Alpaca calls the side of a short entry "sell_short". The shares were
    # borrowed, but for the ledger it is a sale like any other: reading it as
    # anything else leaves every short out of the ledger.
    fills = parse_activities(
        [
            {
                "activity_type": "FILL",
                "id": "act-1",
                "symbol": "SBUX",
                "side": "sell_short",
                "qty": "59",
                "price": "89.98",
                "order_id": "ord-9",
                "order_status": "filled",
                "transaction_time": "2024-01-02T15:04:05.123456Z",
            }
        ]
    )

    assert len(fills) == 1
    assert fills[0].side == "sell"


async def test_a_short_entry_and_its_cover_make_one_short_operation():
    # If the short entry is left out, the cover is read as an entry and the
    # whole account shows a profit it never made.
    fills = parse_activities(
        [
            {
                "activity_type": "FILL",
                "id": "act-1",
                "symbol": "SBUX",
                "side": "sell_short",
                "qty": "10",
                "price": "100.00",
                "order_id": "ord-1",
                "order_status": "filled",
                "transaction_time": "2024-01-02T15:00:00Z",
            },
            {
                "activity_type": "FILL",
                "id": "act-2",
                "symbol": "SBUX",
                "side": "buy",
                "qty": "10",
                "price": "90.00",
                "order_id": "ord-2",
                "order_status": "filled",
                "transaction_time": "2024-01-03T15:00:00Z",
            },
        ]
    )

    operations, still_open = match_operations(fills)

    assert len(operations) == 1
    assert operations[0].direction == "SHORT"
    assert operations[0].qty == pytest.approx(10.0)
    assert operations[0].pnl == pytest.approx(100.0)
    assert still_open == []
