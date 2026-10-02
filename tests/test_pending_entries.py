"""Pending entries: visible in /posiciones, announced when they fill.

An order sent while the market is shut is accepted by Alpaca and fills at the
open. Between those two moments the operator has to be able to see it, and the
instant it fills they have to be told. Both are covered here.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from botalpaca.app import _order_filled
from botalpaca.db import Database, TradeRepository
from botalpaca.db.models import TradeModel
from botalpaca.domain import TradeStatus
from botalpaca.domain.enums import (
    SetupType,
    SignalDirection,
    TradingEnvironment,
)
from botalpaca.journal.service import TradeJournal
from botalpaca.monitoring.service import PositionAlert, PositionMonitor
from botalpaca.notifications import format as fmt
from botalpaca.notifications.service import PENDING_LABEL, render_fill, render_positions

PAPER = TradingEnvironment.PAPER


def _trade(**kw) -> TradeModel:
    base = dict(
        environment=PAPER.value,
        symbol="AAPL",
        direction="long",
        strategy="momentum",
        setup="momentum",
        timeframe="1h",
        status=TradeStatus.OPEN.value,
        qty=5.0,
        entry_price=330.44,
        stop_price=327.14,
        target_price=337.04,
        score=80.0,
        rr=2.0,
        risk_amount=16.5,
        indicators={},
        confluences=[],
        opened_at=dt.datetime.now(dt.UTC),
    )
    base.update(kw)
    return TradeModel(**base)


# ---------------------------------------------------------------- fill detection


def test_an_accepted_order_is_not_treated_as_filled():
    """pending_new / new means Alpaca is holding it for the open."""
    assert _order_filled(None) is False
    assert _order_filled(SimpleNamespace(status="new", filled_avg_price=None)) is False
    assert _order_filled(SimpleNamespace(status="accepted", filled_avg_price=None)) is False
    assert _order_filled(SimpleNamespace(status="pending_new", filled_avg_price=None)) is False


def test_a_real_fill_is_recognised():
    assert _order_filled(SimpleNamespace(status="filled", filled_avg_price=331.0)) is True
    assert _order_filled(SimpleNamespace(status="new", filled_avg_price=331.0)) is True
    assert _order_filled(SimpleNamespace(status="partially_filled", filled_avg_price=None)) is True


# ------------------------------------------------------------------- the card


def test_pending_entry_is_shown_when_there_are_no_positions():
    """An order that has not filled is the only thing on screen; it must appear."""
    text = render_positions([], PAPER, pending=[_trade()])
    assert PENDING_LABEL in text
    assert "AAPL" in text
    assert "330.44" in text
    assert "327.14" in text
    assert "337.04" in text
    assert "Sin posiciones" not in text


def test_pending_entry_does_not_claim_a_pnl():
    """A zero P&L card would read as a flat trade rather than an unfilled order."""
    text = render_positions([], PAPER, pending=[_trade()])
    assert "no realizado" not in text
    assert "P&amp;L" not in text


def test_pending_and_filled_live_side_by_side():
    position = SimpleNamespace(
        symbol="NVDA",
        side="buy",
        qty=3.0,
        avg_entry_price=180.0,
        current_price=185.0,
        unrealized_pl=15.0,
        unrealized_plpc=0.027,
        market_value=555.0,
    )
    text = render_positions([position], PAPER, pending=[_trade()])
    assert PENDING_LABEL in text
    assert "NVDA" in text
    assert "AAPL" in text
    assert "no realizado" in text


def test_empty_screen_still_says_so():
    text = render_positions([], PAPER, pending=[])
    assert "Sin posiciones" in text


def test_fill_notification_names_the_levels():
    alert = PositionAlert(
        symbol="AAPL",
        environment=PAPER,
        previous_score=0.0,
        current_score=0.0,
        kind="fill",
        fill_price=331.20,
        stop_price=327.14,
        target_price=337.04,
        qty=5.0,
    )
    assert alert.is_fill is True
    text = render_fill(alert)
    assert "AAPL" in text
    assert "331.20" in text
    assert "327.14" in text
    assert "337.04" in text


def test_a_deterioration_is_not_mistaken_for_a_fill():
    alert = PositionAlert(
        symbol="AAPL", environment=PAPER, previous_score=82.0, current_score=61.0
    )
    assert alert.is_fill is False


# --------------------------------------------------------------- the repository


async def test_pending_trades_are_excluded_from_filled_ones(database: Database):
    pending = await TradeJournal(database).open_trade(
        environment=PAPER,
        symbol="AAPL",
        direction=SignalDirection.LONG,
        strategy=SetupType.MOMENTUM,
        setup=SetupType.MOMENTUM,
        timeframe="1h",
        qty=5.0,
        entry_price=330.44,
        stop_price=327.14,
        target_price=337.04,
        score=80.0,
        rr=2.0,
        risk_amount=16.5,
        atr=2.0,
        regime="trending",
        sector="tech",
    )
    async with database.session() as session:
        repo = TradeRepository(session)
        rows = await repo.get_pending(PAPER)
        assert [r.id for r in rows] == [pending.id]

        # It still counts as an open trade, which is what keeps it inside the
        # exposure limits while it waits.
        assert await repo.count_open(PAPER) == 1

        marked = await repo.mark_filled(
            pending.id, PAPER, filled_at=dt.datetime.now(dt.UTC), entry_price=331.20
        )
        assert marked is True
        # Marking twice must not re-announce the fill.
        assert await repo.mark_filled(pending.id, PAPER, filled_at=dt.datetime.now(dt.UTC)) is False

    async with database.session() as session:
        repo = TradeRepository(session)
        assert list(await repo.get_pending(PAPER)) == []
        row = await repo.get(pending.id, PAPER)
        assert row.filled_at is not None
        # The recorded entry becomes what Alpaca actually paid.
        assert row.entry_price == 331.20
        assert await repo.count_open(PAPER) == 1


async def test_pending_lookup_is_keyed_on_the_alpaca_order(database: Database):
    row = _trade(entry_order_id="order-abc")
    async with database.session() as session:
        repo = TradeRepository(session)
        session.add(row)
        await session.flush()
        assert (await repo.get_pending_for_order("order-abc", PAPER)).id == row.id
        assert await repo.get_pending_for_order("order-zzz", PAPER) is None
        assert await repo.get_pending_for_order("", PAPER) is None


# --------------------------------------------------------------- the monitor


class _Portfolio:
    def __init__(self, positions):
        self._positions = positions

    async def get_positions(self):
        return list(self._positions)


def _monitor(database: Database, positions, notifications: list[str]) -> PositionMonitor:
    async def notify(text: str) -> None:
        notifications.append(text)

    return PositionMonitor(
        _Portfolio(positions),
        SimpleNamespace(analyze_one=None),
        SimpleNamespace(),
        database,
        environment=PAPER,
        notify=notify,
    )


async def test_monitor_announces_a_pending_entry_that_filled(database: Database):
    journal = TradeJournal(database)
    await journal.open_trade(
        environment=PAPER,
        symbol="AAPL",
        direction=SignalDirection.LONG,
        strategy=SetupType.MOMENTUM,
        setup=SetupType.MOMENTUM,
        timeframe="1h",
        qty=5.0,
        entry_price=330.44,
        stop_price=327.14,
        target_price=337.04,
        score=80.0,
        rr=2.0,
        risk_amount=16.5,
        atr=2.0,
        regime="trending",
        sector="tech",
    )
    # The market opened and Alpaca now reports the position.
    live = SimpleNamespace(
        symbol="AAPL", side="buy", qty=5.0, avg_entry_price=331.20,
        current_price=331.20, unrealized_pl=0.0, unrealized_plpc=0.0,
        market_value=1656.0,
    )
    sent: list[str] = []
    alerts = await _monitor(database, [live], sent).check_fills([live])

    assert [a.kind for a in alerts] == ["fill"]
    assert alerts[0].symbol == "AAPL"
    assert alerts[0].fill_price == 331.20
    assert alerts[0].stop_price == 327.14
    assert alerts[0].target_price == 337.04

    async with database.session() as session:
        assert list(await TradeRepository(session).get_pending(PAPER)) == []


async def test_monitor_does_not_reannounce_the_same_fill(database: Database):
    await TradeJournal(database).open_trade(
        environment=PAPER,
        symbol="AAPL",
        direction=SignalDirection.LONG,
        strategy=SetupType.MOMENTUM,
        setup=SetupType.MOMENTUM,
        timeframe="1h",
        qty=5.0,
        entry_price=330.44,
        stop_price=327.14,
        target_price=337.04,
        score=80.0,
        rr=2.0,
        risk_amount=16.5,
        atr=2.0,
        regime="trending",
        sector="tech",
    )
    live = SimpleNamespace(
        symbol="AAPL", side="buy", qty=5.0, avg_entry_price=331.20,
        current_price=331.20, unrealized_pl=0.0, unrealized_plpc=0.0,
        market_value=1656.0,
    )
    monitor = _monitor(database, [live], [])
    assert len(await monitor.check_fills([live])) == 1
    # Second pass: the trade is already marked filled, so silence.
    assert await monitor.check_fills([live]) == []


async def test_a_still_queued_order_is_not_reported_as_filled(database: Database):
    """The bug this guards: Alpaca creates the position only on the fill.

    With the market shut the order sits in the queue and there is no position.
    Announcing a fill there told the operator the trade was open when it had
    never executed.
    """
    journal = TradeJournal(database)
    await journal.open_trade(
        environment=PAPER,
        symbol="AAPL",
        direction=SignalDirection.LONG,
        strategy=SetupType.MOMENTUM,
        setup=SetupType.MOMENTUM,
        timeframe="1h",
        qty=5.0,
        entry_price=330.44,
        stop_price=327.14,
        target_price=337.04,
        score=0.0,
        rr=2.0,
        risk_amount=16.5,
        atr=None,
        regime="unknown",
        sector=None,
        entry_reason="prueba de entrada pendiente",
        entry_order_id="dc39dd67-461b-48da-a525-e314adc84f0e",
    )

    sent: list[str] = []
    # Zero live positions: the order is queued, not filled.
    assert await _monitor(database, [], sent).check_fills([]) == []
    assert sent == []

    async with database.session() as session:
        pending = list(await TradeRepository(session).get_pending(PAPER))
    assert [p.symbol for p in pending] == ["AAPL"]
    assert pending[0].filled_at is None


async def test_no_notification_when_nothing_is_pending(database: Database):
    live = SimpleNamespace(
        symbol="NVDA", side="buy", qty=3.0, avg_entry_price=180.0,
        current_price=185.0, unrealized_pl=15.0, unrealized_plpc=0.027,
        market_value=555.0,
    )
    assert await _monitor(database, [live], []).check_fills([live]) == []


def test_glyphs_needed_by_the_new_card_exist():
    for name in ("TARGET", "CLOCK", "INFO", "SHIELD", "WARN", "BULL"):
        assert getattr(fmt, name), f"missing glyph {name}"
