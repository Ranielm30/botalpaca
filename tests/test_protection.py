"""Position protection manager: stops, break-even, trailing, time stop, reconcile."""

from __future__ import annotations

import datetime as dt

import pytest
from alpaca.trading.enums import OrderClass

from botalpaca.db.repositories import ProtectionRepository
from botalpaca.domain.enums import ProtectionKind, TimeInForce, TradingEnvironment
from botalpaca.domain.errors import OrderRejectedByBrokerError, ValidationError
from botalpaca.execution.client import BrokerError
from botalpaca.execution.engine import ExecutionEngine
from botalpaca.protection.manager import (
    PositionProtectionManager,
    _blocking_limit_order,
    r_multiple_of,
)

from .conftest import FakeTradingClient, fake_order_raw, make_position

PAPER = TradingEnvironment.PAPER


def _stop_order(order_id: str, symbol: str = "AAPL", stop: float = 97.0):
    return fake_order_raw(
        order_id, symbol=symbol, order_type="stop", side="sell", stop_price=stop, status="new"
    )


def _tp_order(order_id: str, symbol: str = "AAPL", limit: float = 106.0):
    return fake_order_raw(
        order_id, symbol=symbol, order_type="limit", side="sell", limit_price=limit, status="new"
    )


async def _setup(database, orders=None, env=PAPER):
    client = FakeTradingClient(env)
    for order in orders or []:
        client.orders[order.id] = order
    engine = ExecutionEngine(
        client, database, active_environment=env, require_confirmation=False
    )
    manager = PositionProtectionManager(engine, database)
    return manager, engine, client


# -- r multiple ---------------------------------------------------------------------
def test_r_multiple_uses_real_stop_distance():
    pos = make_position(entry=100.0, current=104.0)
    assert r_multiple_of(pos, 98.0) == pytest.approx(2.0)


def test_r_multiple_falls_back_to_percent_without_stop():
    pos = make_position(entry=100.0, current=102.0)
    assert r_multiple_of(pos) == pytest.approx(2.0)


# -- state resolution ----------------------------------------------------------------
async def test_state_reads_live_alpaca_orders(database):
    manager, engine, _ = await _setup(database, [_stop_order("s1", stop=97.0), _tp_order("t1")])
    state = await manager.get_state(PAPER, "AAPL")
    assert state.has_stop is True
    assert state.has_take_profit is True
    assert state.stop_price == pytest.approx(97.0)
    assert state.stop_order_id == "s1"


async def test_state_for_short_uses_buy_exit_side(database):
    """A short position is protected by a BUY stop, not a SELL one."""
    sell_stop = _stop_order("s1", stop=105.0)
    manager, engine, _ = await _setup(database, [sell_stop])
    pos = make_position(qty=-10, entry=100.0, current=98.0)
    # The sell stop is an entry, not this position's protection.
    assert (await manager.state_for_position(PAPER, pos)).has_stop is False

    buy_stop = fake_order_raw(
        "b1", symbol="AAPL", order_type="stop", side="buy", stop_price=105.0, status="new"
    )
    manager, engine, _ = await _setup(database, [buy_stop])
    state = await manager.state_for_position(PAPER, pos)
    assert state.has_stop is True
    assert state.symbol == "AAPL"
    assert state.stop_order_id == "b1"


async def test_state_is_isolated_per_environment(database):
    manager, engine, client = await _setup(database, [_stop_order("s1")])
    assert (await manager.get_state(PAPER, "AAPL")).has_stop is True

    # A REAL client is a different broker account entirely: it has no orders.
    real_manager, _, _ = await _setup(database, [], env=TradingEnvironment.REAL)
    assert (await real_manager.get_state(TradingEnvironment.REAL, "AAPL")).has_stop is False


async def test_state_carries_the_frozen_baseline_from_the_ledger(database):
    """state_for_position publishes the baseline, so get_state has to as well.

    The two doors disagreeing meant R was measured from nothing whenever the
    caller came through this one, even though the ledger already knew the level.
    """
    manager, engine, _ = await _setup(database, [_stop_order("s1", stop=97.0)])
    position = make_position(symbol="AAPL", entry=100.0, current=102.0)

    await manager.reconcile(environment=PAPER, positions=[position], protect_missing=True)
    async with database.session() as session:
        frozen = (await ProtectionRepository(session).get(PAPER, "AAPL")).initial_stop_price
    assert frozen == pytest.approx(97.0)

    state = await manager.get_state(PAPER, "AAPL")
    assert state.initial_stop_price == pytest.approx(frozen)


async def test_state_falls_back_to_the_live_stop_when_the_baseline_is_absent(database):
    """Rows written before the baseline existed still name a real level."""
    manager, engine, _ = await _setup(database, [_stop_order("s1", stop=97.0)])
    async with database.session() as session:
        await ProtectionRepository(session).upsert(
            PAPER, "AAPL", qty=10, entry_price=100.0, stop_order_id="s1", stop_price=97.0
        )

    state = await manager.get_state(PAPER, "AAPL")
    assert state.initial_stop_price == pytest.approx(97.0)


# -- entry protection registration ---------------------------------------------------
async def test_register_entry_protection_persists(database):
    manager, engine, _ = await _setup(database)
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10,
        entry_price=100.0,
        stop_order_id="s1",
        take_profit_order_id="t1",
        order_class="bracket",
    )
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row is not None
    assert row.stop_order_id == "s1"
    assert row.time_stop_at is not None
    assert (row.time_stop_at - dt.datetime.now(dt.UTC).replace(tzinfo=None)).total_seconds() > 0


# -- ensure_stop --------------------------------------------------------------------
async def test_ensure_stop_is_a_noop_when_already_protected(database):
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    state = await manager.ensure_stop(
        environment=PAPER, position=make_position(current=102.0), stop_price=96.0
    )
    assert state.stop_price == pytest.approx(97.0)
    assert not client.submitted


async def test_ensure_stop_creates_a_real_order(database):
    manager, engine, client = await _setup(database)
    state = await manager.ensure_stop(
        environment=PAPER, position=make_position(current=102.0), stop_price=98.0
    )
    assert len(client.submitted) == 1
    assert state.has_stop is True
    assert state.stop_price == pytest.approx(98.0)
    assert state.stop_order_id


async def test_ensure_stop_sanitizes_stop_on_wrong_side(database):
    manager, engine, client = await _setup(database)
    state = await manager.ensure_stop(
        environment=PAPER, position=make_position(current=102.0), stop_price=105.0
    )
    assert state.stop_price < 102.0


async def test_ensure_stop_uses_atr_fallback(database):
    manager, engine, client = await _setup(database)
    state = await manager.ensure_stop(
        environment=PAPER, position=make_position(current=102.0), atr=3.0
    )
    assert state.stop_price == pytest.approx(102.0 - 3.0 * 2.0)


async def test_ensure_stop_never_claims_success_without_order_id(database):
    manager, engine, client = await _setup(database)
    async def _blank_id(request):
        return fake_order_raw("")  # empty id

    client.submit_order = _blank_id
    with pytest.raises(ValidationError):
        await manager.ensure_stop(environment=PAPER, position=make_position(current=102.0), stop_price=98.0)


async def test_a_remembered_target_survives_the_rebuild_of_a_stop(database):
    """Placing an emergency stop must not blank out the target.

    The DAY legs die at the close and take their ids with them, but the levels
    are the record of what the trade was opened to do. ``upsert`` assigns every
    field it is handed, so a stop persisted on its own used to wipe the target
    -- and reconciliation reads exactly that column to put the take-profit
    back. One rebuild was enough to lose it for the rest of the trade.
    """
    manager, _, client = await _setup(database, [], env=PAPER)
    position = make_position(symbol="XOM", entry=178.24, current=178.0)
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="XOM",
        qty=57.0,
        entry_price=178.24,
        take_profit_price=198.64,
        initial_stop_price=167.5,
    )
    # The overnight close killed every leg: nothing is live in Alpaca.
    assert client.orders == {}
    state = await manager.ensure_stop(
        environment=PAPER, position=position, atr=5.0, reason="reconciliación"
    )
    assert state.has_stop is True
    # Read back through the manager: the level is still on our side, so the
    # next reconciliation pass has something to restore.
    assert (await manager.get_state(PAPER, "XOM")).take_profit_price == pytest.approx(198.64)


async def test_the_state_carries_the_target_the_ledger_remembers(database):
    """No live take-profit is not the same as no target.

    A position whose target order expired still has a level recorded against
    it. Reporting ``None`` there made every reader of the state believe the
    position had never had one, which is how a protected trade ended up
    advertised as open-ended.
    """
    manager, _, _ = await _setup(database, [], env=PAPER)
    position = make_position(symbol="DOCU", entry=80.42, current=81.0)
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="DOCU",
        qty=126.0,
        entry_price=80.42,
        take_profit_price=90.87,
    )
    state = await manager.state_for_position(PAPER, position)
    # The order is gone, so this is not a live take-profit...
    assert state.has_take_profit is False
    assert state.take_profit_order_id is None
    # ...but the level is still known.
    assert state.take_profit_price == pytest.approx(90.87)


# -- partial coverage ----------------------------------------------------------------
async def test_an_oco_counts_its_shares_once(database):
    """The take-profit and the stop of an OCO guard the *same* shares.

    They are alternatives, not two lots. Adding both would claim 172 shares
    were protected when only 86 exist, and the top-up built on that number
    would be refused by Alpaca for "insufficient qty available".
    """
    stop = fake_order_raw(
        "sl-1", symbol="WMT", order_type="stop", side="sell",
        stop_price=114.21, qty=86.0, status="held",
    )
    parent = fake_order_raw(
        "tp-1", symbol="WMT", order_type="limit", side="sell",
        limit_price=127.59, qty=86.0, status="new", order_class="oco", legs=[stop],
    )
    manager, _, _ = await _setup(database, [parent, stop])
    state = await manager.state_for_position(
        PAPER, make_position(symbol="WMT", entry=117.23, current=117.5)
    )
    assert state.has_stop is True
    assert state.has_take_profit is True
    assert state.covered_qty == pytest.approx(86.0)


async def test_two_separate_lots_are_counted_apart(database):
    """Two independent stops guard two different lots, and both add up."""
    first = fake_order_raw(
        "sl-1", symbol="WMT", order_type="stop", side="sell",
        stop_price=111.88, qty=112.0, status="new",
    )
    second = fake_order_raw(
        "sl-2", symbol="WMT", order_type="stop", side="sell",
        stop_price=114.21, qty=86.0, status="new",
    )
    manager, _, _ = await _setup(database, [first, second])
    state = await manager.state_for_position(
        PAPER, make_position(symbol="WMT", entry=117.23, current=117.5)
    )
    assert state.covered_qty == pytest.approx(198.0)


async def test_a_top_up_protects_only_the_shares_nothing_else_holds(database):
    """One lot survived the close, the other was opened without protection.

    Asking Alpaca for the whole 198 would be rejected, because the 86 behind
    the live stop are already reserved. Only the 112 left over are free.
    """
    existing = fake_order_raw(
        "sl-1", symbol="WMT", order_type="stop", side="sell",
        stop_price=114.21, qty=86.0, status="new",
    )
    manager, _, client = await _setup(database, [existing])
    position = make_position(symbol="WMT", entry=117.23, current=117.5, qty=198.0)
    await manager.ensure_stop(
        environment=PAPER, position=position, atr=2.0, qty=112.0, reason="reconciliación"
    )
    assert client.submitted, "a top-up order must go out"
    assert client.submitted[-1].qty == pytest.approx(112.0)
    # The lot that was already protected keeps its stop untouched.
    assert "sl-1" not in client.cancelled


async def test_reconciliation_completes_the_shares_a_symbol_left_naked(database):
    """The position is 198, the live stop covers 86: the rest is unprotected.

    SQLite keeps one row per symbol, so a protected lot used to hide the other
    one completely and reconciliation reported the symbol as covered.
    """
    existing = fake_order_raw(
        "sl-1", symbol="WMT", order_type="stop", side="sell",
        stop_price=114.21, qty=86.0, status="new",
    )
    manager, _, client = await _setup(database, [existing])
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="WMT",
        qty=86.0,
        entry_price=117.23,
        stop_order_id="sl-1",
        stop_price=114.21,
        initial_stop_price=111.88,
    )
    position = make_position(symbol="WMT", entry=117.23, current=117.5, qty=198.0)
    notes = await manager.reconcile(
        environment=PAPER, positions=[position], protect_missing=True
    )
    assert client.submitted, "the uncovered lot must get a stop of its own"
    assert client.submitted[-1].qty == pytest.approx(112.0)
    assert any("WMT" in note and "faltaban" in note for note in notes), notes


# -- break-even ----------------------------------------------------------------------
async def test_move_to_break_even_replaces_stop(database):
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    state = await manager.move_to_break_even(environment=PAPER, position=make_position(entry=100.0, current=103.0))
    assert client.replaced
    order_id, request = client.replaced[0]
    assert order_id == "s1"
    assert request.stop_price > 100.0  # above entry, with the buffer
    assert state.break_even_active is True


async def test_move_to_break_even_noop_when_already_profitable_stop(database):
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=102.0)])
    await manager.move_to_break_even(environment=PAPER, position=make_position(entry=100.0, current=103.0))
    assert not client.replaced


async def test_move_to_break_even_creates_stop_when_missing(database):
    manager, engine, client = await _setup(database)
    await manager.move_to_break_even(environment=PAPER, position=make_position(entry=100.0, current=103.0))
    assert client.submitted or client.replaced


# -- trailing stop: the safe transition ---------------------------------------------
async def test_transition_places_provisional_stop_before_cancelling_legs(database):
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0), _tp_order("t1")])
    order: list[str] = []
    original_submit = client.submit_order
    original_cancel = client.cancel_order_by_id

    async def submit(request):
        order.append(f"submit:{request.symbol}")
        return await original_submit(request)

    async def cancel(order_id):
        order.append(f"cancel:{order_id}")
        return await original_cancel(order_id)

    client.submit_order = submit
    client.cancel_order_by_id = cancel

    await manager.enable_trailing_stop(
        environment=PAPER, position=make_position(entry=100.0, current=102.0), trail_percent=2.0
    )
    # Alpaca reserves the shares for every open exit order, so the replacement
    # has to be created only after the old legs are released. Submitting first is
    # rejected outright with "insufficient qty available".
    assert order[0].startswith("cancel:")
    assert any(e.startswith("submit:") for e in order)
    assert any(e == "cancel:s1" for e in order)


async def test_transition_reports_a_bare_position_when_the_replacement_fails(database):
    """The old legs are already cancelled, so failure must be stated, not hidden."""
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0), _tp_order("t1")])
    client.submit_error = BrokerError("broker down", status_code=500)
    with pytest.raises(ValidationError) as excinfo:
        await manager.enable_trailing_stop(
            environment=PAPER, position=make_position(entry=100.0, current=102.0), trail_percent=2.0
        )
    assert "SIN STOP" in str(excinfo.value)


async def test_trailing_stop_uses_atr_when_larger(database):
    manager, engine, client = await _setup(database)
    state = await manager.enable_trailing_stop(
        environment=PAPER, position=make_position(current=102.0), atr=5.0
    )
    assert state.has_trailing is True
    # atr_pct 5/102 ~ 4.9% * 2.5 > default 2.0
    assert state.trail_percent and state.trail_percent > 2.0


async def test_arming_a_trail_does_not_loosen_the_stop_already_in_place(database):
    """The bridge stop must not hand back protection that was already earned.

    Break-even was secured at 100.50 and the price sits at 102. Deriving the
    bridge stop from the price alone gives 102 * (1 - 2% * 1.5) = 98.94, which
    is looser than what the position already had: arming a trailing stop made
    the position *less* protected.
    """
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=100.5)])

    await manager.enable_trailing_stop(
        environment=PAPER, position=make_position(entry=100.0, current=102.0), trail_percent=2.0
    )

    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row.stop_price == 100.5


async def test_arming_a_trail_on_a_short_does_not_loosen_the_stop(database):
    """The mirror rule: for a short the stop only ever moves down."""
    # A short is closed by buying, so its protective stop is a buy stop.
    buy_stop = fake_order_raw(
        "s1", symbol="AAPL", order_type="stop", side="buy", stop_price=103.5, status="new"
    )
    manager, engine, client = await _setup(database, [buy_stop])
    short = make_position(qty=-10.0, entry=100.0, current=102.0)

    await manager.enable_trailing_stop(environment=PAPER, position=short, trail_percent=2.0)

    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row.stop_price == 103.5


async def test_arming_a_trail_releases_the_bridge_stop_before_sending_the_trailing(database):
    """The bridge stop holds every share, so the trailing order never lands.

    Alpaca answers a create made on top of it with "insufficient qty
    available", naming the order that still reserves the position. Waiting the
    release out is the only way through: the reservation is asynchronous, so
    cancelling alone still loses the race.
    """
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=100.5)])
    events: list[tuple[str, str]] = []
    original_submit = client.submit_order
    original_cancel = client.cancel_order_by_id

    async def submit(*args, **kwargs):
        order = await original_submit(*args, **kwargs)
        events.append(("submit", getattr(order, "id", "")))
        return order

    async def cancel(*args, **kwargs):
        events.append(("cancel", args[0] if args else kwargs.get("order_id", "")))
        return await original_cancel(*args, **kwargs)

    client.submit_order = submit
    client.cancel_order_by_id = cancel

    state = await manager.enable_trailing_stop(
        environment=PAPER,
        position=make_position(entry=100.0, current=102.0),
        trail_percent=2.0,
    )

    assert state.has_trailing is True
    # The old legs go first, the bridge stop stands in for them, and only then
    # may the trailing order claim the shares.
    assert events[0] == ("cancel", "s1")
    submits = [entry for entry in events if entry[0] == "submit"]
    assert len(submits) == 2
    bridge_id = submits[0][1]
    assert bridge_id and bridge_id != "s1"
    assert events[-1] == ("submit", submits[1][1])
    assert events.index(("submit", bridge_id)) < events.index(("cancel", bridge_id))
    assert "s1" in client.cancelled
    assert bridge_id in client.cancelled


async def test_a_failed_trailing_arm_puts_the_bridge_stop_back(database):
    """Releasing the bridge stop is only safe if it can be laid down again."""
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=100.5)])
    attempts = {"n": 0}
    laid: list[str] = []
    original_submit = client.submit_order

    async def submit(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 2:
            raise BrokerError("broker down", status_code=500)
        order = await original_submit(*args, **kwargs)
        laid.append(getattr(order, "id", ""))
        return order

    client.submit_order = submit

    with pytest.raises(OrderRejectedByBrokerError, match="broker down"):
        await manager.enable_trailing_stop(
            environment=PAPER,
            position=make_position(entry=100.0, current=102.0),
            trail_percent=2.0,
        )

    # The bridge stood in first, went down for the trailing order, and a real
    # stop was put back once that arm failed: the position is never bare.
    assert "s1" in client.cancelled
    assert len(laid) == 2
    assert laid[0] in client.cancelled


async def test_disable_trailing_stop(database):

    manager, engine, client = await _setup(database)
    trail = fake_order_raw("tr1", symbol="AAPL", order_type="trailing_stop", side="sell", status="new")
    trail.trail_percent = 2.0
    client.orders["tr1"] = trail
    state = await manager.disable_trailing_stop(environment=PAPER, position=make_position(current=102.0))
    assert state.has_trailing is False
    assert "tr1" in client.cancelled


# -- cancel -------------------------------------------------------------------------
async def test_cancel_protection_cancels_stop_and_tp(database):
    manager, engine, client = await _setup(database, [_stop_order("s1"), _tp_order("t1")])
    state = await manager.cancel_protection(environment=PAPER, position=make_position(current=102.0))
    assert state.has_stop is False
    assert state.has_take_profit is False
    assert set(client.cancelled) == {"s1", "t1"}


async def test_cancel_protection_can_target_only_one_kind(database):
    manager, engine, client = await _setup(database, [_stop_order("s1"), _tp_order("t1")])
    await manager.cancel_protection(
        environment=PAPER, position=make_position(current=102.0), kinds=[ProtectionKind.STOP_LOSS]
    )
    assert client.cancelled == ["s1"]


# -- progressive protection ---------------------------------------------------------
async def test_progressive_step_requires_min_r(database):
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    result = await manager.progressive_step(
        environment=PAPER, position=make_position(entry=100.0, current=100.5), step_pct=1.0
    )
    assert result is None
    assert not client.replaced


async def test_progressive_step_moves_stop_up(database):
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    state = await manager.progressive_step(
        environment=PAPER, position=make_position(entry=100.0, current=104.0), step_pct=1.0
    )
    assert client.replaced
    assert state.stop_price > 97.0


# -- time stop -----------------------------------------------------------------------
async def test_time_stop_deadline_and_due(database):
    manager, engine, _ = await _setup(database)
    assert await manager.time_stop_deadline(PAPER, "AAPL") is None
    await manager.register_entry_protection(
        environment=PAPER, symbol="AAPL", qty=10, entry_price=100.0, time_stop_minutes=30
    )
    deadline = await manager.time_stop_deadline(PAPER, "AAPL")
    assert deadline is not None
    aware = deadline if deadline.tzinfo else deadline.replace(tzinfo=dt.UTC)
    assert await manager.is_time_stop_due(
        PAPER, "AAPL", now=aware + dt.timedelta(minutes=1)
    ) is True
    assert await manager.is_time_stop_due(
        PAPER, "AAPL", now=aware - dt.timedelta(minutes=1)
    ) is False


# -- reconciliation / recovery ------------------------------------------------------
async def test_reconcile_creates_emergency_stop_when_missing(database):
    manager, engine, client = await _setup(database)
    actions = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", current=102.0, entry=100.0)],
        open_trade_atr={"AAPL": 3.0},
        protect_missing=True,
    )
    assert client.submitted
    assert any("stop de emergencia creado" in note for note in actions)


async def test_reconcile_uses_percent_fallback_when_no_atr(database):
    """After a restart there is no ATR history; the position still gets a stop."""
    manager, engine, client = await _setup(database)
    await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", current=100.0, entry=100.0)],
        protect_missing=True,
    )
    assert client.submitted
    stop = [o for o in client.submitted if getattr(o, "stop_price", None)]
    assert stop and stop[0].stop_price < 100.0


async def test_reconcile_is_idempotent(database):
    manager, engine, client = await _setup(database)
    positions = [make_position(symbol="AAPL", current=102.0)]
    await manager.reconcile(environment=PAPER, positions=positions, protect_missing=True)
    count = len(client.submitted)
    client.orders = {
        o.id: o for o in client.orders.values()
    }
    for o in client.orders.values():
        if o.order_type == "stop":
            o.symbol = "AAPL"
    await manager.reconcile(environment=PAPER, positions=positions, protect_missing=True)
    assert len(client.submitted) == count


async def test_reconcile_clears_stale_rows(database):
    manager, engine, _ = await _setup(database)
    await manager.register_entry_protection(
        environment=PAPER, symbol="AAPL", qty=10, entry_price=100.0, stop_order_id="s1"
    )
    await manager.reconcile(environment=PAPER, positions=[])
    async with database.session() as session:
        assert await ProtectionRepository(session).get(PAPER, "AAPL") is None


async def test_clearing_protection_releases_the_orders_before_dropping_the_row(database):
    """A manual close used to leave two SIMPLE orders standing for a dead position.

    clear() only deleted the SQLite row, so reconcile's sweep — which iterates
    ``stored - live`` — could never see those orders again. They held the shares
    for ever and refused every new entry on the symbol with "cannot open a long
    buy while a short sell order is open".
    """
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    await manager.register_entry_protection(
        environment=PAPER, symbol="AAPL", qty=10, entry_price=100.0, stop_order_id="s1"
    )

    await manager.clear(PAPER, "AAPL")

    assert "s1" in client.cancelled
    async with database.session() as session:
        assert await ProtectionRepository(session).get(PAPER, "AAPL") is None


async def test_clearing_protection_keeps_the_row_when_a_cancel_fails(database):
    """A failed cancel must not orphan the orders: the row is the only handle on them."""
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    await manager.register_entry_protection(
        environment=PAPER, symbol="AAPL", qty=10, entry_price=100.0, stop_order_id="s1"
    )

    async def refuse(order_id, *args, **kwargs):
        raise BrokerError("broker down", status_code=500)

    original = client.cancel_order_by_id
    client.cancel_order_by_id = refuse
    try:
        await manager.clear(PAPER, "AAPL")
    finally:
        client.cancel_order_by_id = original

    assert "s1" not in client.cancelled
    async with database.session() as session:
        assert await ProtectionRepository(session).get(PAPER, "AAPL") is not None


async def test_reconcile_never_creates_real_protection_for_paper_positions(database):
    manager, engine, client = await _setup(database)
    await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", environment=TradingEnvironment.REAL, current=102.0)],
        protect_missing=True,
    )
    async with database.session() as session:
        rows = await ProtectionRepository(session).all_for(TradingEnvironment.REAL)
    assert not rows


def _as_oco_group(*orders, group="grp-1"):
    """Report orders the way Alpaca does when they really are one group."""
    for order in orders:
        order.order_class = OrderClass.OCO
        order.group_ids = (group,)
    return list(orders)


async def test_reconcile_adopts_a_position_sqlite_never_saw(database):
    """A live position with no row used to be reported and then forgotten.

    The note said it had been adopted while nothing was written, so the
    frozen-risk baseline stayed NULL for the whole life of the trade.
    """
    manager, engine, _ = await _setup(database, [_stop_order("s1", stop=97.0)])
    position = make_position(symbol="AAPL", entry=100.0, current=102.0)

    notes = await manager.reconcile(
        environment=PAPER, positions=[position], protect_missing=True
    )

    assert any("adoptada" in n for n in notes)
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row is not None
    assert row.qty == pytest.approx(position.qty)
    assert row.entry_price == pytest.approx(100.0)
    assert row.stop_order_id == "s1"


async def test_adoption_freezes_the_baseline_and_never_rewrites_it(database):
    manager, engine, _ = await _setup(database, [_stop_order("s1", stop=97.0)])
    position = make_position(symbol="AAPL", entry=100.0, current=102.0)

    await manager.reconcile(environment=PAPER, positions=[position], protect_missing=True)
    async with database.session() as session:
        frozen = (await ProtectionRepository(session).get(PAPER, "AAPL")).initial_stop_price
    assert frozen == pytest.approx(97.0)

    # The stop is ratcheted up. The baseline must not follow it, or R would
    # shrink every time the trade protects itself.
    await manager.move_to_break_even(
        environment=PAPER, position=make_position(entry=100.0, current=103.0)
    )
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row.stop_price > frozen
    assert row.initial_stop_price == pytest.approx(frozen)


async def test_adoption_does_not_reorder_anything_on_alpaca(database):
    """Adoption only records exits Alpaca already accepted."""
    manager, engine, client = await _setup(database)
    position = make_position(symbol="AAPL", entry=100.0, current=102.0)

    await manager.reconcile(
        environment=PAPER, positions=[position], protect_missing=False
    )

    assert client.submitted == []
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row is not None
    assert row.stop_order_id is None



async def test_a_vanished_position_has_its_live_orders_cancelled(database):
    """DOCU closed, the row was deleted, and two SIMPLE exits stayed LIVE behind.

    Deleting the SQLite row alone left the broker orders standing. They were
    a stop and a limit with no position under them, so a touch of either would
    have opened a position nobody asked for. Reconcile has to cancel the exits
    of a position Alpaca no longer reports, not just forget them.
    """
    manager, _, client = await _setup(
        database, [_stop_order("s1", symbol="DOCU", stop=97.0)]
    )
    async with database.session() as session:
        await ProtectionRepository(session).upsert(
            PAPER, "DOCU", qty=129.0, entry_price=72.14, stop_order_id="s1",
            stop_price=97.0,
        )

    notes = await manager.reconcile(
        environment=PAPER, positions=[], protect_missing=True
    )

    assert "s1" in client.cancelled
    assert any("orden" in n and "cancelada" in n for n in notes)
    async with database.session() as session:
        assert await ProtectionRepository(session).get(PAPER, "DOCU") is None


# -- the levels the broker agreed to have to outlive the legs ---------------------------


async def test_an_expired_bracket_stop_is_not_wider_than_the_risk_it_opened(database):
    """A bracket whose DAY legs died at the close must not risk more overnight.

    Alpaca cancels both children of a DAY bracket when the market shuts, so
    reconciliation rebuilds the stop from the closing price. Measured that way,
    QQQ would have gone from a 12.34 risk at the entry to 28.21 -- more than
    double, for doing nothing. The frozen baseline is the only record of the
    risk the caller approved, so it decides the level.
    """
    manager, _, client = await _setup(database, [], env=PAPER)  # no live orders
    position = make_position(symbol="QQQ", entry=753.738, current=747.93)
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="QQQ",
        qty=33.0,
        entry_price=753.738,
        initial_stop_price=741.40,
    )
    state = await manager.ensure_stop(
        environment=PAPER,
        position=position,
        atr=8.76,
        reason="reconciliación",
    )
    # The fallback alone would put the stop at 747.93 - 2*8.76 = 730.41.
    assert state.stop_price == pytest.approx(741.40)
    assert client.submitted  # a real order went out, not just a number


async def test_an_emergency_stop_still_tightens_when_the_baseline_is_too_far(database):
    """The baseline caps the risk upward, it never loosens a stop that is closer.

    A trade that already sits below where its stop was opened gets a re-stop
    tighter than the baseline, and that has to survive: the rule is a floor on
    protection, not a preference for the opening number.
    """
    manager, _, client = await _setup(database, [], env=PAPER)
    position = make_position(symbol="AAPL", entry=100.0, current=104.0)
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        initial_stop_price=97.0,
    )
    state = await manager.ensure_stop(
        environment=PAPER, position=position, atr=1.0, reason="reconciliación"
    )
    # Fallback: 104.0 - 2*1.0 = 102.0, tighter than the frozen 97.0.
    assert state.stop_price == pytest.approx(102.0)


async def test_the_levels_alpaca_accepted_are_written_not_just_the_order_ids(database):
    """The ids alone point at DAY legs that expire, so the prices are recorded too.

    Losing the levels is what left six positions without a target: the bracket
    legs died at the close and the row only remembered which orders they were.
    """
    manager, _, _ = await _setup(database, [], env=PAPER)
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        stop_order_id="s-1",
        take_profit_order_id="t-1",
        stop_price=95.0,
        take_profit_price=110.0,
    )
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row.stop_price == pytest.approx(95.0)
    assert row.take_profit_price == pytest.approx(110.0)


async def test_a_position_that_never_recorded_a_target_can_still_be_restored(database):
    """Rebuilding a dead bracket needs the level, and the level comes along now."""
    manager, _, _ = await _setup(database, [], env=PAPER)
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        stop_order_id="s-1",
        take_profit_order_id="t-1",
        stop_price=95.0,
        take_profit_price=110.0,
    )
    position = make_position(symbol="AAPL", entry=100.0, current=102.0)
    notes = await manager.reconcile(
        environment=PAPER, positions=[position], protect_missing=True
    )
    assert any("objetivo de beneficio restaurado" in n for n in notes)

async def test_a_stop_looser_than_the_risk_it_opened_with_is_tightened(database):
    """The entry risk is the one that was approved, and it decides the stop.

    Every time a DAY bracket dies at the close, the replacement stop is
    derived from the closing price. That is how QQQ ended the day on a
    725.53 stop against a 741.40 baseline: the risk the position was
    opened with doubled without anyone deciding it. ``ensure_stop`` only
    ever *creates* a stop, so nothing came back to compare the two.
    """
    manager, _, _ = await _setup(
        database,
        [_stop_order("s1", symbol="QQQ", stop=725.53),
         _tp_order("t1", symbol="QQQ", limit=779.38)],
        env=PAPER,
    )
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="QQQ",
        qty=33.0,
        entry_price=753.738,
        initial_stop_price=741.40,
        stop_order_id="s1",
        take_profit_order_id="t1",
        stop_price=725.53,
        take_profit_price=779.38,
    )
    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="QQQ", entry=753.738, current=748.25)],
        protect_missing=True,
    )
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "QQQ")
    assert row.stop_price == pytest.approx(741.40)


async def test_a_short_stop_is_measured_the_other_way_round(database):
    """For a short the tighter stop is the *lower* price, mirrored from a long."""
    manager, _, _ = await _setup(
        database,
        [fake_order_raw("s1", symbol="SBUX", order_type="stop", side="buy",
                        stop_price=96.05, status="new"),
         fake_order_raw("t1", symbol="SBUX", order_type="limit", side="buy",
                        limit_price=79.83, status="new")],
        env=PAPER,
    )
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="SBUX",
        qty=59.0,
        entry_price=89.98,
        initial_stop_price=95.05,
        stop_order_id="s1",
        take_profit_order_id="t1",
        stop_price=96.05,
        take_profit_price=79.83,
    )
    await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="SBUX", qty=-59.0, entry=89.98, current=93.21)],
        protect_missing=True,
    )
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "SBUX")
    assert row.stop_price == pytest.approx(95.05)


async def test_a_tighter_stop_is_never_loosened_by_the_comparison(database):
    """The comparison only ever tightens. A stop already closer than the risk
    the trade opened with is left exactly where it is."""
    manager, _, _ = await _setup(
        database,
        [_stop_order("s1", symbol="MSFT", stop=506.93),
         _tp_order("t1", symbol="MSFT", limit=586.32)],
        env=PAPER,
    )
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="MSFT",
        qty=23.0,
        entry_price=518.92,
        initial_stop_price=506.93,
        stop_order_id="s1",
        take_profit_order_id="t1",
        stop_price=506.93,
        take_profit_price=586.32,
    )
    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="MSFT", entry=518.92, current=523.21)],
        protect_missing=True,
    )
    assert not any("ajustado al riesgo" in n for n in notes)
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "MSFT")
    assert row.stop_price == pytest.approx(506.93)


async def test_a_frozen_level_already_past_the_market_is_left_alone(database):
    """A stop beyond the price is refused by Alpaca, and a wide stop that works
    beats an unprotected position."""
    manager, _, _ = await _setup(
        database,
        [_stop_order("s1", symbol="AAPL", stop=97.0),
         _tp_order("t1", symbol="AAPL", limit=110.0)],
        env=PAPER,
    )
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        initial_stop_price=103.0,  # already above the market: not a legal stop
        stop_order_id="s1",
        take_profit_order_id="t1",
        stop_price=97.0,
        take_profit_price=110.0,
    )
    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )
    assert not any("ajustado al riesgo" in n for n in notes)


# -- the target is protection too ---------------------------------------------------
async def test_reconcile_restores_a_missing_take_profit(database):
    """A stop without a target is half a position: it caps the loss and lets a
    winner run forever. Reconciliation used to only ever restore the stop."""
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    # The ledger remembers the level the trade was opened with; Alpaca has no
    # order carrying it.
    await manager._persist(
        PAPER, "AAPL", qty=10.0, entry_price=100.0, take_profit_price=110.0
    )

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    assert any("objetivo de beneficio restaurado" in n for n in notes)
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row is not None
    assert row.take_profit_order_id is not None


async def test_reconcile_says_nothing_when_both_exits_are_already_there(database):
    manager, engine, client = await _setup(
        database,
        _as_oco_group(_stop_order("s1", stop=97.0), _tp_order("t1")),
    )
    await manager._persist(
        PAPER, "AAPL", qty=10.0, entry_price=100.0, take_profit_price=110.0
    )

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    assert notes == []


async def test_two_loose_exits_are_folded_into_one_group(database):
    """A stop and a target standing on their own cannot both fire.

    Alpaca lets one order hold the position, so two independent exits is one
    exit too many. Reconciliation folds them into a single OCO.
    """
    manager, engine, client = await _setup(
        database, [_stop_order("s1", stop=97.0), _tp_order("t1")]
    )
    await manager._persist(
        PAPER, "AAPL", qty=10.0, entry_price=100.0, take_profit_price=110.0
    )

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    assert any("un solo grupo" in n for n in notes), notes
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row.order_class == "oco"
    assert row.take_profit_order_id is not None
    assert row.stop_order_id not in (None, "s1")


async def test_a_target_the_market_has_already_passed_is_refused(database):
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    with pytest.raises(ValidationError, match="lado equivocado"):
        await manager.ensure_take_profit(
            environment=PAPER,
            position=make_position(symbol="AAPL", entry=100.0, current=102.0),
            target_price=98.0,
        )


async def test_a_target_added_over_a_live_stop_lets_that_stop_go_first(database):
    """Alpaca counts a stop's shares as held, so the group cannot land on top
    of it. The stop is released, and the group carries it from then on."""
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    await manager._persist(
        PAPER,
        "AAPL",
        qty=10.0,
        entry_price=100.0,
        stop_order_id="s1",
        stop_price=97.0,
    )

    merged = await manager.ensure_take_profit(
        environment=PAPER,
        position=make_position(symbol="AAPL", entry=100.0, current=102.0),
        target_price=110.0,
    )

    assert "s1" in client.cancelled
    assert merged.has_take_profit and merged.has_stop
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert row.take_profit_order_id is not None
    assert row.stop_order_id != "s1"
    assert row.stop_price == 97.0


async def test_a_group_that_fails_leaves_the_stop_back_in_place(database, monkeypatch):
    """Releasing the stop is the only way through, so it has to come back if
    the group does not land. The position must not be left unprotected."""
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0)])
    await manager._persist(
        PAPER,
        "AAPL",
        qty=10.0,
        entry_price=100.0,
        stop_order_id="s1",
        stop_price=97.0,
        # A target the ledger believes in but the broker never confirmed.
        take_profit_order_id="t-ghost",
        take_profit_price=110.0,
    )
    real = client.submit_order
    seen: list[str] = []

    async def fail_the_group_only(request):
        # Only the OCO is refused; the rollback stop has to go through.
        seen.append(str(getattr(request, "order_class", "")))
        if len(seen) == 1:
            raise RuntimeError("the broker said no")
        return await real(request)

    monkeypatch.setattr(client, "submit_order", fail_the_group_only)

    with pytest.raises(RuntimeError):
        await manager.ensure_take_profit(
            environment=PAPER,
            position=make_position(symbol="AAPL", entry=100.0, current=102.0),
            target_price=110.0,
        )

    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    assert len(seen) == 2, "the group was tried once and the stop went back"
    assert row.stop_order_id not in (None, "s1")
    assert row.stop_price == 97.0
    # Nothing is protecting the upside now, so the ledger must stop saying so.
    assert row.take_profit_order_id is None


# -- a leftover limit on the other side that blocks every exit ----------------------
def _wash_trade_refusal(order_id: str) -> BrokerError:
    """The refusal Alpaca sends while a naked limit lives on the opposite side."""
    return BrokerError(
        'Alpaca rejected the request: {"code":40310000,'
        f'"existing_order_id":"{order_id}",'
        '"message":"potential wash trade detected. use complex orders",'
        '"reject_reason":"opposite side limit order exists. '
        'use complex/limit/stop_limit orders"}',
        status_code=403,
    )


def _refuse_once(client, error: Exception) -> None:
    """Fail the next submission like Alpaca, and only the next one."""
    original = client.submit_order
    pending = [error]

    async def submit(request):
        if pending:
            raise pending.pop()
        return await original(request)

    client.submit_order = submit


async def _skip_share_wait(monkeypatch, manager) -> None:
    """The blocking order is open by design here, so the release wait would spin."""

    async def _instant(*args, **kwargs):
        return True

    monkeypatch.setattr(manager, "_await_shares_free", _instant)


def test_the_blocking_order_id_is_read_off_the_refusal():
    assert (
        _blocking_limit_order(_wash_trade_refusal("4f04b5eb-cbdb-4593-acb7"))
        == "4f04b5eb-cbdb-4593-acb7"
    )
    assert _blocking_limit_order(BrokerError("not enough buying power", status_code=403)) is None


async def test_a_stop_refused_for_a_wash_trade_takes_that_limit_down(database, monkeypatch):
    """The shape PLTR was stuck in: a short with a leftover sell limit."""
    blocker = fake_order_raw(
        "blk-1", symbol="PLTR", order_type="limit", side="sell", limit_price=222.08, status="new"
    )
    manager, _, client = await _setup(database, [blocker])
    _refuse_once(client, _wash_trade_refusal("blk-1"))
    await _skip_share_wait(monkeypatch, manager)

    state = await manager.ensure_stop(
        environment=PAPER,
        position=make_position(symbol="PLTR", qty=-35.0, entry=195.88, current=196.5),
        stop_price=199.0,
    )

    assert state.has_stop is True
    assert state.stop_order_id
    assert client.cancelled == ["blk-1"]
    assert client.orders["blk-1"].status == "canceled"


async def test_a_target_refused_for_a_wash_trade_takes_that_limit_down(database, monkeypatch):
    """The same blocker is what stopped the take-profit from ever landing."""
    blocker = fake_order_raw(
        "blk-3", symbol="PLTR", order_type="limit", side="sell", limit_price=222.08, status="new"
    )
    manager, _, client = await _setup(database, [blocker])
    _refuse_once(client, _wash_trade_refusal("blk-3"))
    await _skip_share_wait(monkeypatch, manager)

    state = await manager.ensure_take_profit(
        environment=PAPER,
        position=make_position(symbol="PLTR", qty=-35.0, entry=195.88, current=196.5),
        target_price=185.0,
    )

    assert state.has_take_profit is True
    assert state.take_profit_order_id
    assert client.cancelled == ["blk-3"]


async def test_a_refusal_that_is_not_a_wash_trade_touches_no_order(database, monkeypatch):
    """Only Alpaca pointing at a specific order earns the right to cancel it."""
    blocker = fake_order_raw(
        "blk-2", symbol="AAPL", order_type="limit", side="buy", limit_price=95.0, status="new"
    )
    manager, _, client = await _setup(database, [blocker])
    _refuse_once(
        client,
        BrokerError(
            'Alpaca rejected the request: {"code":40310001,'
            '"message":"not enough buying power to place this order"}',
            status_code=403,
        ),
    )
    await _skip_share_wait(monkeypatch, manager)

    with pytest.raises(ValidationError):
        await manager.ensure_stop(
            environment=PAPER,
            position=make_position(symbol="AAPL", entry=100.0, current=102.0),
            stop_price=97.0,
        )

    assert client.cancelled == []
    assert client.orders["blk-2"].status == "new"


# -- exits that die with the session --------------------------------------------------


def _day_exit(
    order_id: str,
    *,
    symbol: str = "AAPL",
    order_type: str = "stop",
    side: str = "sell",
    **over: object,
):
    return fake_order_raw(
        order_id,
        symbol=symbol,
        order_type=order_type,
        side=side,
        time_in_force="day",
        **over,
    )


async def test_a_day_exit_is_retired_so_the_pass_rebuilds_it_as_gtc(database):
    # A bracket's legs inherit the time-in-force of the entry that carried them,
    # so both exits arrive session-scoped and the close takes them.
    manager, _, client = await _setup(
        database,
        [
            _day_exit("sl-day", stop_price=97.0, status="new"),
            _day_exit("tp-day", order_type="limit", limit_price=106.0, status="new"),
        ],
    )
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        stop_order_id="sl-day",
        stop_price=97.0,
        take_profit_order_id="tp-day",
        take_profit_price=106.0,
        initial_stop_price=97.0,
    )

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    assert any("day retirada" in note for note in notes), notes
    assert {"sl-day", "tp-day"} <= set(client.cancelled)
    assert client.submitted, "the pass has to put protection back"
    for request in client.submitted:
        assert getattr(request, "time_in_force", None) == TimeInForce.GTC


async def test_a_gtc_exit_is_left_where_it_is(database):
    manager, _, client = await _setup(
        database,
        [
            _stop_order("sl-gtc"),
            _tp_order("tp-gtc"),
        ],
    )
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        stop_order_id="sl-gtc",
        stop_price=97.0,
        take_profit_order_id="tp-gtc",
        take_profit_price=106.0,
        initial_stop_price=97.0,
    )

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    assert not any("day retirada" in note for note in notes), notes


async def test_a_waiting_entry_is_not_a_day_exit_to_retire(database):
    # The entry is also a DAY stop, and on the opposite side of the position:
    # cancelling it would throw away an order that has not opened anything yet.
    manager, _, client = await _setup(
        database,
        [_day_exit("entry-day", side="buy", stop_price=101.0, status="new")],
    )

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    assert not any("day retirada" in note for note in notes), notes
    assert "entry-day" not in client.cancelled


async def test_a_trailing_symbol_keeps_the_exit_it_has(database):
    manager, _, client = await _setup(
        database,
        [
            fake_order_raw(
                "trail-1",
                symbol="AAPL",
                order_type="trailing_stop",
                side="sell",
                status="new",
                time_in_force="day",
            )
        ],
    )

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    assert not any("day retirada" in note for note in notes), notes
    assert "trail-1" not in client.cancelled


# -- a target that outlived its stop --------------------------------------------------


async def test_a_target_without_a_stop_is_rebuilt_as_one_group(database):
    # A live target holds the shares, so a stop that goes out on its own is
    # refused for "insufficient qty available". The stop has to arrive as the
    # other leg of the group, and the group needs a level to be built with.
    manager, _, client = await _setup(database, [_tp_order("tp-1")])

    state = await manager.ensure_stop(
        environment=PAPER,
        position=make_position(symbol="AAPL", entry=100.0, current=100.0),
        atr=2.0,
        reason="prueba",
    )

    request = client.submitted[-1]
    assert request.take_profit is not None
    assert request.stop_loss is not None
    assert "tp-1" in client.cancelled
    assert state.has_stop is True


async def test_a_refused_group_still_leaves_a_stop_behind(database):
    # The group was refused and the target it needed was already released, so
    # the pass has to fall back to a bare stop: a position with no stop at all
    # is the one shape the operator cannot accept.
    manager, _, client = await _setup(database, [_tp_order("tp-1")])
    _refuse_once(client, BrokerError("el broker no admite el grupo", status_code=403))

    state = await manager.ensure_stop(
        environment=PAPER,
        position=make_position(symbol="AAPL", entry=100.0, current=100.0),
        atr=2.0,
        reason="prueba",
    )

    assert state.has_stop is True
    request = client.submitted[-1]
    assert getattr(request, "take_profit", None) is None
    assert getattr(request, "stop_loss", None) is None


async def test_reconciliation_rebuilds_a_target_that_lost_its_stop(database):
    # The shape production ended up in: the target is alive, the stop is gone.
    # The pass has to hand the position back with both levels standing.
    manager, _, client = await _setup(database, [_tp_order("tp-1")])
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        take_profit_order_id="tp-1",
        take_profit_price=106.0,
        initial_stop_price=97.0,
    )

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=100.0)],
        protect_missing=True,
    )

    request = client.submitted[-1]
    assert request.take_profit is not None
    assert request.stop_loss is not None
    assert not any("SIN stop" in note for note in notes)


async def test_a_stop_that_would_cross_the_target_is_pulled_back(database):
    # The group is only buildable when the stop sits on the losing side of the
    # target. A stop that crossed it would leave the position with a target and
    # no stop, which is the shape the operator cannot accept.
    manager, _, client = await _setup(database, [_tp_order("tp-1", limit=110.0)])
    await manager.register_entry_protection(
        environment=PAPER,
        symbol="AAPL",
        qty=10.0,
        entry_price=100.0,
        take_profit_order_id="tp-1",
        take_profit_price=110.0,
        initial_stop_price=150.0,
    )

    await manager.ensure_stop(
        environment=PAPER,
        position=make_position(symbol="AAPL", entry=100.0, current=200.0),
        atr=2.0,
        reason="prueba",
    )

    request = client.submitted[-1]
    assert request.take_profit is not None
    assert request.stop_loss is not None
    assert request.stop_loss.stop_price < request.take_profit.limit_price, notes
