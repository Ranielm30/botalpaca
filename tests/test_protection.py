"""Position protection manager: stops, break-even, trailing, time stop, reconcile."""

from __future__ import annotations

import datetime as dt

import pytest
from alpaca.trading.enums import OrderClass

from botalpaca.db.repositories import ProtectionRepository
from botalpaca.domain.enums import ProtectionKind, TradingEnvironment
from botalpaca.domain.errors import ValidationError
from botalpaca.execution.client import BrokerError
from botalpaca.execution.engine import ExecutionEngine
from botalpaca.protection.manager import PositionProtectionManager, r_multiple_of

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
