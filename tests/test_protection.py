"""Position protection manager: stops, break-even, trailing, time stop, reconcile."""

from __future__ import annotations

import datetime as dt

import pytest

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
    # Protection is never left naked: the replacement stop is confirmed first.
    assert order[0].startswith("submit:")
    assert any(e.startswith("cancel:s1") for e in order)


async def test_transition_aborts_without_touching_anything_on_broker_failure(database):
    manager, engine, client = await _setup(database, [_stop_order("s1", stop=97.0), _tp_order("t1")])
    client.submit_error = BrokerError("broker down", status_code=500)
    with pytest.raises(ValidationError):
        await manager.enable_trailing_stop(
            environment=PAPER, position=make_position(entry=100.0, current=102.0), trail_percent=2.0
        )
    assert not client.cancelled  # the old bracket legs are untouched


async def test_trailing_stop_uses_atr_when_larger(database):
    manager, engine, client = await _setup(database)
    state = await manager.enable_trailing_stop(
        environment=PAPER, position=make_position(current=102.0), atr=5.0
    )
    assert state.has_trailing is True
    # atr_pct 5/102 ~ 4.9% * 2.5 > default 2.0
    assert state.trail_percent and state.trail_percent > 2.0


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
