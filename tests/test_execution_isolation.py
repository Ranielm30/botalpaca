"""Execution engine: environment barrier, confirmations, idempotency, kill switch."""

from __future__ import annotations

import pytest

from botalpaca.db.repositories import AppStateRepository, OrderAuditRepository
from botalpaca.domain.enums import OrderClass, OrderType, SignalDirection, TradingEnvironment
from botalpaca.domain.errors import (
    ConfirmationRequiredError,
    DuplicateOrderError,
    EnvironmentMismatchError,
    KillSwitchError,
    OrderRejectedByBrokerError,
    RiskRejectedError,
)
from botalpaca.domain.models import TradePlan
from botalpaca.execution.client import BrokerError
from botalpaca.execution.engine import (
    STATUS_FAILED,
    STATUS_SUBMITTED,
    ExecutionEngine,
)
from botalpaca.security.guards import KILL_SWITCH_KEY

from .conftest import FakeTradingClient, fake_order_raw


def _plan(**over) -> TradePlan:
    base = dict(
        symbol="AAPL",
        environment=TradingEnvironment.PAPER,
        direction=SignalDirection.LONG,
        order_type=OrderType.MARKET,
        qty=10.0,
        order_class=OrderClass.BRACKET,
        stop_loss=97.0,
        take_profit=106.0,
    )
    base.update(over)
    return TradePlan(**base)


async def _engine(database, env=TradingEnvironment.PAPER, client=None, **kw):
    client = client or FakeTradingClient(env)
    return ExecutionEngine(client, database, active_environment=env, **kw), client


# -- environment barrier -------------------------------------------------------------
async def test_constructor_rejects_mismatched_client(database):
    client = FakeTradingClient(TradingEnvironment.PAPER)
    with pytest.raises(EnvironmentMismatchError):
        ExecutionEngine(client, database, active_environment=TradingEnvironment.REAL)


async def test_paper_order_never_reaches_real_client(database):
    engine, client = await _engine(database, TradingEnvironment.PAPER)
    plan = _plan(environment=TradingEnvironment.REAL)
    with pytest.raises(EnvironmentMismatchError):
        await engine.submit(plan, risk_approved=True, confirmed=True)
    assert not client.submitted


async def test_real_order_never_reaches_paper_client(database):
    engine, client = await _engine(database, TradingEnvironment.PAPER)
    with pytest.raises(EnvironmentMismatchError):
        await engine.cancel_order("o-1", environment=TradingEnvironment.REAL)
    assert not client.cancelled


async def test_protective_submission_respects_barrier(database):
    engine, _ = await _engine(database, TradingEnvironment.PAPER)
    from botalpaca.execution.builder import OrderBuilder

    request = OrderBuilder.build_protective_limit(
        symbol="AAPL", qty=10, exit_side=SignalDirection.LONG, limit_price=110.0
    )
    with pytest.raises(EnvironmentMismatchError):
        await engine.submit_protective(
            environment=TradingEnvironment.REAL, request=request, symbol="AAPL"
        )


async def test_close_position_respects_barrier(database):
    engine, client = await _engine(database, TradingEnvironment.PAPER)
    with pytest.raises(EnvironmentMismatchError):
        await engine.close_position("AAPL", environment=TradingEnvironment.REAL)
    assert not client.closed


# -- risk and confirmation gates -----------------------------------------------------
async def test_submit_requires_risk_approval(database):
    engine, client = await _engine(database)
    with pytest.raises(RiskRejectedError):
        await engine.submit(_plan(), risk_approved=False, confirmed=True)
    assert not client.submitted


async def test_submit_requires_explicit_confirmation(database):
    engine, client = await _engine(database)
    with pytest.raises(ConfirmationRequiredError):
        await engine.submit(_plan(), risk_approved=True, confirmed=False)
    assert not client.submitted


async def test_guard_order_is_barrier_then_risk_then_confirmation(database):
    engine, _ = await _engine(database)
    # Environment mismatch wins over the missing risk approval.
    with pytest.raises(EnvironmentMismatchError):
        await engine.submit(_plan(environment=TradingEnvironment.REAL), risk_approved=False, confirmed=False)


# -- happy path ----------------------------------------------------------------------
async def test_submit_happy_path_records_audit(database):
    engine, client = await _engine(database)
    result = await engine.submit(_plan(), risk_approved=True, confirmed=True, confirmed_by=111)
    assert bool(result) is True
    assert result.environment is TradingEnvironment.PAPER
    assert len(client.submitted) == 1
    assert result.order is not None
    assert result.order.environment is TradingEnvironment.PAPER
    async with database.session() as session:
        rows = await OrderAuditRepository(session).recent(TradingEnvironment.PAPER)
    assert rows
    assert any(r.status == STATUS_SUBMITTED for r in rows)


async def test_submit_never_claims_success_without_broker_id(database):
    engine, client = await _engine(database)
    async def _submit(request):
        return fake_order_raw("real-id")

    client.submit_order = _submit
    result = await engine.submit(_plan(), risk_approved=True, confirmed=True)
    assert result.order.id == "real-id"
    assert result.audit_id


async def test_broker_rejection_is_recorded_and_raised(database):
    engine, client = await _engine(database)
    client.submit_error = BrokerError("insufficient buying power", status_code=403)
    with pytest.raises(OrderRejectedByBrokerError):
        await engine.submit(_plan(), risk_approved=True, confirmed=True)
    async with database.session() as session:
        rows = await OrderAuditRepository(session).recent(TradingEnvironment.PAPER)
    assert any(r.status == STATUS_FAILED for r in rows)


# -- idempotency / duplicate protection ----------------------------------------------
async def test_duplicate_submission_is_blocked(database):
    engine, client = await _engine(database)
    await engine.submit(_plan(), risk_approved=True, confirmed=True)
    with pytest.raises(DuplicateOrderError):
        await engine.submit(_plan(), risk_approved=True, confirmed=True)
    assert len(client.submitted) == 1


async def test_duplicate_key_is_explicit(database):
    engine, client = await _engine(database)
    await engine.submit(_plan(), risk_approved=True, confirmed=True, idempotency_key="fixed-key")
    with pytest.raises(DuplicateOrderError):
        await engine.submit(
            _plan(qty=11.0), risk_approved=True, confirmed=True, idempotency_key="fixed-key"
        )
    assert len(client.submitted) == 1


async def test_default_idempotency_key_differs_by_environment(database):
    paper_key = ExecutionEngine.default_idempotency_key(_plan())
    real_key = ExecutionEngine.default_idempotency_key(
        _plan(environment=TradingEnvironment.REAL)
    )
    assert paper_key != real_key
    assert len(paper_key) <= 32


# -- kill switch ---------------------------------------------------------------------
async def test_kill_switch_blocks_submission(database):
    engine, client = await _engine(database)
    async with database.session() as session:
        await AppStateRepository(session).set(KILL_SWITCH_KEY, True)
    with pytest.raises(KillSwitchError):
        await engine.submit(_plan(), risk_approved=True, confirmed=True)
    assert not client.submitted


async def test_kill_switch_release_restores_trading(database):
    engine, client = await _engine(database)
    async with database.session() as session:
        repo = AppStateRepository(session)
        await repo.set(KILL_SWITCH_KEY, True)
        await repo.set(KILL_SWITCH_KEY, False)
    await engine.submit(_plan(), risk_approved=True, confirmed=True)
    assert len(client.submitted) == 1


# -- order lifecycle -----------------------------------------------------------------
async def test_cancel_order(database):
    engine, client = await _engine(database)
    client.orders["o-1"] = fake_order_raw("o-1")
    assert await engine.cancel_order("o-1", environment=TradingEnvironment.PAPER) is True
    assert "o-1" in client.cancelled


async def test_cancel_orders_for_symbol(database):
    engine, client = await _engine(database)
    client.orders["a"] = fake_order_raw("a", symbol="AAPL")
    client.orders["b"] = fake_order_raw("b", symbol="MSFT")
    assert await engine.cancel_orders_for_symbol("AAPL", environment=TradingEnvironment.PAPER) is True
    assert client.cancelled == ["a"]


async def test_replace_order(database):
    from botalpaca.execution.builder import OrderBuilder

    engine, client = await _engine(database)
    client.orders["o-1"] = fake_order_raw("o-1")
    state = await engine.replace_order(
        "o-1", OrderBuilder.build_replace(stop_price=99.0), environment=TradingEnvironment.PAPER
    )
    assert client.orders["o-1"].stop_price == 99.0
    assert state.id == "o-1"


async def test_close_position_requires_confirmation_when_configured(database):
    engine, client = await _engine(database, require_confirmation=True)
    with pytest.raises(ConfirmationRequiredError):
        await engine.close_position("AAPL", environment=TradingEnvironment.PAPER, confirmed=False)
    assert not client.closed


async def test_close_position_closes_everything_by_default(database):
    engine, client = await _engine(database)
    result = await engine.close_position("AAPL", environment=TradingEnvironment.PAPER)
    assert client.closed and client.closed[0][0] == "AAPL"
    assert bool(result) is True


async def test_close_position_accepts_partial_qty(database):
    engine, client = await _engine(database)
    await engine.close_position("AAPL", environment=TradingEnvironment.PAPER, qty="1")
    request = client.closed_calls[0]["request"]
    assert request.qty == "1"
    assert request.percentage is None


async def test_close_position_all_when_nothing_given(database):
    engine, client = await _engine(database)
    await engine.close_position("AAPL", environment=TradingEnvironment.PAPER)
    request = client.closed_calls[0]["request"]
    assert request.percentage == "all"
    assert request.qty is None


async def test_get_open_orders_exposes_nested_legs(database):
    engine, client = await _engine(database)
    leg = fake_order_raw("leg-1", symbol="AAPL", order_type="stop", side="sell", order_class="bracket")
    parent = fake_order_raw("parent-1", order_class="bracket", legs=[leg])
    client.orders["parent-1"] = parent
    orders = await engine.get_open_orders(nested=True)
    assert orders and orders[0].id == "parent-1"


async def test_audits_are_isolated_per_environment(database):
    paper_engine, _ = await _engine(database, TradingEnvironment.PAPER)
    real_engine, _ = await _engine(database, TradingEnvironment.REAL)
    await paper_engine.submit(_plan(), risk_approved=True, confirmed=True)
    async with database.session() as session:
        repo = OrderAuditRepository(session)
        assert len(await repo.recent(TradingEnvironment.PAPER)) > 0
        assert len(await repo.recent(TradingEnvironment.REAL)) == 0
