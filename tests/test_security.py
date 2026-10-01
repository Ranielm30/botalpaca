"""Security: allowlist, rate limit, circuit breaker, kill switch, confirmations."""

from __future__ import annotations

import pytest

from botalpaca.domain.enums import TradingEnvironment
from botalpaca.domain.errors import AuthorizationError, KillSwitchError
from botalpaca.security.confirmation import (
    CONFIRMATION_TTL_SECONDS,
    REAL_CONFIRM_TOKEN,
    ConfirmationKind,
    ConfirmationRegistry,
)
from botalpaca.security.guards import (
    ACTIVE_ENV_KEY,
    CIRCUIT_FAILURE_THRESHOLD,
    KILL_SWITCH_KEY,
    AllowList,
    CircuitBreaker,
    CircuitOpenError,
    RateLimiter,
    RateLimitError,
    SecurityLayer,
)

PAPER = TradingEnvironment.PAPER
REAL = TradingEnvironment.REAL


# -- allowlist ----------------------------------------------------------------------
def test_allowlist_accepts_configured_user():
    allow = AllowList.from_ids([111, 222])
    assert allow.is_allowed(111)
    assert allow.is_allowed(222)
    assert not allow.is_allowed(999)


def test_allowlist_rejects_unknown_user():
    allow = AllowList.from_ids([111])
    with pytest.raises(AuthorizationError):
        allow.require(999)


def test_allowlist_treats_none_as_denied():
    allow = AllowList.from_ids([111])
    assert allow.is_allowed(None) is False


def test_empty_allowlist_rejects_everyone():
    allow = AllowList.from_ids([])
    with pytest.raises(AuthorizationError):
        allow.require(1)


def test_allowlist_require_returns_the_user_id():
    assert AllowList.from_ids([111]).require(111) == 111


# -- rate limiter -------------------------------------------------------------------
async def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter(per_minute=3)
    for _ in range(3):
        await limiter.check(1)
    with pytest.raises(RateLimitError):
        await limiter.check(1)


async def test_rate_limiter_is_per_user():
    limiter = RateLimiter(per_minute=2)
    await limiter.check(1)
    await limiter.check(1)
    with pytest.raises(RateLimitError):
        await limiter.check(1)
    await limiter.check(2)


async def test_rate_limiter_reset():
    limiter = RateLimiter(per_minute=1)
    await limiter.check(1)
    with pytest.raises(RateLimitError):
        await limiter.check(1)
    limiter.reset()
    await limiter.check(1)


async def test_rate_limiter_reset_single_user():
    limiter = RateLimiter(per_minute=1)
    await limiter.check(1)
    await limiter.check(2)
    limiter.reset(1)
    await limiter.check(1)
    with pytest.raises(RateLimitError):
        await limiter.check(2)


# -- circuit breaker ---------------------------------------------------------------
async def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60.0)
    await breaker.record_failure()
    await breaker.record_failure()
    assert breaker.is_open is True
    with pytest.raises(CircuitOpenError):
        await breaker.ensure_available()


async def test_circuit_breaker_success_resets():
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60.0)
    await breaker.record_failure()
    await breaker.record_success()
    await breaker.record_failure()
    assert breaker.is_open is False


async def test_circuit_breaker_closes_after_cooldown():
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=0.0)
    await breaker.record_failure()
    # cooldown of 0 means the breaker is already half-open again
    assert breaker.remaining_seconds() == 0.0
    await breaker.ensure_available()


async def test_circuit_breaker_reset_opens_again():
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60.0)
    await breaker.record_failure()
    assert breaker.is_open is True
    await breaker.reset()
    assert breaker.is_open is False
    assert breaker.failures == 0


async def test_default_threshold():
    breaker = CircuitBreaker()
    for _ in range(CIRCUIT_FAILURE_THRESHOLD):
        await breaker.record_failure()
    assert breaker.is_open is True


# -- kill switch -------------------------------------------------------------------
async def test_kill_switch_roundtrip(database):
    security = SecurityLayer(database, AllowList.from_ids([111]))
    assert await security.kill_switch_state() == (False, None)
    await security.engage_kill_switch("manual stop")
    engaged, reason = await security.kill_switch_state()
    assert engaged is True
    assert reason == "manual stop"
    with pytest.raises(KillSwitchError):
        await security.ensure_trading_allowed()
    await security.release_kill_switch()
    assert await security.ensure_trading_allowed() is True


async def test_kill_switch_survives_reinstantiation(database):
    security = SecurityLayer(database, AllowList.from_ids([111]))
    await security.engage_kill_switch("x")
    other = SecurityLayer(database, AllowList.from_ids([111]))
    engaged, _ = await other.kill_switch_state()
    assert engaged is True


async def test_kill_switch_reason_cleared_on_release(database):
    security = SecurityLayer(database, AllowList.from_ids([111]))
    await security.engage_kill_switch("y")
    await security.release_kill_switch()
    assert await security.kill_switch_state() == (False, None)


async def test_persisted_environment_roundtrip(database):
    security = SecurityLayer(database, AllowList.from_ids([111]))
    assert await security.persisted_environment() is None
    await security.persist_environment(REAL)
    assert await security.persisted_environment() == REAL
    assert (await security.persisted_environment()).value == REAL.value


async def test_persisted_environment_key_is_shared(database):
    """The env must be stored under the key the container reads."""
    from botalpaca.db.repositories import AppStateRepository

    security = SecurityLayer(database, AllowList.from_ids([111]))
    await security.persist_environment(PAPER)
    async with database.session() as session:
        assert await AppStateRepository(session).get(ACTIVE_ENV_KEY) == PAPER.value


async def test_broker_call_under_breaker(database):
    security = SecurityLayer(database, AllowList.from_ids([111]))
    for _ in range(CIRCUIT_FAILURE_THRESHOLD):
        await security.circuit_breaker.record_failure()
    with pytest.raises(CircuitOpenError):
        await security.broker_call(lambda: 1)


async def test_broker_call_success_and_failure_are_tracked(database):
    security = SecurityLayer(database, AllowList.from_ids([111]))
    await security.broker_call(lambda: 7)
    assert security.circuit_breaker.failures == 0

    def boom():
        raise RuntimeError("broker down")

    with pytest.raises(RuntimeError):
        await security.broker_call(boom)
    assert security.circuit_breaker.failures == 1


# -- confirmation registry ----------------------------------------------------------
async def test_confirmation_request_and_consume():
    registry = ConfirmationRegistry()
    await registry.request(
        kind=ConfirmationKind.TRADE,
        user_id=1,
        environment=REAL,
        summary="buy AAPL",
        details={"qty": 1},
    )
    pending = await registry.peek(1)
    assert pending is not None
    assert pending.is_real is True
    assert pending.environment == REAL
    assert await registry.consume(user_id=1, kind=ConfirmationKind.TRADE) is not None
    assert await registry.peek(1) is None


async def test_confirmation_kind_mismatch_does_not_consume():
    registry = ConfirmationRegistry()
    await registry.request(
        kind=ConfirmationKind.TRADE, user_id=1, environment=PAPER, summary="s", details={}
    )
    assert await registry.consume(user_id=1, kind=ConfirmationKind.CLOSE) is None
    assert await registry.peek(1) is not None


async def test_confirmation_expires():
    registry = ConfirmationRegistry(ttl_seconds=-1.0)
    await registry.request(
        kind=ConfirmationKind.TRADE, user_id=1, environment=PAPER, summary="s", details={}
    )
    assert await registry.consume(user_id=1) is None


async def test_one_pending_confirmation_per_user():
    registry = ConfirmationRegistry()
    await registry.request(
        kind=ConfirmationKind.TRADE, user_id=1, environment=PAPER, summary="a", details={}
    )
    await registry.request(
        kind=ConfirmationKind.CLOSE, user_id=1, environment=PAPER, summary="b", details={}
    )
    assert (await registry.peek(1)).summary == "b"
    await registry.request(
        kind=ConfirmationKind.TRADE, user_id=2, environment=PAPER, summary="c", details={}
    )
    assert (await registry.peek(1)) is not None
    assert (await registry.peek(2)) is not None


def test_real_confirm_token():
    assert REAL_CONFIRM_TOKEN == "REAL"
    assert CONFIRMATION_TTL_SECONDS > 0


async def test_clear_confirmation():
    registry = ConfirmationRegistry()
    await registry.request(
        kind=ConfirmationKind.TRADE, user_id=1, environment=PAPER, summary="s", details={}
    )
    await registry.clear(1)
    assert await registry.peek(1) is None


async def test_consume_wrong_user_returns_none():
    registry = ConfirmationRegistry()
    await registry.request(
        kind=ConfirmationKind.TRADE, user_id=1, environment=PAPER, summary="s", details={}
    )
    assert await registry.consume(user_id=999) is None
    assert await registry.peek(1) is not None


def test_kill_switch_key_constant():
    assert KILL_SWITCH_KEY == "kill_switch"
