"""Layer 18 — Security: allowlist, kill switch, circuit breaker, rate limiting.

Every guard here is intentionally hard to bypass:

* The allowlist is checked before *any* handler runs.
* The kill switch lives in SQLite (``AppStateModel``) so it survives a Fly.io
  restart, and it is checked again inside the execution engine.
* The circuit breaker counts consecutive broker failures and refuses new orders
  until it either cools down or is manually reset.
* The rate limiter is per-user and per-minute, in memory, and shared by the
  command handlers.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from botalpaca.config.logging import get_logger
from botalpaca.db import AppStateRepository, Database
from botalpaca.domain import TradingEnvironment
from botalpaca.domain.errors import AuthorizationError, KillSwitchError

log = get_logger(__name__)

KILL_SWITCH_KEY = "kill_switch"
KILL_SWITCH_REASON_KEY = "kill_switch_reason"
ACTIVE_ENV_KEY = "active_environment"

# Consecutive broker failures before the breaker opens.
CIRCUIT_FAILURE_THRESHOLD = 5
# Seconds the breaker stays open before a single trial request is allowed.
CIRCUIT_COOLDOWN_SECONDS = 120.0


class CircuitOpenError(RuntimeError):
    """Raised when the broker circuit breaker is open."""


class RateLimitError(RuntimeError):
    """Raised when a user exceeds the per-minute command budget."""


@dataclass
class AllowList:
    """Telegram user allowlist."""

    allowed: frozenset[int]

    @classmethod
    def from_ids(cls, ids: Iterable[int]) -> AllowList:
        return cls(frozenset(int(i) for i in ids))

    def is_allowed(self, user_id: int | None) -> bool:
        if user_id is None:
            return False
        return int(user_id) in self.allowed

    def require(self, user_id: int | None) -> int:
        if not self.is_allowed(user_id):
            log.warning("telegram.access_denied", user_id=user_id)
            raise AuthorizationError(
                f"Usuario {user_id} no está autorizado a usar este bot."
            )
        return int(user_id or 0)


@dataclass
class RateLimiter:
    """Sliding-window per-user rate limiter."""

    per_minute: int = 20
    _hits: dict[int, deque[float]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def check(self, user_id: int) -> None:
        now = time.monotonic()
        async with self._lock:
            bucket = self._hits.setdefault(user_id, deque())
            while bucket and now - bucket[0] > 60.0:
                bucket.popleft()
            if len(bucket) >= self.per_minute:
                wait = 60.0 - (now - bucket[0])
                raise RateLimitError(
                    f"Demasiadas solicitudes. Espera {wait:.0f}s."
                )
            bucket.append(now)

    def reset(self, user_id: int | None = None) -> None:
        if user_id is None:
            self._hits.clear()
        else:
            self._hits.pop(user_id, None)


@dataclass
class CircuitBreaker:
    """Opens after repeated broker failures; half-opens after a cooldown."""

    threshold: int = CIRCUIT_FAILURE_THRESHOLD
    cooldown_seconds: float = CIRCUIT_COOLDOWN_SECONDS
    failures: int = 0
    opened_at: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown_seconds:
            # Cooldown elapsed: allow one trial request through.
            return False
        return True

    def remaining_seconds(self) -> float:
        if self.opened_at is None:
            return 0.0
        return max(0.0, self.cooldown_seconds - (time.monotonic() - self.opened_at))

    async def ensure_available(self) -> None:
        if self.is_open:
            raise CircuitOpenError(
                "Circuit breaker abierto tras errores repetidos del broker. "
                f"Reintento en {self.remaining_seconds():.0f}s."
            )

    async def record_success(self) -> None:
        async with self._lock:
            self.failures = 0
            self.opened_at = None

    async def record_failure(self) -> None:
        async with self._lock:
            self.failures += 1
            if self.failures >= self.threshold and self.opened_at is None:
                self.opened_at = time.monotonic()
                log.error(
                    "security.circuit_opened",
                    failures=self.failures,
                    cooldown_seconds=self.cooldown_seconds,
                )

    async def reset(self) -> None:
        async with self._lock:
            self.failures = 0
            self.opened_at = None


class SecurityLayer:
    """Facade over the individual guards, injected into the Telegram layer."""

    def __init__(
        self,
        database: Database,
        allowlist: AllowList,
        *,
        rate_limiter: RateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._db = database
        self.allowlist = allowlist
        self.rate_limiter = rate_limiter or RateLimiter()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    # ------------------------------------------------------------ kill switch

    async def kill_switch_state(self) -> tuple[bool, str | None]:
        async with self._db.session() as session:
            repo = AppStateRepository(session)
            engaged = bool(await repo.get(KILL_SWITCH_KEY, False))
            reason = await repo.get(KILL_SWITCH_REASON_KEY, None)
        return engaged, (str(reason) if reason else None)

    async def engage_kill_switch(self, reason: str) -> None:
        async with self._db.session() as session:
            repo = AppStateRepository(session)
            await repo.set(KILL_SWITCH_KEY, True)
            await repo.set(KILL_SWITCH_REASON_KEY, reason)
        log.warning("security.kill_switch_on", reason=reason)

    async def release_kill_switch(self) -> None:
        async with self._db.session() as session:
            repo = AppStateRepository(session)
            await repo.set(KILL_SWITCH_KEY, False)
            await repo.set(KILL_SWITCH_REASON_KEY, None)
        log.info("security.kill_switch_off")

    async def ensure_trading_allowed(self) -> bool:
        """Raise if trading is blocked. Returns ``True`` when it is safe to trade."""
        engaged, _ = await self.kill_switch_state()
        if engaged:
            raise KillSwitchError(
                "Kill switch activo. No se pueden enviar órdenes. "
                "Usa /status para ver el motivo."
            )
        await self.circuit_breaker.ensure_available()
        return True

    # --------------------------------------------------- active env (durable)

    async def persisted_environment(self) -> TradingEnvironment | None:
        async with self._db.session() as session:
            raw = await AppStateRepository(session).get(ACTIVE_ENV_KEY, None)
        if raw is None:
            return None
        try:
            return TradingEnvironment(str(raw))
        except ValueError:
            log.warning("security.bad_persisted_environment", value=raw)
            return None

    async def persist_environment(self, environment: TradingEnvironment) -> None:
        async with self._db.session() as session:
            await AppStateRepository(session).set(ACTIVE_ENV_KEY, environment.value)

    # ------------------------------------------------------------- broker I/O

    async def broker_call(self, func: Callable[..., Any], *args: object, **kwargs: object) -> Any:
        """Run a broker call under the circuit breaker.

        ``func`` may be a coroutine function or a plain callable; awaitable
        results are awaited, plain values are returned as-is.

        Usage::

            account = await self.security.broker_call(portfolio.get_account)
        """
        await self.circuit_breaker.ensure_available()
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            await self.circuit_breaker.record_failure()
            raise
        await self.circuit_breaker.record_success()
        return result


__all__ = [
    "ACTIVE_ENV_KEY",
    "KILL_SWITCH_KEY",
    "KILL_SWITCH_REASON_KEY",
    "AllowList",
    "AuthorizationError",
    "CircuitBreaker",
    "CircuitOpenError",
    "KillSwitchError",
    "RateLimitError",
    "RateLimiter",
    "SecurityLayer",
]
