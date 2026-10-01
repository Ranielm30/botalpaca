"""Layer 18 — Explicit confirmation flows.

Two-step confirmations are required for anything that can move money:

* a *trade* confirmation (what will be bought/sold, at what size, with what
  stop/target), and
* an *environment switch* confirmation, which additionally has to acknowledge
  that the destination may be REAL money.

Pending confirmations are held in memory with a short TTL so a stale
confirmation cannot be replayed later, and every confirmation is bound to the
Telegram user that requested it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from botalpaca.domain import TradingEnvironment

CONFIRMATION_TTL_SECONDS = 300.0
# A REAL trade needs a stronger, explicitly typed acknowledgement.
REAL_CONFIRM_TOKEN = "REAL"


class ConfirmationKind(StrEnum):
    TRADE = "trade"
    ENVIRONMENT = "env"
    CLOSE = "close"
    CANCEL = "cancel"
    MODIFY = "modify"
    TRAILING = "trailing"
    BREAK_EVEN = "break_even"
    PROGRESSIVE = "progressive"


@dataclass
class PendingConfirmation:
    kind: ConfirmationKind
    user_id: int
    environment: TradingEnvironment
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) > CONFIRMATION_TTL_SECONDS

    @property
    def is_real(self) -> bool:
        return self.environment.is_real


class ConfirmationRegistry:
    """Holds at most one pending confirmation per user."""

    def __init__(self, *, ttl_seconds: float = CONFIRMATION_TTL_SECONDS) -> None:
        self._pending: dict[int, PendingConfirmation] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def request(
        self,
        *,
        kind: ConfirmationKind,
        user_id: int,
        environment: TradingEnvironment,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> PendingConfirmation:
        pending = PendingConfirmation(
            kind=kind,
            user_id=user_id,
            environment=environment,
            summary=summary,
            details=details or {},
        )
        async with self._lock:
            self._pending[user_id] = pending
        return pending

    async def peek(self, user_id: int) -> PendingConfirmation | None:
        async with self._lock:
            pending = self._pending.get(user_id)
        if pending is None:
            return None
        if (time.monotonic() - pending.created_at) > self._ttl:
            await self.clear(user_id)
            return None
        return pending

    async def consume(
        self, *, user_id: int, kind: ConfirmationKind | None = None
    ) -> PendingConfirmation | None:
        """Pop the pending confirmation, if it matches the expected kind.

        A mismatch is *not* consumed: a stale trade confirmation must not be
        silently consumed by an unrelated button press.
        """
        async with self._lock:
            pending = self._pending.get(user_id)
            if pending is None:
                return None
            if (time.monotonic() - pending.created_at) > self._ttl:
                self._pending.pop(user_id, None)
                return None
            if kind is not None and pending.kind != kind:
                return None
            self._pending.pop(user_id, None)
            return pending

    async def clear(self, user_id: int) -> None:
        async with self._lock:
            self._pending.pop(user_id, None)


__all__ = [
    "CONFIRMATION_TTL_SECONDS",
    "REAL_CONFIRM_TOKEN",
    "ConfirmationKind",
    "ConfirmationRegistry",
    "PendingConfirmation",
]
