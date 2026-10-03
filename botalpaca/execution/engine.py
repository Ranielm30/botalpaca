"""Execution Engine: the only path from a decision to an Alpaca order.

Three guarantees are enforced here, in this order, before any network call:

1. **Environment barrier.** The plan's environment must be exactly the active
   environment. There is no automatic PAPER to REAL escalation, ever.
2. **Kill switch.** A tripped kill switch blocks every submit.
3. **Idempotency / duplicate protection.** An idempotency key is required and a
   previously submitted key short-circuits to the recorded result instead of
   placing a second order.

Every attempt -- submitted, rejected or failed -- is written to the immutable
order audit table, and a claim on the idempotency key is inserted *before* the
broker call so a crash mid-flight cannot produce a duplicate on retry.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Any

from botalpaca.config import get_logger
from botalpaca.db import Database, OrderAuditModel, OrderAuditRepository
from botalpaca.domain import (
    ConfirmationRequiredError,
    DuplicateOrderError,
    EnvironmentMismatchError,
    KillSwitchError,
    OrderRejectedByBrokerError,
    OrderState,
    RiskRejectedError,
    TradePlan,
    TradingEnvironment,
)
from botalpaca.execution.builder import OrderBuilder
from botalpaca.execution.client import (
    AlpacaTradingClient,
    BrokerError,
    order_status_value,
)
from botalpaca.execution.mapping import to_order_state

# Alpaca reports order status as an enum whose repr is ``OrderStatus.NEW``. Only
# these states are final; everything else may still fill, so it must stay visible
# when looking for protection.
_TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "replaced", "rejected"})

logger = get_logger(__name__)

__all__ = ["ExecutionEngine", "SubmissionResult", "KILL_SWITCH_KEY"]

KILL_SWITCH_KEY = "kill_switch"

# Audited actions
ACTION_SUBMIT = "SUBMIT"
ACTION_CANCEL = "CANCEL"
ACTION_REPLACE = "REPLACE"
ACTION_CLOSE = "CLOSE"
ACTION_PROTECT = "PROTECT"

STATUS_PENDING = "PENDING"
STATUS_SUBMITTED = "SUBMITTED"
STATUS_FAILED = "FAILED"
STATUS_REJECTED = "REJECTED"


class SubmissionResult:
    """Outcome of an order submission, including the audit record id."""

    __slots__ = ("order", "audit_id", "client_order_id", "environment", "created")

    def __init__(
        self,
        order: OrderState | None,
        audit_id: int | None,
        client_order_id: str,
        environment: TradingEnvironment,
        *,
        created: bool = True,
    ) -> None:
        self.order = order
        self.audit_id = audit_id
        self.client_order_id = client_order_id
        self.environment = environment
        self.created = created

    def __bool__(self) -> bool:
        return self.order is not None

    @property
    def status(self) -> str:
        return self.order.status if self.order else STATUS_FAILED


class ExecutionEngine:
    """Submits, replaces, and cancels orders against one environment's client.

    The engine is instantiated with the client for a specific environment and
    re-created when ``/modo`` switches environments, so it is structurally
    incapable of reaching the other account.
    """

    def __init__(
        self,
        client: AlpacaTradingClient,
        database: Database,
        *,
        active_environment: TradingEnvironment,
        require_confirmation: bool = True,
    ) -> None:
        if client.environment_name is not active_environment:
            raise EnvironmentMismatchError(
                "Execution client environment does not match the active environment."
            )
        self._client = client
        self._db = database
        self.active_environment = active_environment
        self.require_confirmation = require_confirmation

    # -- guards --------------------------------------------------------------

    def _assert_environment(self, environment: TradingEnvironment) -> None:
        """The security barrier. Not negotiable, not configurable."""
        if environment is not self.active_environment:
            raise EnvironmentMismatchError(
                f"Order for {environment.value} rejected: the active environment is "
                f"{self.active_environment.value}. Orders are never routed across "
                f"environments."
            )

    async def _kill_switch_engaged(self, session: Any) -> bool:
        from botalpaca.db import AppStateRepository

        return bool(await AppStateRepository(session).get(KILL_SWITCH_KEY, False))

    # -- submission ----------------------------------------------------------

    async def submit(
        self,
        plan: TradePlan,
        *,
        risk_approved: bool = False,
        confirmed: bool = False,
        confirmed_by: int | None = None,
        idempotency_key: str | None = None,
        risk_score: float = 0.0,
        extra_audit: dict[str, Any] | None = None,
    ) -> SubmissionResult:
        """Submit an entry order for ``plan``.

        Preconditions, all checked before any Alpaca call:

        * ``plan.environment`` is the active environment,
        * the kill switch is off,
        * ``risk_approved`` is true (the Risk Engine must have approved),
        * ``confirmed`` is true (Telegram confirmation gate), and for REAL the
          caller must have gone through the real-money confirmation card,
        * the idempotency key has not already been submitted.
        """
        self._assert_environment(plan.environment)

        if not risk_approved:
            raise RiskRejectedError(
                ["The Risk Engine has not approved this order."],
            )
        if self.require_confirmation and not confirmed:
            raise ConfirmationRequiredError(
                "This order requires explicit confirmation before it is submitted."
            )

        key = idempotency_key or self.default_idempotency_key(plan)

        async with self._db.session() as session:
            audit_repo = OrderAuditRepository(session)
            if await audit_repo.idempotency_key_exists(plan.environment, key):
                raise DuplicateOrderError(
                    f"An order with idempotency key {key} was already submitted in "
                    f"{plan.environment.value}. Duplicate submissions are refused."
                )
            if await self._kill_switch_engaged(session):
                raise KillSwitchError(
                    "Kill switch is active. All order submission is blocked."
                )

            client_order_id = self._client_order_id(plan, key)
            claim = await audit_repo.log(
                OrderAuditModel(
                    environment=plan.environment.value,
                    action=ACTION_SUBMIT,
                    symbol=plan.symbol.upper(),
                    order_type=plan.order_type.value,
                    order_class=plan.order_class.value,
                    qty=plan.qty or None,
                    notional=plan.notional,
                    side=plan.direction.entry_side.value,
                    limit_price=plan.limit_price,
                    stop_price=plan.stop_loss,
                    status=STATUS_PENDING,
                    client_order_id=client_order_id,
                    idempotency_key=key,
                    risk_score=risk_score,
                    confirmed_by=confirmed_by,
                    request=_plan_payload(plan, extra_audit),
                )
            )
            audit_id = claim.id

        # Build outside the transaction: a shape Alpaca cannot accept must not
        # leave a pending audit row behind.
        request = OrderBuilder.build_entry(plan, client_order_id=client_order_id)

        try:
            raw_order = await self._client.submit_order(request)
        except BrokerError as exc:
            await self._record_failure(audit_id, plan.environment, str(exc))
            raise OrderRejectedByBrokerError(str(exc)) from exc

        order = to_order_state(raw_order, plan.environment)
        async with self._db.session() as session:
            repo = OrderAuditRepository(session)
            row = await repo.find_by_idempotency_key(plan.environment, key)
            if row is not None:
                row.status = STATUS_SUBMITTED
                row.broker_order_id = order.id
                row.response = _jsonable(order)
            await repo.log(
                OrderAuditModel(
                    environment=plan.environment.value,
                    action=ACTION_SUBMIT,
                    symbol=plan.symbol.upper(),
                    order_type=order.order_type.value,
                    order_class=order.order_class.value,
                    qty=order.qty,
                    side=order.side.value,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    status=STATUS_SUBMITTED,
                    client_order_id=order.client_order_id or client_order_id,
                    broker_order_id=order.id,
                    idempotency_key=None,
                    risk_score=risk_score,
                    confirmed_by=confirmed_by,
                    response=_jsonable(order),
                )
            )
        logger.info(
            "execution.submitted",
            environment=plan.environment.value,
            symbol=plan.symbol.upper(),
            order_id=order.id,
            status=order.status,
            client_order_id=client_order_id,
        )
        return SubmissionResult(order, audit_id, client_order_id, plan.environment)

    async def submit_protective(
        self,
        *,
        environment: TradingEnvironment,
        request: Any,
        symbol: str,
        action: str = ACTION_PROTECT,
        idempotency_key: str | None = None,
    ) -> SubmissionResult:
        """Submit a standalone protective order (stop, target, trailing stop)."""
        self._assert_environment(environment)
        key = idempotency_key or uuid.uuid4().hex

        async with self._db.session() as session:
            repo = OrderAuditRepository(session)
            if await repo.idempotency_key_exists(environment, key):
                raise DuplicateOrderError(
                    f"Protective order {key} already submitted in {environment.value}."
                )
            if await self._kill_switch_engaged(session):
                raise KillSwitchError("Kill switch is active. Blocking protective order.")
            await repo.log(
                OrderAuditModel(
                    environment=environment.value,
                    action=action,
                    symbol=symbol.upper(),
                    status=STATUS_PENDING,
                    client_order_id=key,
                    idempotency_key=key,
                    request=_describe_request(request),
                )
            )

        try:
            raw_order = await self._client.submit_order(request)
        except BrokerError as exc:
            await self._record_failure(None, environment, str(exc), symbol=symbol)
            raise OrderRejectedByBrokerError(str(exc)) from exc

        order = to_order_state(raw_order, environment)
        async with self._db.session() as session:
            repo = OrderAuditRepository(session)
            row = await repo.find_by_idempotency_key(environment, key)
            if row is not None:
                row.status = STATUS_SUBMITTED
                row.broker_order_id = order.id
                row.response = _jsonable(order)
        return SubmissionResult(order, None, key, environment)

    # -- lifecycle -----------------------------------------------------------

    async def cancel_order(self, order_id: str, *, environment: TradingEnvironment) -> bool:
        self._assert_environment(environment)
        await self._client.cancel_order_by_id(order_id)
        await self._audit_simple(ACTION_CANCEL, environment, order_id=order_id, status="CANCELLED")
        logger.info("execution.cancelled", environment=environment.value, order_id=order_id)
        return True

    async def cancel_orders_for_symbol(
        self, symbol: str, *, environment: TradingEnvironment
    ) -> bool:
        self._assert_environment(environment)
        await self._client.cancel_orders(symbol=symbol.upper())
        await self._audit_simple(
            ACTION_CANCEL, environment, symbol=symbol.upper(), status="CANCELLED"
        )
        return True

    async def replace_order(
        self,
        order_id: str,
        request: Any,
        *,
        environment: TradingEnvironment,
    ) -> OrderState:
        self._assert_environment(environment)
        raw_order = await self._client.replace_order_by_id(order_id, request)
        order = to_order_state(raw_order, environment)
        await self._audit_simple(
            ACTION_REPLACE,
            environment,
            order_id=order_id,
            symbol=order.symbol,
            status="REPLACED",
            response=_jsonable(order),
        )
        return order

    async def close_position(
        self,
        symbol: str,
        *,
        environment: TradingEnvironment,
        qty: str | None = None,
        percentage: str | None = None,
        confirmed: bool = True,
    ) -> SubmissionResult:
        """Close a position. ``percentage``/``qty`` use Alpaca's string format."""
        self._assert_environment(environment)
        if self.require_confirmation and not confirmed:
            raise ConfirmationRequiredError("Closing a position requires confirmation.")
        if qty is None and percentage is None:
            # Alpaca rejects a ClosePositionRequest with neither qty nor
            # percentage. The whole position is 100%, not the string "all" --
            # the API answers "percentage must be between 0 and 100" and the
            # position stays open.
            percentage = "100"

        # Alpaca reserves the shares for every open exit order. A protected
        # position therefore cannot be closed: the protective stop holds exactly
        # the quantity being sold and the close comes back "insufficient qty
        # available ... held_for_orders". Release the exits first, then close --
        # otherwise every timed close fails and the position stays open with the
        # very protection meant to end it.
        await self.cancel_orders_for_symbol(symbol.upper(), environment=environment)

        # The client builds the ClosePositionRequest itself; passing one in used
        # to raise TypeError and left the position open.
        try:
            raw_order = await self._client.close_position(
                symbol.upper(), qty=qty, percentage=percentage
            )
        except BrokerError as exc:
            await self._record_failure(None, environment, str(exc), symbol=symbol)
            raise OrderRejectedByBrokerError(str(exc)) from exc
        order = to_order_state(raw_order, environment)
        await self._audit_simple(
            ACTION_CLOSE,
            environment,
            symbol=symbol.upper(),
            order_id=order.id,
            status="SUBMITTED",
            response=_jsonable(order),
        )
        return SubmissionResult(order, None, order.id or uuid.uuid4().hex, environment)

    # -- queries -------------------------------------------------------------

    async def get_open_orders(self, *, nested: bool = True) -> list[OrderState]:
        """Open orders for the active environment.

        ``nested=True`` is what exposes bracket/OCO child legs, which the
        protection manager needs.
        """
        from alpaca.trading.enums import QueryOrderStatus

        raw_orders = await self._client.get_orders(
            status=QueryOrderStatus.OPEN, limit=500, nested=nested
        )
        return [to_order_state(o, self.active_environment) for o in raw_orders]

    async def get_open_orders_for_symbol(self, symbol: str) -> list[OrderState]:
        orders = await self.get_open_orders(nested=True)
        target = symbol.upper()
        return [o for o in orders if o.symbol == target]

    async def get_live_orders_for_symbol(self, symbol: str) -> list[OrderState]:
        """Orders for one symbol that can still act, ``HELD`` ones included.

        A bracket child left in ``HELD`` reserves the shares without protecting
        anything, and it is invisible to an OPEN query. Protection must be able
        to find and clear it.
        """
        orders = await self.get_live_orders(symbols=[symbol.upper()])
        target = symbol.upper()
        return [o for o in orders if o.symbol == target]

    async def get_live_orders(self, *, symbols: Sequence[str] | None = None) -> list[OrderState]:
        """Orders that can still act, including the ones an OPEN query hides.

        ``QueryOrderStatus`` only offers OPEN, CLOSED and ALL, so a bracket child
        left in ``HELD`` is invisible to an OPEN query -- yet it still reserves the
        shares. ``ALL`` is queried and terminal states are dropped here.

        ``nested`` must stay off. Verified against the PAPER API: the same query
        returns 38 rows including the HELD order unnested, and 18 rows with zero
        HELD when nested. Nesting folds children into the parent and drops the
        ones that are not open, which is exactly the order protection has to
        find. The children still arrive on the parent's ``legs``, so leaving
        ``nested`` off costs nothing.
        """
        from alpaca.trading.enums import QueryOrderStatus

        raw_orders = await self._client.get_orders(
            status=QueryOrderStatus.ALL, limit=500, symbols=list(symbols) if symbols else None
        )
        return [
            to_order_state(o, self.active_environment)
            for o in raw_orders
            if order_status_value(o) not in _TERMINAL_STATUSES
        ]

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def default_idempotency_key(plan: TradePlan) -> str:
        """Deterministic key for an entry so retries cannot double-fill."""
        material = "|".join(
            [
                plan.environment.value,
                plan.symbol.upper(),
                plan.direction.value,
                plan.order_type.value,
                f"{plan.qty:.6f}" if plan.qty else f"n{plan.notional:.2f}",
                f"{plan.limit_price}" if plan.limit_price else "-",
                f"{plan.stop_loss}" if plan.stop_loss else "-",
                f"{plan.take_profit}" if plan.take_profit else "-",
                plan.opportunity_fingerprint or "-",
            ]
        )
        import hashlib

        return hashlib.sha1(material.encode("utf-8")).hexdigest()[:32]  # noqa: S324

    def _client_order_id(self, plan: TradePlan, key: str) -> str:
        """``client_order_id`` <= 48 chars, alphanumeric only."""
        prefix = f"ba{plan.environment.value[:1]}{plan.order_class.value[:1]}"
        return f"{prefix}{key[:44]}"

    async def _record_failure(
        self,
        audit_id: int | None,
        environment: TradingEnvironment,
        error: str,
        *,
        symbol: str | None = None,
    ) -> None:
        if audit_id is None:
            return
        async with self._db.session() as session:
            from sqlalchemy import select

            row = (
                await session.execute(
                    select(OrderAuditModel).where(OrderAuditModel.id == audit_id)
                )
            ).scalar_one_or_none()
            if row is not None:
                row.status = STATUS_FAILED
                row.error = error[:2000]
        logger.warning(
            "execution.failed", environment=environment.value, symbol=symbol, error=error
        )

    async def _audit_simple(
        self,
        action: str,
        environment: TradingEnvironment,
        *,
        symbol: str | None = None,
        order_id: str | None = None,
        status: str,
        response: dict[str, Any] | None = None,
    ) -> None:
        async with self._db.session() as session:
            repo = OrderAuditRepository(session)
            await repo.log(
                OrderAuditModel(
                    environment=environment.value,
                    action=action,
                    symbol=symbol.upper() if symbol else None,
                    broker_order_id=order_id,
                    status=status,
                    response=response,
                    created_at=dt.datetime.now(dt.UTC),
                )
            )


def _plan_payload(plan: TradePlan, extra: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "symbol": plan.symbol.upper(),
        "environment": plan.environment.value,
        "direction": plan.direction.value,
        "order_type": plan.order_type.value,
        "time_in_force": plan.time_in_force.value,
        "order_class": plan.order_class.value,
        "qty": plan.qty or None,
        "notional": plan.notional,
        "limit_price": plan.limit_price,
        "stop_loss": plan.stop_loss,
        "take_profit": plan.take_profit,
        "risk_amount": plan.risk_amount,
        "risk_pct": plan.risk_pct,
        "rr": plan.rr,
        "score": plan.score,
        "strategy": plan.strategy.value if plan.strategy else None,
        "timeframe": plan.timeframe,
        "fingerprint": plan.opportunity_fingerprint,
        "notes": list(plan.notes),
    }
    if extra:
        payload["extra"] = extra
    return payload


def _describe_request(request: Any) -> dict[str, Any]:
    if request is None:
        return {}
    for method in ("model_dump", "dict"):
        fn = getattr(request, method, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:  # pragma: no cover - defensive
                continue
            if isinstance(data, dict):
                return _jsonable(data)
    return {}


def _jsonable(value: Any) -> dict[str, Any]:
    """Best-effort JSON-safe view of a domain model or dict."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _stringify(dump(mode="json"))
        except Exception:  # pragma: no cover - defensive
            try:
                return _stringify(dump())
            except Exception:
                return {}
    if isinstance(value, dict):
        return _stringify(value)
    return {}


def _stringify(data: Any) -> dict[str, Any]:
    import json
    from datetime import date, datetime

    def default(obj: Any) -> str:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return str(obj)

    try:
        return json.loads(json.dumps(data, default=default))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {}


# Re-exported for callers that need the enum when building plans.
__all__.append("OrderClass")
