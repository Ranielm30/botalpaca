"""Layer 8 — Position Protection Manager.

Design constraints taken from the *real* Alpaca API, not from assumptions:

* A trailing stop **cannot** be a bracket/OCO child leg. Enabling trailing on a
  bracket position therefore needs a *transition*: a replacement stop is
  submitted and confirmed first (so the position is never unprotected), then the
  bracket legs are cancelled, then the trailing stop is submitted. Nothing is ever
  reported as placed without a real broker confirmation.
* A take-profit leg is **limit-only** in Alpaca.
* Bracket/OCO legs are **exit-only**.
* Partial exits are possible by reducing the ``qty`` of a protective order.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence

from botalpaca.config import ProtectionSettings, get_settings
from botalpaca.config.logging import get_logger
from botalpaca.db import Database, ProtectionRepository
from botalpaca.domain import (
    BotalpacaError,
    OrderClass,
    OrderSide,
    OrderState,
    OrderType,
    PositionSnapshot,
    ProtectionKind,
    ProtectionState,
    TradingEnvironment,
    ValidationError,
)
from botalpaca.execution import (
    BrokerError,
    ExecutionEngine,
    OrderBuilder,
    order_status_value,
)

log = get_logger(__name__)

_STOP_TYPES = {OrderTypeValue for OrderTypeValue in ("stop", "stop_limit")}


def _type_value(order: OrderState) -> str:
    return order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type)


def _is_trailing(order: OrderState) -> bool:
    return _type_value(order) == "trailing_stop"


# Order states that reserve the shares but can never execute.
# Alpaca keeps counting a cancelled order as held briefly, so the same
# terminal set the state resolver uses decides when shares are free.
_TERMINAL = frozenset({"filled", "canceled", "expired", "replaced", "rejected"})

_INERT_STATUSES = frozenset({"held", "pending_cancel", "pending_replace"})


# Order classes that hold their exits as a group. Cancelling any one leg of a
# group cancels all of them, so a leg that belongs to a live group must never be
# treated as debris -- doing so is what tore a working take-profit down.
_GROUP_CLASSES = {"oco", "bracket", "oto"}


def _group_child_ids(orders: Sequence[OrderState]) -> set[str]:
    """Ids of every exit order that belongs to a live advanced group.

    Alpaca parks a group's stop in ``HELD`` so the shares are not reserved
    twice, while the take-profit sits ``NEW`` as the parent. A ``HELD`` child of
    such a group is working protection, not an orphan holding the position
    hostage.
    """
    children: set[str] = set()
    for order in orders:
        # The engine already resolved membership, and it did so BEFORE dropping
        # terminal rows -- because the only row carrying the parent-to-child
        # link is the entry order, which is FILLED the moment its bracket
        # children exist. Re-deriving it here from live rows alone misses
        # exactly that parent and calls a working stop an orphan.
        resolved = getattr(order, "group_ids", None)
        if resolved:
            for member in resolved:
                if str(member) != str(order.id):
                    children.add(str(member))
            continue
        # Fallback for orders that were not resolved, which is the old
        # behaviour: read whatever the parent still carries.
        if _class_value(order) not in _GROUP_CLASSES:
            continue
        for leg in getattr(order, "legs", None) or ():
            if leg.id:
                children.add(str(leg.id))
    return children


def _class_value(order: OrderState) -> str:
    raw = getattr(order, "order_class", None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().lower()


def _is_inert(order: OrderState, *, protected_ids: set[str] | None = None) -> bool:
    """True when an order holds the shares without being working protection.

    A leg of a live bracket/OCO group is excluded: it is ``HELD`` by design and
    it is the stop the position is relying on.
    """
    if protected_ids and str(order.id) in protected_ids:
        return False
    return order_status_value(order) in _INERT_STATUSES



# Alpaca answers in raw broker JSON. Dumping that into an operator's chat is
# unreadable and leaks internals that mean nothing outside the log, so each
# known rejection gets a sentence. The raw error still goes to the log.
_REASONS: tuple[tuple[str, str], ...] = (
    (
        "insufficient qty available",
        "Alpaca tiene reservadas las acciones para otra orden viva. "
        "Se liberaran cuando el mercado abra.",
    ),
    (
        "order pending cancel",
        "hay una cancelacion pendiente en Alpaca desde antes",
    ),
    (
        "percentage must be between",
        "el valor de cierre enviado no es valido",
    ),
    (
        "not enough buying power",
        "no hay poder de compra suficiente para esa orden",
    ),
)


def readable_reason(error: object) -> str:
    """Turn a broker rejection into one sentence a human can act on."""
    text = str(error).lower()
    for needle, reason in _REASONS:
        if needle in text:
            return reason
    # Anything unknown: keep the first line only, never a JSON blob.
    first = str(error).strip().splitlines()[0] if str(error).strip() else "error desconocido"
    return first[:140]


class PositionProtectionManager:
    """Owns the lifecycle of every protective order in ONE environment."""

    def __init__(
        self,
        engine: ExecutionEngine,
        database: Database,
        *,
        settings: ProtectionSettings | None = None,
    ) -> None:
        self._engine = engine
        self._db = database
        self._settings = settings or get_settings().protection

    @property
    def settings(self) -> ProtectionSettings:
        return self._settings

    # ------------------------------------------------------------------- state

    async def get_state(self, environment: TradingEnvironment, symbol: str) -> ProtectionState:
        """Live protection. Alpaca's open orders are the source of truth."""
        symbol = symbol.upper()
        orders = await self._engine.get_live_orders_for_symbol(symbol)
        # Without a live position we assume a long book (the only case where a
        # protective sell leg cannot be mistaken for an entry).
        live = self._state_from_orders(environment, symbol, orders, OrderSide.SELL)

        async with self._db.session() as session:
            row = await ProtectionRepository(session).get(environment, symbol)
        if row is None:
            return live

        live.break_even_active = bool(row.break_even_active)
        live.time_stop_at = row.time_stop_at
        if row.order_class:
            try:
                live.order_class = OrderClass(row.order_class)
            except ValueError:
                live.order_class = None
        if row.stop_order_id and live.stop_order_id != row.stop_order_id:
            live.notes.append(
                f"El stop registrado en SQLite ({row.stop_order_id}) no coincide con Alpaca"
            )
        return live

    def _state_from_orders(
        self,
        environment: TradingEnvironment,
        symbol: str,
        orders: list[OrderState],
        exit_side: OrderSide | None,
    ) -> ProtectionState:
        state = ProtectionState(symbol=symbol, environment=environment)
        if exit_side is None:
            exit_side = OrderSide.SELL
        mine = [o for o in orders if o.symbol.upper() == symbol]
        # Resolve group membership before judging anything: a HELD stop that
        # belongs to a live OCO/bracket is the stop, not an orphan. Without
        # this the bot cancels the group and takes the take-profit down with it.
        protected = _group_child_ids(mine)
        for order in mine:
            # A HELD order reserves the shares but can never execute. Treating it
            # as protection is what made the bot believe a stop was in place
            # while nothing could actually fire.
            if _is_inert(order, protected_ids=protected):
                state.inert_order_ids.append(order.id)
                continue
            kind = _type_value(order)
            if _is_trailing(order):
                state.has_trailing = True
                state.trailing_order_id = order.id
                state.trail_percent = order.trail_percent
                state.trail_price = order.trail_price
                continue
            if kind in _STOP_TYPES and order.side is exit_side:
                state.has_stop = True
                state.stop_order_id = order.id
                state.stop_price = order.stop_price
            elif kind == "limit" and order.side is exit_side:
                state.has_take_profit = True
                state.take_profit_order_id = order.id
                state.take_profit_price = order.limit_price
            if state.order_class is None and order.order_class in (
                OrderClass.BRACKET,
                OrderClass.OCO,
            ):
                state.order_class = order.order_class
        if state.order_class is None and state.has_stop and state.has_take_profit:
            # Only a group Alpaca actually reported counts. Inferring OCO from
            # "there is a stop and there is a target" wrote an order class
            # into the ledger that the broker never confirmed, and everything
            # downstream that trusts it (break-even, the autonomy, the ledger)
            # then reasons on a guess.
            def _groups(order_id: str | None) -> set[str]:
                if order_id is None:
                    return set()
                for candidate in mine:
                    if candidate.id == order_id:
                        return {str(g) for g in (getattr(candidate, "group_ids", None) or ())}
                return set()

            if _groups(state.stop_order_id) & _groups(state.take_profit_order_id):
                state.order_class = OrderClass.OCO
        return state

    async def state_for_position(
        self, environment: TradingEnvironment, position: PositionSnapshot
    ) -> ProtectionState:
        """Protection state resolved against a known position direction."""
        symbol = position.symbol.upper()
        # Live orders, not just OPEN: a child left in HELD still reserves the
        # shares and has to be surfaced so it can be cleared.
        orders = await self._engine.get_live_orders_for_symbol(symbol)
        exit_side = OrderSide.SELL if position.qty >= 0 else OrderSide.BUY
        live = self._state_from_orders(environment, symbol, orders, exit_side)
        async with self._db.session() as session:
            row = await ProtectionRepository(session).get(environment, symbol)
        if row is None:
            return live
        live.break_even_active = bool(row.break_even_active)
        live.time_stop_at = row.time_stop_at
        # Falls back to the live stop only for rows registered before the
        # baseline existed, and only while nothing has moved it yet.
        live.initial_stop_price = row.initial_stop_price or row.stop_price
        return live

    # ------------------------------------------------------------ registration

    async def register_entry_protection(
        self,
        *,
        environment: TradingEnvironment,
        symbol: str,
        qty: float,
        entry_price: float,
        stop_order_id: str | None = None,
        take_profit_order_id: str | None = None,
        stop_price: float | None = None,
        take_profit_price: float | None = None,
        order_class: OrderClass | str | None = None,
        time_stop_minutes: int | None = None,
        initial_stop_price: float | None = None,
    ) -> ProtectionState:
        """Persist the protection Alpaca actually accepted for a fresh entry.

        ``initial_stop_price`` freezes the risk this trade was opened with. It is
        written only if the row has none, so re-registering a trade never resets
        the baseline a position is already being judged against.

        ``stop_price`` and ``take_profit_price`` write the *levels*, not just
        the order ids. The ids are only links to the legs Alpaca agreed to, and
        those legs are DAY orders that expire at the close; once they are gone
        the ids point at nothing and the levels have to be remembered on our
        side for the protection to be rebuilt the next morning.
        """
        order_class_value = getattr(order_class, "value", order_class)
        minutes = self._settings.default_time_stop_minutes if time_stop_minutes is None else time_stop_minutes
        time_stop_at = (
            dt.datetime.now(dt.UTC) + dt.timedelta(minutes=minutes) if minutes > 0 else None
        )
        async with self._db.session() as session:
            repo = ProtectionRepository(session)
            await repo.upsert(
                environment,
                symbol,
                qty=abs(qty),
                entry_price=entry_price,
                stop_order_id=stop_order_id,
                take_profit_order_id=take_profit_order_id,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                order_class=order_class_value,
                time_stop_at=time_stop_at,
            )
            if initial_stop_price is not None:
                await repo.record_initial_stop(environment, symbol, initial_stop_price)
        log.info(
            "protection.registered",
            environment=environment.value,
            symbol=symbol.upper(),
            has_stop=stop_order_id is not None,
            has_take_profit=take_profit_order_id is not None,
        )
        return await self.get_state(environment, symbol)

    async def _submit_exit(
        self,
        environment: TradingEnvironment,
        symbol: str,
        request: object,
        *,
        ignore: Sequence[str] = (),
        attempts: int = 4,
    ):
        """Send an exit order, waiting out the broker's release of the shares.

        Alpaca reserves the shares for every open exit order, and cancelling one
        only *starts* the release: the order sits in ``pending_cancel`` and the
        broker answers a create issued straight away with ``insufficient qty
        available ... held_for_orders: N``, naming the very order that was just
        cancelled. A single retry gave up inside that window and the trailing
        handover never armed, so a trade that reached the trigger kept a stop the
        operator believed had been replaced.

        Each attempt therefore waits for the shares to actually come back before
        trying again, and gives up only when the broker has demonstrably not
        released them.
        """
        for attempt in range(max(attempts, 1)):
            try:
                return await self._engine.submit_protective(
                    environment=environment,
                    request=request,
                    symbol=symbol,
                )
            except (BrokerError, BotalpacaError, RuntimeError) as exc:
                if "insufficient qty" not in str(exc):
                    raise
                last = exc
                log.info(
                    "protection.exit_resubmitting",
                    environment=environment.value,
                    symbol=symbol,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt + 1 >= max(attempts, 1):
                    break
                await self._await_shares_free(environment, symbol, ignore=ignore)
        raise last


    async def _await_shares_free(
        self,
        environment: TradingEnvironment,
        symbol: str,
        *,
        ignore: Sequence[str] = (),
        attempts: int = 30,
        delay: float = 0.5,
    ) -> bool:
        """Wait until Alpaca stops counting a cancelled order as held.

        Cancelling an exit order does not release its shares straight away. A
        create issued in the same breath is rejected with
        ``insufficient qty available ... held_for_orders: 4`` naming the very
        order that was just cancelled, and the replacement silently never lands.
        Polling for the order to actually go terminal is the difference between
        a protection swap that works and one that appears to.
        """
        skip = set(ignore)
        # Alpaca's release is asynchronous and its latency is not published. A
        # fixed 3.2s budget lost the race on the live box, so the ceiling is
        # generous and the delay backs off instead of hammering the API.
        ceiling = max(attempts, 1)
        for attempt in range(ceiling):
            try:
                orders = await self._engine.get_live_orders_for_symbol(symbol)
            except (BotalpacaError, BrokerError, RuntimeError) as exc:
                log.warning(
                    "protection.share_probe_failed",
                    environment=environment.value,
                    symbol=symbol,
                    error=str(exc),
                )
            else:
                held = [
                    o.id
                    for o in orders
                    if o.id not in skip and order_status_value(o) not in _TERMINAL
                ]
                if not held:
                    return True
            if attempt < attempts - 1:
                await asyncio.sleep(min(delay * (1.5 ** attempt), 2.5))
        return False

    async def _clear_inert_orders(
        self,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        state: ProtectionState,
    ) -> list[str]:
        """Cancel HELD or stuck orders so the shares become available again.

        Returns the ids that could not be cleared, because the caller then knows
        its replacement exit is likely to be rejected.

        A cancel that Alpaca *accepts* is not a cancel that Alpaca *completes*.
        Reporting success on acceptance made reconcile print "ordenes inertes
        liberadas" and then fail on the very next submit with the shares still
        held, every single cycle. So the wait is here, before anything claims the
        shares are back.
        """
        if not state.inert_order_ids:
            return []
        cleared: list[str] = []
        stuck: list[str] = []
        for order_id in state.inert_order_ids:
            try:
                await self._engine.cancel_order(order_id, environment=environment)
                cleared.append(order_id)
            except Exception as exc:  # noqa: BLE001 - report it, never hide it
                # "order pending cancel" means Alpaca already accepted this
                # cancel and is working on it. Reporting it as a failure every
                # cycle is noise that hides the failures that matter.
                if "pending cancel" in str(exc).lower():
                    cleared.append(order_id)
                    continue
                stuck.append(order_id)
                log.warning(
                    "protection.inert_cancel_failed",
                    environment=environment.value,
                    symbol=position.symbol,
                    order_id=order_id,
                    error=str(exc),
                )
        if cleared:
            # Wait for the broker to actually let go of the shares. Only then is
            # it true that the position can be re-protected.
            if await self._await_shares_free(
                environment, position.symbol, ignore=tuple(stuck)
            ):
                freed = len(cleared)
            else:
                # Still holding: report it as stuck rather than claim success.
                stuck.extend(cleared)
                freed = 0
            state.inert_order_ids = list(stuck)
            if freed:
                state.notes.append(
                    f"Ordenes inertes liberadas ({freed}): las acciones ya estan disponibles"
                )
            log.info(
                "protection.inert_cleared",
                environment=environment.value,
                symbol=position.symbol,
                cleared=freed,
                stuck=len(stuck),
            )
        return stuck

    async def clear(self, environment: TradingEnvironment, symbol: str) -> None:
        async with self._db.session() as session:
            await ProtectionRepository(session).delete(environment, symbol)
        log.info("protection.cleared", environment=environment.value, symbol=symbol.upper())

    # ------------------------------------------------------------- protection

    async def _restore_stop(
        self,
        environment: TradingEnvironment,
        position: object,
        stop_price: float,
    ) -> None:
        """Put a bare stop back after an upgrade to an OCO group failed.

        Releasing the stop to make room for the group is the only way Alpaca
        will accept it, so it has a rollback. Best effort by design: the caller
        is already on its way to raising, and letting this one escape would
        bury the original failure. What it must never do is stay silent, so
        every outcome here is logged.
        """
        try:
            await self._await_shares_free(environment, position.symbol)
            result = await self._submit_exit(
                environment=environment,
                request=OrderBuilder.build_protective_stop(
                    symbol=position.symbol,
                    qty=abs(position.qty),
                    exit_side=position.direction,
                    stop_price=stop_price,
                ),
                symbol=position.symbol,
            )
        except Exception as exc:  # noqa: BLE001 - already failing; do not mask the cause
            log.error(
                "protection.stop_restore_failed",
                environment=environment.value,
                symbol=position.symbol,
                stop=stop_price,
                error=str(exc),
            )
            return
        if result.order is None or not result.order.id:
            log.error(
                "protection.stop_restore_unconfirmed",
                environment=environment.value,
                symbol=position.symbol,
                stop=stop_price,
            )
            return
        await self._persist(
            environment,
            position.symbol,
            qty=abs(position.qty),
            entry_price=position.avg_entry_price,
            stop_order_id=result.order.id,
            stop_price=stop_price,
            # The group never landed, so nothing is protecting the upside now.
            # Clearing the stale ids keeps the ledger from advertising a target
            # and an OCO that do not exist at the broker.
            take_profit_order_id=None,
            order_class=None,
        )
        log.warning(
            "protection.stop_restored",
            environment=environment.value,
            symbol=position.symbol,
            stop=stop_price,
            order=result.order.id,
        )

    async def _merge_into_oco(
        self,
        environment: TradingEnvironment,
        position: object,
        state: ProtectionState,
        *,
        stop_price: float | None = None,
        target_price: float | None = None,
        replace_stop_id: str | None = None,
    ) -> ProtectionState | None:
        """Fold a standalone take-profit and the stop into one OCO group.

        Two independent exits cannot share the shares, so the stop has to be
        released before the target can be protected and vice versa. An OCO
        carries both at once: the parent is the take-profit limit, the child is
        the stop. Anything less leaves the operator with a risk number and no
        target, which is exactly what a card advertising "R:R 1.5" must never do.

        ``target_price`` lets a caller restore a level the ledger still knows
        but the live state no longer carries.

        ``replace_stop_id`` names a live standalone stop that has to be
        released first. Alpaca counts the shares of a stop as held, so a group
        that also wants them is rejected outright. That release is the only way
        through, which is why it is undone the moment the group does not land:
        a position must never stay naked because an upgrade was attempted.
        """
        target = target_price if target_price is not None else state.take_profit_price
        stop = stop_price if stop_price is not None else state.stop_price
        if target is None or stop is None:
            return None
        try:
            request = OrderBuilder.build_oco_exits(
                symbol=position.symbol,
                qty=abs(position.qty),
                exit_side=position.direction,
                stop_price=stop,
                take_profit_price=target,
            )
        except (ValidationError, ValueError) as exc:
            log.warning("protection.oco_rejected", symbol=position.symbol, error=str(exc))
            return None

        # The standalone target must go before the group can take the shares.
        if state.take_profit_order_id:
            try:
                await self._engine.cancel_order(
                    state.take_profit_order_id, environment=environment
                )
            except (BotalpacaError, BrokerError, RuntimeError) as exc:
                log.warning(
                    "protection.target_release_failed",
                    environment=environment.value,
                    symbol=position.symbol,
                    error=str(exc),
                )
                return None

        # Same for a standalone stop. Cancelled only here, immediately before
        # the group is attempted, and restored on any failure below.
        released_stop_id = replace_stop_id
        if released_stop_id:
            try:
                await self._engine.cancel_order(released_stop_id, environment=environment)
            except (BotalpacaError, BrokerError, RuntimeError) as exc:
                log.warning(
                    "protection.stop_release_failed",
                    environment=environment.value,
                    symbol=position.symbol,
                    stop=stop,
                    error=str(exc),
                )
                return None

        try:
            await self._await_shares_free(
                environment, position.symbol, ignore=state.inert_order_ids
            )
            result = await self._submit_exit(
                environment=environment, request=request, symbol=position.symbol
            )
        except Exception as exc:  # noqa: BLE001 - re-raised after the rollback below
            log.warning(
                "protection.oco_failed",
                environment=environment.value,
                symbol=position.symbol,
                stop=stop,
                take_profit=target,
                error=str(exc),
            )
            if released_stop_id:
                await self._restore_stop(environment, position, stop)
            raise
        if result.order is None or not result.order.id:
            log.warning("protection.oco_unconfirmed", symbol=position.symbol)
            if released_stop_id:
                await self._restore_stop(environment, position, stop)
            return None
        stop_id = result.order.id
        for leg in result.order.legs or ():
            if str(leg.order_type.value) == str(OrderType.STOP.value):
                stop_id = leg.id or stop_id
        await self._persist(
            environment,
            position.symbol,
            qty=abs(position.qty),
            entry_price=position.avg_entry_price,
            stop_order_id=stop_id,
            stop_price=stop,
            take_profit_order_id=result.order.id,
            take_profit_price=target,
            order_class=OrderClass.OCO,
        )
        state.has_stop = True
        state.stop_order_id = stop_id
        state.stop_price = stop
        state.has_take_profit = True
        state.take_profit_order_id = result.order.id
        state.take_profit_price = target
        state.notes.append(f"Stop {stop} y objetivo {target} unidos en un OCO")
        log.warning(
            "protection.oco_created",
            environment=environment.value,
            symbol=position.symbol,
            stop=stop,
            take_profit=target,
            parent=result.order.id,
        )
        return state


    async def ensure_stop(
        self,
        *,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        stop_price: float | None = None,
        atr: float | None = None,
        reason: str = "posición sin stop",
    ) -> ProtectionState:
        """Guarantee a stop exists. Places a real order when missing."""
        state = await self.state_for_position(environment, position)
        # An inert child still reserves the shares even when the position looks
        # protected, so Alpaca rejects every future exit for "insufficient qty
        # available". Clearing it is not optional just because a live stop
        # exists: the shares stay locked for the rest of the trade otherwise.
        if state.inert_order_ids:
            await self._clear_inert_orders(environment, position, state)
        if state.has_stop:
            return state

        # Alpaca reserves the shares behind every open exit order, so two
        # independent exits on one position are rejected outright. The way to
        # hold a stop AND a target is an OCO group, whose parent is the
        # take-profit limit and whose child is the stop. It was live-verified on
        # the paper account, so a missing target is never an excuse to drop one.
        if state.take_profit_order_id:
            # The caller usually knows the level it wants; the state may carry
            # none at all when there is no stop yet, which is exactly the case
            # this branch exists for.
            merged = await self._merge_into_oco(
                environment, position, state, stop_price=stop_price
            )
            # The group already carries the stop. Falling through would build
            # a bare stop on top of it, which takes the shares and leaves the
            # operator with a stop and no target -- the exact state we just
            # spent this round removing.
            if merged is not None and merged.has_stop:
                return merged


        # The fallback is measured from the live price, which is the only way to
        # protect a naked position with no entry price and no ATR. It knows
        # nothing about the risk the trade was approved for. A bracket whose DAY
        # legs expired at the close looks naked, and rebuilding the stop from the
        # closing price can put it *wider* than the one Alpaca just let die --
        # the position then carries twice the risk it was opened with, having
        # risked nothing in between. The frozen baseline is the surviving record
        # of that risk, so it wins here whenever it is still a legal stop.
        fallback = self._fallback_stop(position, atr)
        price = (
            stop_price
            if stop_price is not None
            else self._tighten_with_baseline(
                fallback, position, state.initial_stop_price
            )
        )
        if price is None:
            raise ValidationError(
                f"No se puede calcular un stop para {position.symbol} sin ATR ni precio explícito"
            )
        price = self._sanitize_stop(price, position)

        await self._await_shares_free(
            environment, position.symbol, ignore=state.inert_order_ids
        )
        # A broker refusal must reach the operator as a sentence about the
        # position, not as a raw exception. An unprotected position is the one
        # outcome they have to act on, so the wording says so plainly.
        try:
            result = await self._submit_exit(
                environment=environment,
                request=OrderBuilder.build_protective_stop(
                    symbol=position.symbol,
                    qty=abs(position.qty),
                    exit_side=position.direction,
                    stop_price=price,
                ),
                symbol=position.symbol,
            )
        except (BotalpacaError, BrokerError, RuntimeError) as exc:
            log.error(
                "protection.stop_refused",
                environment=environment.value,
                symbol=position.symbol,
                error=str(exc),
            )
            raise ValidationError(
                f"No se pudo proteger {position.symbol}: {readable_reason(exc)}. "
                "LA POSICION ESTA SIN STOP"
            ) from exc
        if result.order is None or not result.order.id:
            raise ValidationError(
                f"Alpaca no confirmó el stop de protección para {position.symbol}"
            )
        await self._persist(
            environment,
            position.symbol,
            qty=abs(position.qty),
            entry_price=position.avg_entry_price,
            stop_order_id=result.order.id,
            stop_price=price,
        )
        # Freeze the baseline here as well, not just on the way in. A position
        # that reached this point without one is an orphan that had no stop in
        # Alpaca when it was adopted, so ``_adopt_orphan`` had nothing to
        # freeze and the R rules were left measuring against a stop that moves.
        # record_initial_stop never rewrites, so this is safe to call every time.
        async with self._db.session() as session:
            await ProtectionRepository(session).record_initial_stop(
                environment, position.symbol, price
            )
        state.has_stop = True
        state.stop_order_id = result.order.id
        state.stop_price = price
        state.notes.append(f"Stop creado automáticamente en {price} ({reason})")
        log.warning(
            "protection.stop_created",
            environment=environment.value,
            symbol=position.symbol,
            stop=price,
            reason=reason,
        )
        return state

    async def ensure_take_profit(
        self,
        *,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        target_price: float | None,
        reason: str = "posición sin objetivo de beneficio",
    ) -> ProtectionState:
        """Guarantee a profit target exists. Places a real order when missing.

        The mirror of :meth:`ensure_stop`. A stop on its own protects the
        downside and then lets a winner sit forever, which is exactly what a
        half-dismantled bracket leaves behind: both legs go away together and
        only the stop gets rebuilt.
        """
        state = await self.state_for_position(environment, position)
        if state.has_take_profit:
            return state
        if target_price is None:
            raise ValidationError(
                f"No se conoce el objetivo de beneficio de {position.symbol}"
            )
        long_position = position.qty >= 0
        on_the_right_side = (
            position.current_price < target_price
            if long_position
            else position.current_price > target_price
        )
        if not on_the_right_side:
            raise ValidationError(
                f"El objetivo {target_price} de {position.symbol} ya quedo del lado "
                "equivocado del mercado; no tiene sentido perseguirlo"
            )

        # Two independent exits cannot share the shares, so a target added on
        # top of a live stop has to travel with it inside one OCO group.
        if state.has_stop:
            merged = await self._merge_into_oco(
                environment,
                position,
                state,
                target_price=target_price,
                # The live stop holds every share, so Alpaca would refuse the
                # group. Naming it here lets the merge release and, if anything
                # goes wrong, put it straight back.
                replace_stop_id=state.stop_order_id,
            )
            if merged is not None and merged.has_take_profit:
                return merged
            raise ValidationError(
                f"No se pudo reunir el stop y el objetivo de {position.symbol} "
                "en un solo grupo"
            )

        await self._await_shares_free(
            environment, position.symbol, ignore=state.inert_order_ids
        )
        try:
            result = await self._submit_exit(
                environment=environment,
                request=OrderBuilder.build_protective_limit(
                    symbol=position.symbol,
                    qty=abs(position.qty),
                    exit_side=position.direction,
                    limit_price=target_price,
                ),
                symbol=position.symbol,
            )
        except (BotalpacaError, BrokerError, RuntimeError) as exc:
            log.error(
                "protection.take_profit_refused",
                environment=environment.value,
                symbol=position.symbol,
                error=str(exc),
            )
            raise ValidationError(
                f"No se pudo fijar el objetivo de {position.symbol}: "
                f"{readable_reason(exc)}. LA POSICION NO TIENE OBJETIVO DE BENEFICIO"
            ) from exc
        if result.order is None or not result.order.id:
            raise ValidationError(
                f"Alpaca no confirmo el objetivo de beneficio para {position.symbol}"
            )
        await self._persist(
            environment,
            position.symbol,
            qty=abs(position.qty),
            entry_price=position.avg_entry_price,
            take_profit_order_id=result.order.id,
            take_profit_price=target_price,
        )
        state.has_take_profit = True
        state.take_profit_order_id = result.order.id
        state.take_profit_price = target_price
        state.notes.append(
            f"Objetivo de beneficio fijado en {target_price} ({reason})"
        )
        log.warning(
            "protection.take_profit_created",
            environment=environment.value,
            symbol=position.symbol,
            take_profit=target_price,
            order=result.order.id,
            reason=reason,
        )
        return state

    def _fallback_stop(self, position: PositionSnapshot, atr: float | None) -> float | None:
        """ATR-based stop, or a conservative fixed percentage when ATR is unknown.

        Returning *something* matters more than being clever here: a naked
        position is the one state that can lose money without limit, so
        reconciliation uses a wide, clearly-documented default rather than
        leaving the position unprotected.
        """
        if position.current_price <= 0:
            return None
        if atr is not None and atr > 0:
            distance = atr * self._settings.fallback_stop_atr_multiplier
        else:
            distance = position.current_price * self._settings.fallback_stop_pct / 100.0
        if position.qty >= 0:
            return position.current_price - distance
        return position.current_price + distance

    @staticmethod
    def _tighten_with_baseline(
        price: float | None, position: PositionSnapshot, baseline: float | None
    ) -> float | None:
        """Keep an emergency stop no looser than the risk the trade opened with.

        The caller passed that risk when it approved the position and Alpaca
        accepted it, but the *price* only lives in the order ids, and those are
        DAY legs that die at the close. The frozen baseline is the one record
        that survives, so when a naked position has to be re-stopped the
        baseline decides the level rather than the price the market closed at.

        For a long the tighter stop is the *higher* price, and for a short the
        lower one -- the mirror of the protection direction. Returns ``price``
        untouched when there is nothing to tighten against, or when the
        baseline already sits on the wrong side of the market: a stop placed
        beyond the price is rejected by Alpaca, and an unprotected position is
        a worse outcome than a wide one.
        """
        if price is None or baseline is None or position.current_price <= 0:
            return price
        price = round(float(price), 2)
        frozen = round(float(baseline), 2)
        if position.qty >= 0 and frozen < position.current_price:
            return max(price, frozen)
        if position.qty < 0 and frozen > position.current_price:
            return min(price, frozen)
        return price

    @staticmethod
    def _sanitize_stop(price: float, position: PositionSnapshot) -> float:
        price = round(float(price), 2)
        if price <= 0:
            raise ValidationError(f"Stop inválido ({price})")
        if position.current_price > 0:
            if position.qty >= 0 and price >= position.current_price:
                price = round(position.current_price * 0.99, 2)
            elif position.qty < 0 and price <= position.current_price:
                price = round(position.current_price * 1.01, 2)
        return price

    async def _discard_stale_order(
        self,
        environment: TradingEnvironment,
        symbol: str,
        stale_id: str | None,
        current_id: str,
    ) -> bool:
        """Make sure the order a replace superseded is genuinely gone.

        Alpaca answers a successful replace with a new id but does not promise
        the old order reached a terminal state. When it stays live the position
        ends up carrying two exits, and the looser one can still fire -- which
        turns a locked-in break-even into a realised loss.
        """
        if not stale_id or stale_id == current_id:
            return False
        try:
            await self._await_shares_free(environment, symbol, ignore=(current_id,))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "protection.stale_probe_failed",
                environment=environment.value,
                symbol=symbol,
                error=str(exc),
            )
        live = await self._engine.get_live_orders_for_symbol(symbol)
        if any(order.id == stale_id for order in live):
            log.warning(
                "protection.stale_order_survived_replace",
                environment=environment.value,
                symbol=symbol,
                stale_order_id=stale_id,
            )
            try:
                await self._engine.cancel_order(stale_id, environment=environment)
            except (BotalpacaError, BrokerError, RuntimeError) as exc:
                log.error(
                    "protection.stale_order_cancel_failed",
                    environment=environment.value,
                    symbol=symbol,
                    stale_order_id=stale_id,
                    error=str(exc),
                )
                return False
            log.info(
                "protection.stale_order_cancelled",
                environment=environment.value,
                symbol=symbol,
                stale_order_id=stale_id,
                replaced_by=current_id,
            )
            return True
        return False

    # ------------------------------------------------------------- break-even

    async def move_to_break_even(
        self,
        *,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        buffer_pct: float | None = None,
    ) -> ProtectionState:
        """Move the stop to break-even (+ buffer) once the trade is green."""
        entry = position.avg_entry_price
        if entry <= 0:
            raise ValidationError("Sin precio de entrada válido para break-even")
        buffer = self._settings.break_even_buffer_pct if buffer_pct is None else buffer_pct
        long_position = position.qty >= 0
        new_stop = (
            round(entry * (1 + buffer / 100.0), 2) if long_position else round(entry * (1 - buffer / 100.0), 2)
        )

        state = await self.state_for_position(environment, position)
        # A trailing stop is the exit for this position now. Building a fixed
        # stop underneath it leaves two exits fighting over the same shares:
        # each pass cancels the other and the position spends the day with no
        # working protection at all.
        if state.has_trailing:
            state.notes.append("Trailing activo: el break-even no aplica")
            return state
        if not state.has_stop or not state.stop_order_id:
            return await self.ensure_stop(
                environment=environment,
                position=position,
                stop_price=new_stop,
                reason="break-even",
            )
        if state.stop_price is not None:
            if long_position and new_stop <= state.stop_price:
                state.notes.append("El stop ya está en break-even o mejor")
                return state
            if not long_position and new_stop >= state.stop_price:
                state.notes.append("El stop ya está en break-even o mejor")
                return state

        try:
            replaced = await self._engine.replace_order(
                state.stop_order_id,
                OrderBuilder.build_replace(stop_price=new_stop),
                environment=environment,
            )
        except (BotalpacaError, BrokerError, RuntimeError) as exc:
            # Alpaca refuses to replace an order that is accepted, pending_new,
            # pending_cancel or pending_replace -- which is exactly the state a
            # bracket child is in right after the entry fills. Cancel and recreate
            # so break-even still lands.
            log.info(
                "protection.break_even_replace_rejected",
                environment=environment.value,
                symbol=position.symbol,
                error=str(exc),
            )
            await self._engine.cancel_order(state.stop_order_id, environment=environment)
            await self._await_shares_free(
                environment, position.symbol, ignore=state.inert_order_ids
            )
            created = await self._submit_exit(
                environment=environment,
                request=OrderBuilder.build_protective_stop(
                    symbol=position.symbol,
                    qty=abs(position.qty),
                    exit_side=position.direction,
                    stop_price=new_stop,
                ),
                symbol=position.symbol,
            )
            if created.order is None or not created.order.id:
                raise ValidationError(
                    f"No se pudo mover el stop de {position.symbol} a break-even: "
                    f"Alpaca no confirmo la orden nueva. LA POSICION ESTA SIN STOP"
                ) from exc
            await self._persist(
                environment,
                position.symbol,
                stop_order_id=created.order.id,
                stop_price=new_stop,
                break_even_active=True,
            )
            state.has_stop = True
            state.stop_order_id = created.order.id
            state.stop_price = new_stop
            state.break_even_active = True
            state.notes.append(f"Stop recreado en break-even ({new_stop})")
            log.info(
                "protection.break_even",
                environment=environment.value,
                symbol=position.symbol,
                new_stop=new_stop,
                method="cancel_recreate",
            )
            return state

        # Alpaca's replace is not atomic and does not guarantee the old order
        # died: a live check caught a position carrying both 332.45 and 327.14.
        # The looser one still fires and closes at a loss, so confirm it is gone
        # and remove it if the broker left it behind.
        previous_id = state.stop_order_id
        await self._discard_stale_order(environment, position.symbol, previous_id, replaced.id)

        await self._persist(
            environment,
            position.symbol,
            stop_order_id=replaced.id,
            stop_price=new_stop,
            break_even_active=True,
        )
        state.has_stop = True
        state.stop_order_id = replaced.id
        state.stop_price = new_stop
        state.break_even_active = True
        state.notes.append(f"Stop movido a break-even ({new_stop})")
        log.info(
            "protection.break_even",
            environment=environment.value,
            symbol=position.symbol,
            new_stop=new_stop,
        )
        return state

    # ---------------------------------------------------------------- trailing

    async def enable_trailing_stop(
        self,
        *,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        trail_percent: float | None = None,
        atr: float | None = None,
        floor_stop: float | None = None,
        widen_to: float | None = None,
    ) -> ProtectionState:
        """Arm a trailing stop, never looser than the protection already earned.

        ``floor_stop`` is the lowest price the trail may sit at. Arming a trailing
        stop below a break-even that has already been secured hands that profit
        straight back, which is exactly what happened: a trade stopped at 332.45
        was re-armed at 327.43 and would have closed at a loss while the ledger
        still claimed break-even was active.

        ``widen_to`` forces a wider trail when the live one has been squeezed
        against the price by a gap.
        """
        if widen_to is not None:
            percent = max(float(widen_to), self._settings.default_trailing_percent)
        else:
            percent = trail_percent or self._settings.default_trailing_percent
        if atr is not None and atr > 0 and position.current_price > 0:
            atr_pct = atr / position.current_price * 100.0 * self._settings.atr_trailing_multiplier
            percent = max(percent, round(atr_pct, 2))
        if floor_stop is not None and position.current_price > 0:
            # Alpaca derives the trail from a percentage of the price, so the
            # floor has to be expressed the same way to be honoured.
            long_position = position.qty >= 0
            room = (
                position.current_price - floor_stop
                if long_position
                else floor_stop - position.current_price
            )
            if room <= 0:
                # The floor is already at or past the price: every possible trail
                # is looser than what was earned, so refuse rather than unwind it.
                raise ValidationError(
                    f"El precio ({position.current_price:.2f}) ya esta en o por debajo del "
                    f"break-even ({floor_stop:.2f}); no se puede armar un trailing mas amplio"
                )
            required_pct = round(room / position.current_price * 100.0, 2)
            percent = max(percent, required_pct)
        if percent <= 0 or percent >= 100:
            raise ValidationError(f"Trail percent inválido ({percent})")

        state = await self.state_for_position(environment, position)
        if state.has_trailing and state.trailing_order_id:
            try:
                replaced = await self._engine.replace_order(
                    state.trailing_order_id,
                    OrderBuilder.build_replace(trail=percent),
                    environment=environment,
                )
            except (BotalpacaError, BrokerError, RuntimeError) as exc:
                # Same Alpaca rule as break-even: no replace while the order is
                # accepted/pending_new. Recreate the trailing order instead.
                log.info(
                    "protection.trailing_replace_rejected",
                    environment=environment.value,
                    symbol=position.symbol,
                    error=str(exc),
                )
                await self._engine.cancel_order(state.trailing_order_id, environment=environment)
                state.has_trailing = False
                state.trailing_order_id = None
            else:
                await self._persist(
                    environment,
                    position.symbol,
                    trail_percent=percent,
                    trailing_order_id=replaced.id,
                )
                state.trail_percent = percent
                state.trailing_order_id = replaced.id
                state.notes.append(f"Trailing actualizado a {percent}%")
                return state

        note = await self._transition_to_trailing(
            environment=environment, position=position, percent=percent
        )
        if note:
            state.notes.append(note)

        result = await self._submit_exit(
            environment=environment,
            request=OrderBuilder.build_trailing_stop(
                symbol=position.symbol,
                qty=abs(position.qty),
                exit_side=position.direction,
                trail_percent=percent,
            ),
            symbol=position.symbol,
        )
        if result.order is None or not result.order.id:
            raise ValidationError(f"Alpaca no confirmó el trailing stop de {position.symbol}")
        await self._persist(
            environment,
            position.symbol,
            qty=abs(position.qty),
            entry_price=position.avg_entry_price,
            trail_percent=percent,
            trailing_order_id=result.order.id,
        )
        state.has_trailing = True
        state.trail_percent = percent
        state.trailing_order_id = result.order.id
        state.notes.append(
            f"Trailing stop activo al {percent}% "
            f"({'protege contra subidas' if position.qty >= 0 else 'protege contra bajadas'})"
        )
        log.info(
            "protection.trailing_enabled",
            environment=environment.value,
            symbol=position.symbol,
            trail_percent=percent,
        )
        return state

    async def disable_trailing_stop(
        self, *, environment: TradingEnvironment, position: PositionSnapshot
    ) -> ProtectionState:
        """Cancel the trailing stop and restore a plain protective stop."""
        state = await self.state_for_position(environment, position)
        if not state.has_trailing or not state.trailing_order_id:
            state.notes.append("No hay trailing stop activo")
            return state
        trail_percent = state.trail_percent or self._settings.default_trailing_percent
        await self._engine.cancel_order(state.trailing_order_id, environment=environment)
        await self._persist(environment, position.symbol, trailing_order_id=None, trail_percent=None)
        if not state.has_stop:
            # The position must never be left unprotected: rebuild a plain stop at
            # the distance the trailing stop was protecting.
            offset = trail_percent * 1.5 / 100.0
            replacement = (
                round(position.current_price * (1 - offset), 2)
                if position.qty >= 0
                else round(position.current_price * (1 + offset), 2)
            )
            return await self.ensure_stop(
                environment=environment,
                position=position,
                stop_price=replacement,
                reason="reemplazo de trailing",
            )
        state.has_trailing = False
        state.trail_percent = None
        state.trailing_order_id = None
        state.notes.append("Trailing stop cancelado")
        return state

    async def _transition_to_trailing(
        self,
        *,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        percent: float,
    ) -> str | None:
        """Swap bracket/OCO legs for a plain stop before adding the trailing stop.

        Alpaca reserves the shares for every open exit order, so the old legs
        must be released BEFORE the replacement is created. Submitting first --
        which this used to do, to avoid a protection gap -- is always rejected
        with "insufficient qty available", which meant a trailing stop could
        never be armed on a bracket-protected position. The unavoidable gap is
        one API round-trip, and if the new stop then fails, the original legs
        are already gone, so the failure is reported loudly rather than
        silently leaving the position bare.
        """
        state = await self.state_for_position(environment, position)
        if not (state.has_stop or state.has_take_profit):
            return None

        long_position = position.qty >= 0
        offset = percent * 1.5 / 100.0
        provisional = round(
            position.current_price * (1 - offset) if long_position else position.current_price * (1 + offset),
            2,
        )
        # The replacement exists only to hold the position for the one round-trip
        # it takes to arm the real trailing order. It must never sit looser than
        # the stop already protecting the position: deriving it from the price
        # alone used to push it below the break-even that had been secured, and
        # the position ended up less protected after "adding" a trailing stop
        # than before. Tighten it to the stop already in place whenever that is
        # still a legal stop (on the protected side of the market).
        if state.stop_price is not None:
            if long_position and state.stop_price < position.current_price:
                provisional = max(provisional, round(state.stop_price, 2))
            elif not long_position and state.stop_price > position.current_price:
                provisional = min(provisional, round(state.stop_price, 2))

        stale = [oid for oid in (state.stop_order_id, state.take_profit_order_id) if oid]
        cancelled: list[str] = []
        for order_id in stale:
            try:
                await self._engine.cancel_order(order_id, environment=environment)
                cancelled.append(order_id)
            except Exception as exc:  # noqa: BLE001 - keep trying the rest
                log.error(
                    "protection.transition.cancel_failed",
                    order_id=order_id,
                    error=str(exc),
                )

        # Cancelling does not release the shares instantly, so wait for the
        # old legs to actually go terminal before asking for the replacement.
        await self._await_shares_free(environment, position.symbol)
        try:
            replacement = await self._submit_exit(
                environment=environment,
                request=OrderBuilder.build_protective_stop(
                    symbol=position.symbol,
                    qty=abs(position.qty),
                    exit_side=position.direction,
                    stop_price=provisional,
                ),
                symbol=position.symbol,
            )
        except (BotalpacaError, BrokerError, RuntimeError) as exc:
            # The old legs are already cancelled, so the position is now bare.
            # Say so plainly instead of pretending the transition was clean.
            raise ValidationError(
                f"Se cancelaron las protecciones de {position.symbol} pero no se pudo "
                f"crear el stop de reemplazo: {exc}. LA POSICION ESTA SIN STOP"
            ) from exc
        if replacement.order is None or not replacement.order.id:
            raise ValidationError(
                f"Se cancelaron las protecciones de {position.symbol} pero Alpaca no "
                f"confirmo el stop de reemplazo. LA POSICION ESTA SIN STOP"
            )

        await self._persist(
            environment,
            position.symbol,
            qty=abs(position.qty),
            entry_price=position.avg_entry_price,
            stop_order_id=replacement.order.id,
            stop_price=provisional,
            take_profit_order_id=None,
            order_class=None,
        )
        message = (
            f"Stop de reemplazo {replacement.order.id} confirmado en {provisional} "
            f"para armar el trailing"
        )
        if cancelled:
            message += f"; protecciones liberadas primero: {', '.join(cancelled)}"
        return message

    # ------------------------------------------------------------------ cancel

    async def cancel_protection(
        self,
        *,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        kinds: set[ProtectionKind] | None = None,
    ) -> ProtectionState:
        state = await self.state_for_position(environment, position)
        if kinds is None:
            kinds = {ProtectionKind.STOP_LOSS, ProtectionKind.TAKE_PROFIT, ProtectionKind.TRAILING_STOP}
        else:
            # Accept both ProtectionKind members and their raw strings.
            kinds = {ProtectionKind(k) if not isinstance(k, ProtectionKind) else k for k in kinds}
        updates: dict[str, object] = {}

        if ProtectionKind.STOP_LOSS in kinds and state.stop_order_id:
            await self._engine.cancel_order(state.stop_order_id, environment=environment)
            updates.update(stop_order_id=None, stop_price=None)
        if ProtectionKind.TAKE_PROFIT in kinds and state.take_profit_order_id:
            await self._engine.cancel_order(state.take_profit_order_id, environment=environment)
            updates.update(take_profit_order_id=None, take_profit_price=None)
        if ProtectionKind.TRAILING_STOP in kinds and state.trailing_order_id:
            await self._engine.cancel_order(state.trailing_order_id, environment=environment)
            updates.update(trailing_order_id=None, trail_percent=None)

        if updates:
            await self._persist(environment, position.symbol, **updates)
        return await self.state_for_position(environment, position)

    # ------------------------------------------------------------ progressive

    async def progressive_step(
        self,
        *,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        step_pct: float = 1.0,
    ) -> ProtectionState | None:
        s = self._settings
        if not s.progressive_stops:
            return None
        state = await self.state_for_position(environment, position)
        # A trailing stop already ratchets with the price. Adding a fixed stop on
        # top of it would make the two fight: each pass would cancel the other
        # and the position would spend every minute without a working exit.
        if state.has_trailing:
            return None
        # Against the baseline, never the live stop: once break-even has moved
        # the stop above the entry the live stop yields a negative denominator,
        # which turns R negative and the ratchet buffer into nonsense.
        baseline = state.initial_stop_price or state.stop_price
        r_multiple = r_multiple_of(position, baseline)
        if r_multiple is None or r_multiple < s.break_even_trigger_r:
            return None
        buffer = max(s.break_even_buffer_pct, r_multiple * step_pct)
        return await self.move_to_break_even(
            environment=environment, position=position, buffer_pct=buffer
        )

    async def check_momentum_exit(
        self, *, environment: TradingEnvironment, position: PositionSnapshot, momentum_score: float
    ) -> bool:
        """True when a momentum-loss exit rule has been configured and triggered."""
        s = self._settings
        if not s.momentum_exit_enabled:
            return False
        return momentum_score < s.momentum_score_threshold

    # --------------------------------------------------------------- time stop

    async def time_stop_deadline(
        self, environment: TradingEnvironment, symbol: str
    ) -> dt.datetime | None:
        async with self._db.session() as session:
            row = await ProtectionRepository(session).get(environment, symbol)
        return row.time_stop_at if row is not None else None

    async def is_time_stop_due(
        self,
        environment: TradingEnvironment,
        symbol: str,
        *,
        now: dt.datetime | None = None,
    ) -> bool:
        deadline = await self.time_stop_deadline(environment, symbol)
        if deadline is None:
            return False
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=dt.UTC)
        reference = now or dt.datetime.now(dt.UTC)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=dt.UTC)
        return reference >= deadline

    # ---------------------------------------------------------- reconciliation

    async def reconcile(
        self,
        *,
        environment: TradingEnvironment,
        positions: list[PositionSnapshot],
        open_trade_atr: dict[str, float] | None = None,
        protect_missing: bool | None = None,
    ) -> list[str]:
        """Permanent SQLite <-> Alpaca reconciliation. Returns human-readable notes."""
        auto = self._settings.auto_protect_missing_stop if protect_missing is None else protect_missing
        atr_by_symbol = {k.upper(): v for k, v in (open_trade_atr or {}).items()}
        notes: list[str] = []

        async with self._db.session() as session:
            stored = {row.symbol: row for row in await ProtectionRepository(session).all_for(environment)}

        live = {p.symbol.upper(): p for p in positions if p.qty != 0}

        for symbol, position in sorted(live.items()):
            state = await self.state_for_position(environment, position)
            # An inert order reserves the shares without protecting anything,
            # which locks out every future exit on this position. It can outlive
            # the protection that created it, so it is swept here too.
            if state.inert_order_ids:
                blocked = await self._clear_inert_orders(environment, position, state)
                if blocked:
                    notes.append(
                        f"🚨 {symbol}: {len(blocked)} orden(es) ancladas en Alpaca "
                        "retienen las acciones y no se pueden cancelar todavia"
                    )
                else:
                    notes.append(f"🧹 {symbol}: ordenes inertes liberadas")
            # Done before the trailing check below, which ``continue``s: an
            # orphan whose only exit is a trailing stop still needs its row.
            if symbol not in stored:
                # Alpaca holds the position but SQLite never heard of it: the
                # entry predates the protection ledger, or the row was cleared
                # while the shares stayed reserved. Adopt it for real, or the
                # frozen-risk baseline stays NULL forever and every R rule
                # falls back to a stop that moves with the trade.
                await self._adopt_orphan(environment, symbol, position, state)
                notes.append(
                    f"ℹ️ {symbol}: adoptada en SQLite (posición huérfana de Alpaca)"
                )
            # A trailing stop is the exit for this position, so it counts as
            # protection. Asking for a fixed stop on top of it fails anyway --
            # Alpaca reserves the shares for the trailing order and rejects the
            # second exit for insufficient qty available, which reads as a
            # failure to protect a position that is already protected.
            if state.has_trailing:
                continue
            # The target is restored before the stop, because restoring it on a
            # position that still has its stop goes through the OCO path and
            # leaves both exits standing; the reverse order would build a bare
            # limit on top of a stop that already holds the shares.
            if not state.has_take_profit:
                row = stored.get(symbol)
                target = row.take_profit_price if row is not None else None
                if target is None:
                    notes.append(f"⚠️ {symbol}: posición abierta SIN objetivo de beneficio")
                elif not auto:
                    notes.append(
                        f"⚠️ {symbol}: SIN objetivo de beneficio en {target} "
                        "(protección automática desactivada)"
                    )
                else:
                    try:
                        await self.ensure_take_profit(
                            environment=environment,
                            position=position,
                            target_price=target,
                            reason="reconciliación",
                        )
                        notes.append(f"🎯 {symbol}: objetivo de beneficio restaurado en {target}")
                    except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
                        notes.append(
                            f"⚠️ {symbol}: SIN objetivo de beneficio — {readable_reason(exc)}"
                        )
                        log.error(
                            "protection.reconcile.target_failed", symbol=symbol, error=str(exc)
                        )
            elif state.has_stop and state.order_class is None and auto:
                # A stop and a target are both live and the broker reported no
                # group: two independent exits standing on the same shares.
                # Alpaca only lets one of them hold the position, so the other
                # is dead weight that reads as a second exit. Folding them into
                # one OCO is the only shape where both can actually fire, and
                # _merge_into_oco puts the bare stop back if the group is
                # refused.
                try:
                    merged = await self._merge_into_oco(
                        environment=environment,
                        position=position,
                        state=state,
                        stop_price=state.stop_price,
                        target_price=state.take_profit_price,
                        replace_stop_id=state.stop_order_id,
                    )
                except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
                    notes.append(f"⚠️ {symbol}: dos salidas sueltas sin poder agrupar — {readable_reason(exc)}")
                    log.error(
                        "protection.reconcile.group_failed", symbol=symbol, error=str(exc)
                    )
                else:
                    if merged is None or not merged.has_take_profit:
                        notes.append(f"⚠️ {symbol}: dos salidas sueltas y el broker no admite el grupo")
                    else:
                        notes.append(f"🔗 {symbol}: stop y objetivo reunidos en un solo grupo")
            if not state.has_stop:
                if not auto:
                    notes.append(f"⚠️ {symbol}: posición abierta SIN stop en Alpaca")
                    continue
                try:
                    await self.ensure_stop(
                        environment=environment,
                        position=position,
                        atr=atr_by_symbol.get(symbol),
                        reason="reconciliación",
                    )
                    notes.append(f"🛡️ {symbol}: stop de emergencia creado")
                except Exception as exc:  # noqa: BLE001 - reconciliation must not abort
                    notes.append(
                        f"❌ {symbol}: SIN stop — {readable_reason(exc)}"
                    )
                    log.error("protection.reconcile.failed", symbol=symbol, error=str(exc))
        for symbol in sorted(set(stored) - set(live)):
            await self.clear(environment, symbol)
            notes.append(f"🧹 {symbol}: protección borrada (ya no hay posición en Alpaca)")

        if notes:
            log.info("protection.reconciled", environment=environment.value, count=len(notes))
        return notes

    # ---------------------------------------------------------------- helpers

    async def _adopt_orphan(
        self,
        environment: TradingEnvironment,
        symbol: str,
        position: PositionSnapshot,
        state: ProtectionState,
    ) -> None:
        """Write the protection row for a position Alpaca holds and SQLite lost.

        The exits recorded here are the ones Alpaca already accepted, so nothing
        is ordered: this only makes the row agree with the broker.

        The frozen baseline is taken from the stop live *at adoption time*. For
        a trade entered through ``register_entry_protection`` this method never
        runs and the true opening stop is what gets frozen; for an orphan that
        opened before the ledger existed the live stop is the earliest evidence
        left, and it is frozen from here on so R stops moving.
        """
        async with self._db.session() as session:
            repo = ProtectionRepository(session)
            await repo.upsert(
                environment,
                symbol,
                qty=abs(position.qty),
                entry_price=position.avg_entry_price,
                stop_price=state.stop_price,
                take_profit_price=state.take_profit_price,
                trail_percent=state.trail_percent,
                stop_order_id=state.stop_order_id,
                take_profit_order_id=state.take_profit_order_id,
                trailing_order_id=state.trailing_order_id,
                order_class=getattr(state.order_class, "value", state.order_class),
            )
            await repo.record_initial_stop(environment, symbol, state.stop_price)

    async def _persist(
        self, environment: TradingEnvironment, symbol: str, **values: object
    ) -> None:
        async with self._db.session() as session:
            await ProtectionRepository(session).upsert(environment, symbol, **values)


def r_multiple_of(
    position: PositionSnapshot, stop_price: float | None = None
) -> float | None:
    """Unrealized return expressed in R multiples using the live stop.

    R = profit / initial risk per share, where the risk per share comes from
    the *initial* stop the trade was opened with. When no stop is known the
    percent move is returned as a documented approximation, because inventing a
    risk denominator would be worse than a clearly-labelled proxy.
    """
    if position.qty == 0 or position.avg_entry_price <= 0:
        return None
    long_position = position.qty >= 0
    move = (
        position.current_price - position.avg_entry_price
        if long_position
        else position.avg_entry_price - position.current_price
    )
    if stop_price is not None:
        risk = abs(position.avg_entry_price - stop_price)
        if risk > 0:
            return move / risk
    pct_move = position.unrealized_plpc * 100.0
    return None if long_position and pct_move < 0 else abs(pct_move)


__all__ = ["PositionProtectionManager", "r_multiple_of"]
