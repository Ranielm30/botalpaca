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


def _is_inert(order: OrderState) -> bool:
    """True when an order still holds the shares but is not working protection."""
    status = order_status_value(order)
    return status in _INERT_STATUSES



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
        for order in orders:
            if order.symbol.upper() != symbol:
                continue
            # A HELD order reserves the shares but can never execute. Treating it
            # as protection is what made the bot believe a stop was in place
            # while nothing could actually fire.
            if _is_inert(order):
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
        if state.has_stop and state.has_take_profit and state.order_class is None:
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
        order_class: OrderClass | str | None = None,
        time_stop_minutes: int | None = None,
        initial_stop_price: float | None = None,
    ) -> ProtectionState:
        """Persist the protection Alpaca actually accepted for a fresh entry.

        ``initial_stop_price`` freezes the risk this trade was opened with. It is
        written only if the row has none, so re-registering a trade never resets
        the baseline a position is already being judged against.
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
    ):
        """Send an exit order, retrying once the broker frees the shares.

        Alpaca reserves the shares for every open exit order. Even after the old
        leg is cancelled the create can still land while the release is in
        flight, and it comes back as ``insufficient qty available ... naming the
        order that was just cancelled``. Without this the trailing handover
        never arms and the position keeps a stop the operator thinks has been
        replaced.
        """
        for attempt in range(2):
            try:
                return await self._engine.submit_protective(
                    environment=environment,
                    request=request,
                    symbol=symbol,
                )
            except (BrokerError, BotalpacaError, RuntimeError) as exc:
                if "insufficient qty" not in str(exc) or attempt:
                    raise
                log.info(
                    "protection.exit_resubmitting",
                    environment=environment.value,
                    symbol=symbol,
                    error=str(exc),
                )
                await self._await_shares_free(environment, symbol, ignore=ignore)
        return None

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

        # Alpaca reserves the shares for every open exit order, so a live
        # take-profit blocks a stop on the very same position. Alpaca will not
        # let two independent sell orders coexist, and it refuses an OTO whose
        # legs sit on opposite sides of the primary price, which is exactly the
        # long geometry of stop-below / target-above. The stop is the only exit
        # that can be kept alive here, and an unprotected position is worse than
        # one without a target: the target is recoverable by raising the stop as
        # the trade works, a loss is not. So the take-profit is released.
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
                raise ValidationError(
                    f"No se puede proteger {position.symbol}: su take-profit sigue "
                    f"retendiendo las acciones y no se pudo cancelar ({exc}). "
                    f"LA POSICION ESTA SIN STOP"
                ) from exc
            state.notes.append(
                "Take-profit liberado: Alpaca reserva las acciones por orden y "
                "no admite stop y target a la vez"
            )
            log.warning(
                "protection.target_released_for_stop",
                environment=environment.value,
                symbol=position.symbol,
                take_profit_order_id=state.take_profit_order_id,
                take_profit_price=state.take_profit_price,
            )
            state.has_take_profit = False
            state.take_profit_order_id = None
            state.take_profit_price = None

        price = stop_price if stop_price is not None else self._fallback_stop(position, atr)
        if price is None:
            raise ValidationError(
                f"No se puede calcular un stop para {position.symbol} sin ATR ni precio explícito"
            )
        price = self._sanitize_stop(price, position)

        await self._await_shares_free(
            environment, position.symbol, ignore=state.inert_order_ids
        )
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
            # A trailing stop is the exit for this position, so it counts as
            # protection. Asking for a fixed stop on top of it fails anyway --
            # Alpaca reserves the shares for the trailing order and rejects the
            # second exit for insufficient qty available, which reads as a
            # failure to protect a position that is already protected.
            if state.has_trailing:
                continue
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
            if symbol not in stored:
                notes.append(f"ℹ️ {symbol}: adoptada en SQLite (posición huérfana de Alpaca)")

        for symbol in sorted(set(stored) - set(live)):
            await self.clear(environment, symbol)
            notes.append(f"🧹 {symbol}: protección borrada (ya no hay posición en Alpaca)")

        if notes:
            log.info("protection.reconciled", environment=environment.value, count=len(notes))
        return notes

    # ---------------------------------------------------------------- helpers

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
