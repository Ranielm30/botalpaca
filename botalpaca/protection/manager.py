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

import datetime as dt

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
_INERT_STATUSES = frozenset({"held", "pending_cancel", "pending_replace"})


def _is_inert(order: OrderState) -> bool:
    """True when an order still holds the shares but is not working protection."""
    status = order_status_value(order)
    return status in _INERT_STATUSES


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
    ) -> ProtectionState:
        """Persist the protection Alpaca actually accepted for a fresh entry."""
        order_class_value = getattr(order_class, "value", order_class)
        minutes = self._settings.default_time_stop_minutes if time_stop_minutes is None else time_stop_minutes
        time_stop_at = (
            dt.datetime.now(dt.UTC) + dt.timedelta(minutes=minutes) if minutes > 0 else None
        )
        async with self._db.session() as session:
            await ProtectionRepository(session).upsert(
                environment,
                symbol,
                qty=abs(qty),
                entry_price=entry_price,
                stop_order_id=stop_order_id,
                take_profit_order_id=take_profit_order_id,
                order_class=order_class_value,
                time_stop_at=time_stop_at,
            )
        log.info(
            "protection.registered",
            environment=environment.value,
            symbol=symbol.upper(),
            has_stop=stop_order_id is not None,
            has_take_profit=take_profit_order_id is not None,
        )
        return await self.get_state(environment, symbol)

    async def _clear_inert_orders(
        self,
        environment: TradingEnvironment,
        position: PositionSnapshot,
        state: ProtectionState,
    ) -> list[str]:
        """Cancel HELD or stuck orders so the shares become available again.

        Returns the ids that could not be cleared, because the caller then knows
        its replacement exit is likely to be rejected.
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
                stuck.append(order_id)
                log.warning(
                    "protection.inert_cancel_failed",
                    environment=environment.value,
                    symbol=position.symbol,
                    order_id=order_id,
                    error=str(exc),
                )
        if cleared:
            state.inert_order_ids = stuck
            state.notes.append(
                f"Ordenes inertes canceladas ({len(cleared)}) para liberar las acciones"
            )
            log.info(
                "protection.inert_cleared",
                environment=environment.value,
                symbol=position.symbol,
                cleared=len(cleared),
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
        if state.has_stop:
            return state

        # An inert child still reserves the shares, so Alpaca would reject any
        # new exit for "insufficient qty available". Clear it first.
        await self._clear_inert_orders(environment, position, state)

        price = stop_price if stop_price is not None else self._fallback_stop(position, atr)
        if price is None:
            raise ValidationError(
                f"No se puede calcular un stop para {position.symbol} sin ATR ni precio explícito"
            )
        price = self._sanitize_stop(price, position)

        result = await self._engine.submit_protective(
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
            created = await self._engine.submit_protective(
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
    ) -> ProtectionState:
        percent = trail_percent or self._settings.default_trailing_percent
        if atr is not None and atr > 0 and position.current_price > 0:
            atr_pct = atr / position.current_price * 100.0 * self._settings.atr_trailing_multiplier
            percent = max(percent, round(atr_pct, 2))
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

        result = await self._engine.submit_protective(
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

        try:
            replacement = await self._engine.submit_protective(
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
        r_multiple = r_multiple_of(position, state.stop_price)
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
                    notes.append(f"❌ {symbol}: no se pudo crear stop automático ({exc})")
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
