"""Layer 11 — Trade journal.

Records every decision (accepted and rejected signals, opened trades, protection
changes, closures) so the bot can always answer "¿por qué recomendaste AAPL?".
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from botalpaca.config.logging import get_logger
from botalpaca.db import (
    DailyPnlRepository,
    Database,
    SignalRepository,
    TradeRepository,
)
from botalpaca.db.models import (
    DailyPnlModel,
    SignalModel,
    TradeEventModel,
    TradeModel,
)
from botalpaca.domain import (
    Opportunity,
    SetupType,
    SignalDirection,
    TradeStatus,
    TradingEnvironment,
)

log = get_logger(__name__)


class TradeJournal:
    """Append-only decision log, always scoped to one environment."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # ----------------------------------------------------------------- signals

    async def record_signal(
        self,
        opportunity: Opportunity,
        *,
        accepted: bool,
        indicators: dict[str, Any] | None = None,
        timeframe_higher: str | None = None,
    ) -> SignalModel:
        async with self._db.session() as session:
            row = SignalModel(
                environment=opportunity.environment.value,
                symbol=opportunity.symbol.upper(),
                strategy=opportunity.strategy.value,
                setup=opportunity.setup.value,
                timeframe=opportunity.timeframe,
                htf_timeframe=timeframe_higher,
                direction=opportunity.direction.value,
                score=opportunity.score,
                quality=opportunity.quality.value,
                entry=opportunity.entry,
                stop=opportunity.stop,
                target=opportunity.target,
                rr=opportunity.rr,
                atr=opportunity.atr,
                regime=opportunity.regime.value,
                sector=opportunity.sector,
                fingerprint=opportunity.fingerprint,
                confluences=opportunity.confluences,
                reasons=opportunity.reasons,
                indicators=indicators or {},
                accepted=accepted,
                created_at=opportunity.as_of,
            )
            return await SignalRepository(session).create(row)

    async def record_signal_if_new(
        self,
        opportunity: Opportunity,
        *,
        accepted: bool,
        indicators: dict[str, Any] | None = None,
        timeframe_higher: str | None = None,
        within_minutes: int | None = None,
    ) -> SignalModel | None:
        """Record a signal unless the same fingerprint was already seen.

        Returns the stored :class:`SignalModel`, or ``None`` when the signal was
        a duplicate inside the dedup window (nothing was written).

        Both accepted and rejected signals are stored: keeping the rejected ones
        is what stops the statistics from being polluted by hindsight.
        """
        if within_minutes is not None and await self.has_recent_signal(
            opportunity.environment, opportunity.fingerprint, within_minutes=within_minutes
        ):
            return None
        return await self.record_signal(
            opportunity,
            accepted=accepted,
            indicators=indicators,
            timeframe_higher=timeframe_higher,
        )

    async def latest_signal(
        self, environment: TradingEnvironment, fingerprint: str
    ) -> SignalModel | None:
        async with self._db.session() as session:
            return await SignalRepository(session).get_latest_by_fingerprint(
                environment, fingerprint
            )

    async def mark_signal_accepted(
        self, environment: TradingEnvironment, fingerprint: str | None
    ) -> None:
        """Flag the stored signal as acted upon once its trade is submitted."""
        if not fingerprint:
            return
        async with self._db.session() as session:
            repo = SignalRepository(session)
            row = await repo.get_latest_by_fingerprint(environment, fingerprint)
            if row is not None:
                await repo.set_accepted(row.id, environment)

    async def has_recent_signal(
        self, environment: TradingEnvironment, fingerprint: str, *, within_minutes: int
    ) -> bool:
        async with self._db.session() as session:
            return await SignalRepository(session).exists_fingerprint(
                environment, fingerprint, minutes=within_minutes
            )

    async def mark_notified(
        self, environment: TradingEnvironment, signal_id: int
    ) -> None:
        async with self._db.session() as session:
            await SignalRepository(session).mark_notified(signal_id, environment)

    async def link_signal_to_trade(
        self, environment: TradingEnvironment, fingerprint: str, trade_id: int
    ) -> None:
        async with self._db.session() as session:
            signal = await SignalRepository(session).get_latest_by_fingerprint(
                environment, fingerprint
            )
            if signal is not None:
                signal.trade_id = trade_id

    # ------------------------------------------------------------------ trades

    async def open_trade(
        self,
        *,
        environment: TradingEnvironment,
        symbol: str,
        direction: SignalDirection,
        strategy: SetupType,
        setup: SetupType,
        timeframe: str,
        qty: float,
        entry_price: float,
        stop_price: float | None,
        target_price: float | None,
        score: float,
        rr: float,
        risk_amount: float,
        atr: float | None,
        regime: str,
        sector: str | None,
        indicators: dict[str, Any] | None = None,
        confluences: list[str] | None = None,
        entry_reason: str | None = None,
        entry_order_id: str | None = None,
        client_order_id: str | None = None,
        stop_order_id: str | None = None,
        take_profit_order_id: str | None = None,
        time_stop_at: dt.datetime | None = None,
        metadata: dict[str, Any] | None = None,
        filled_at: dt.datetime | None = None,
    ) -> TradeModel:
        async with self._db.session() as session:
            row = TradeModel(
                environment=environment.value,
                symbol=symbol.upper(),
                direction=direction.value,
                strategy=strategy.value,
                setup=setup.value,
                timeframe=timeframe,
                status=TradeStatus.OPEN.value,
                qty=qty,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                score=score,
                rr=rr,
                risk_amount=risk_amount,
                atr=atr,
                regime=regime,
                sector=sector,
                indicators=indicators or {},
                confluences=confluences or [],
                entry_reason=entry_reason,
                entry_order_id=entry_order_id,
                client_order_id=client_order_id,
                stop_order_id=stop_order_id,
                take_profit_order_id=take_profit_order_id,
                has_stop=stop_order_id is not None or stop_price is not None,
                time_stop_at=time_stop_at,
                metadata_json=metadata or {},
                opened_at=dt.datetime.now(dt.UTC),
                filled_at=filled_at,
            )
            session.add(row)
            await session.flush()
            await TradeRepository(session).add_event(
                row.id,
                environment,
                TradeStatus.OPEN.value,
                payload={
                    "note": (
                        "Posición abierta"
                        if filled_at is not None
                        else "Entrada pendiente de ejecución"
                    )
                },
            )
            log.info(
                "journal.trade_opened",
                environment=environment.value,
                symbol=row.symbol,
                qty=qty,
                entry_price=entry_price,
            )
            return row

    async def close_trade(
        self,
        *,
        environment: TradingEnvironment,
        symbol: str,
        exit_price: float,
        pnl: float,
        exit_reason: str,
        exit_reason_note: str | None = None,
        exit_order_id: str | None = None,
        mfe: float | None = None,
        mae: float | None = None,
        trade_id: int | None = None,
    ) -> TradeModel | None:
        """Close an open trade.

        ``trade_id`` names the exact row to settle. Callers that already know
        which trade they are looking at pass it, because picking "the newest
        open row for this symbol" is only a guess: two OPEN rows on one symbol
        means the guess can close the wrong one.
        """
        now = dt.datetime.now(dt.UTC)
        async with self._db.session() as session:
            repo = TradeRepository(session)
            if trade_id is None:
                row = await repo.get_open_for_symbol(symbol, environment)
            else:
                row = await repo.get_open_by_id(trade_id, environment)
            if row is None:
                log.warning("journal.close.not_found", symbol=symbol, environment=environment.value)
                return None

            direction = SignalDirection(row.direction)
            entry = float(row.entry_price or 0.0)
            qty = float(row.qty or 0.0)
            risk_per_share = abs(entry - float(row.stop_price)) if row.stop_price else 0.0

            row.status = TradeStatus.CLOSED.value
            row.exit_price = exit_price
            row.exit_reason = exit_reason
            row.exit_reason_note = exit_reason_note
            row.exit_order_id = exit_order_id
            row.closed_at = now
            # SQLite hands back naive datetimes; normalize before subtracting.
            opened_at = row.opened_at
            if opened_at is not None and opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=dt.UTC)
            row.duration_seconds = (
                int((now - opened_at).total_seconds()) if opened_at is not None else 0
            )
            row.pnl = pnl
            row.pnl_pct = (pnl / (entry * abs(qty)) * 100.0) if entry and qty else 0.0
            row.r_multiple = (
                pnl / (risk_per_share * abs(qty)) if risk_per_share > 0 and qty else 0.0
            )
            if mfe is not None:
                row.mfe = mfe
            if mae is not None:
                row.mae = mae
            if risk_per_share > 0 and qty:
                row.mfe_r = (row.mfe or 0.0) / risk_per_share
                row.mae_r = (row.mae or 0.0) / risk_per_share
            del direction

            await repo.add_event(
                row.id,
                environment,
                TradeStatus.CLOSED.value,
                payload={"note": exit_reason, "exit_reason": exit_reason, "pnl": pnl},
            )
            await self._accumulate_daily_pnl(session, environment, pnl, now.date())
            log.info(
                "journal.trade_closed",
                environment=environment.value,
                symbol=row.symbol,
                pnl=round(pnl, 2),
                r_multiple=round(row.r_multiple, 3),
                reason=exit_reason,
            )
            return row

    async def _accumulate_daily_pnl(
        self, session: Any, environment: TradingEnvironment, pnl: float, day: dt.date
    ) -> None:
        repo = DailyPnlRepository(session)
        row: DailyPnlModel = await repo.upsert_day(environment, day)
        row.realized_pnl = float(row.realized_pnl or 0.0) + pnl
        row.trade_count = int(row.trade_count or 0) + 1
        if pnl > 0:
            row.win_count = int(row.win_count or 0) + 1
        elif pnl < 0:
            row.loss_count = int(row.loss_count or 0) + 1
        row.gross_profit = float(row.gross_profit or 0.0) + (pnl if pnl > 0 else 0.0)
        row.gross_loss = float(row.gross_loss or 0.0) + (abs(pnl) if pnl < 0 else 0.0)

    async def update_mfe_mae(
        self,
        *,
        environment: TradingEnvironment,
        symbol: str,
        favorable_price: float,
        adverse_price: float,
        direction: SignalDirection,
        entry_price: float,
    ) -> None:
        async with self._db.session() as session:
            row = await TradeRepository(session).get_open_for_symbol(symbol, environment)
            if row is None:
                return
            if direction is SignalDirection.LONG:
                row.mfe = max(float(row.mfe or 0.0), favorable_price - entry_price)
                row.mae = min(float(row.mae or 0.0), adverse_price - entry_price)
            else:
                row.mfe = max(float(row.mfe or 0.0), entry_price - favorable_price)
                row.mae = min(float(row.mae or 0.0), entry_price - adverse_price)

    async def add_event(
        self,
        *,
        environment: TradingEnvironment,
        symbol: str,
        event_type: str,
        note: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        async with self._db.session() as session:
            row = await TradeRepository(session).get_open_for_symbol(symbol, environment)
            trade_id = row.id if row is not None else None
            session.add(
                TradeEventModel(
                    environment=environment.value,
                    trade_id=trade_id,
                    event_type=event_type,
                    payload={**(payload or {}), **({"note": note} if note else {})},
                    created_at=dt.datetime.now(dt.UTC),
                )
            )

    # ----------------------------------------------------------------- queries

    async def explain(
        self,
        environment: TradingEnvironment,
        symbol: str,
        *,
        since: dt.datetime | None = None,
    ) -> dict[str, Any] | None:
        """Everything needed to answer "¿por qué recomendaste X?"."""
        async with self._db.session() as session:
            trades = await TradeRepository(session).get_all(
                environment, symbol=symbol, limit=50, since=since
            )
            signals = await SignalRepository(session).recent(
                environment, symbol=symbol.upper(), limit=50
            )
        if not trades and not signals:
            return None

        latest_trade = trades[0] if trades else None
        latest_signal = signals[0] if signals else None
        return {
            "environment": environment.value,
            "symbol": symbol.upper(),
            "trade": latest_trade,
            "signal": latest_signal,
            "history_count": len(trades),
        }


__all__ = ["TradeJournal"]
