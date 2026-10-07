"""Repositories: the only place SQL is written.

Every method that reads or writes financial data requires an explicit
``environment`` argument. There is intentionally no "get all trades" method,
so it is structurally impossible for PAPER data to surface in a REAL query.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from botalpaca.db.models import (
    AppStateModel,
    DailyPnlModel,
    OrderAuditModel,
    PositionProtectionModel,
    SettingModel,
    SignalModel,
    TradeEventModel,
    TradeModel,
)
from botalpaca.domain import (
    TradeStatus,
    TradingEnvironment,
)


def _env_value(environment: TradingEnvironment | str) -> str:
    return environment.value if isinstance(environment, TradingEnvironment) else str(environment)


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    """SQLite drops tzinfo; restore UTC so arithmetic stays correct."""
    if value is None:
        return None
    return value.replace(tzinfo=dt.UTC) if value.tzinfo is None else value


class TradeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    def _base(self, environment: TradingEnvironment) -> Select[Any]:
        return select(TradeModel).where(
            TradeModel.environment == _env_value(environment)
        )

    async def create(self, trade: TradeModel) -> TradeModel:
        self._s.add(trade)
        await self._s.flush()
        return trade

    async def get(self, trade_id: int, environment: TradingEnvironment) -> TradeModel | None:
        stmt = self._base(environment).where(TradeModel.id == trade_id)
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def get_by_client_order_id(
        self, client_order_id: str, environment: TradingEnvironment
    ) -> TradeModel | None:
        stmt = self._base(environment).where(TradeModel.client_order_id == client_order_id)
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def get_open_positions(self, environment: TradingEnvironment) -> Sequence[TradeModel]:
        stmt = (
            self._base(environment)
            .where(TradeModel.status == TradeStatus.OPEN.value)
            .order_by(TradeModel.opened_at)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def get_open_for_symbol(
        self, symbol: str, environment: TradingEnvironment
    ) -> TradeModel | None:
        stmt = (
            self._base(environment)
            .where(TradeModel.status == TradeStatus.OPEN.value)
            .where(TradeModel.symbol == symbol.upper())
            .order_by(TradeModel.opened_at.desc())
        )
        return (await self._s.execute(stmt)).scalars().first()

    async def get_open_by_id(
        self, trade_id: int, environment: TradingEnvironment
    ) -> TradeModel | None:
        """The one OPEN row with this id.

        Settling a trade must not depend on which row happens to be newest for
        a symbol, so the id is resolved directly.
        """
        stmt = (
            self._base(environment)
            .where(TradeModel.id == trade_id)
            .where(TradeModel.status == TradeStatus.OPEN.value)
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def get_pending(
        self, environment: TradingEnvironment
    ) -> Sequence[TradeModel]:
        """Open trades whose entry order has not filled yet.

        A pending trade is a real commitment: Alpaca holds the order and will
        fill it at the open, so it stays out of ``status`` and out of nothing
        else either. It is only excluded from the "positions" view.
        """
        stmt = (
            self._base(environment)
            .where(TradeModel.status == TradeStatus.OPEN.value)
            .where(TradeModel.filled_at.is_(None))
            .where(TradeModel.qty != 0)
            .order_by(TradeModel.opened_at)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def get_pending_for_order(
        self, order_id: str, environment: TradingEnvironment
    ) -> TradeModel | None:
        """The pending trade sitting behind a given Alpaca entry order."""
        if not order_id:
            return None
        stmt = (
            self._base(environment)
            .where(TradeModel.entry_order_id == order_id)
            .where(TradeModel.status == TradeStatus.OPEN.value)
            .where(TradeModel.filled_at.is_(None))
        )
        return (await self._s.execute(stmt)).scalars().first()

    async def mark_filled(
        self,
        trade_id: int,
        environment: TradingEnvironment,
        *,
        filled_at: dt.datetime,
        entry_price: float | None = None,
    ) -> bool:
        """Flip a pending trade to filled. Returns False if it already was.

        The entry price is corrected to what Alpaca actually paid, not what the
        plan assumed, so the P&L the operator sees is the real one.
        """
        row = await self.get(trade_id, environment)
        if row is None or row.filled_at is not None:
            return False
        row.filled_at = filled_at
        if entry_price is not None:
            row.entry_price = entry_price
        await self._s.flush()
        await self.add_event(
            trade_id,
            environment,
            TradeStatus.OPEN.value,
            payload={
                "note": f"Ejecutada a {entry_price:.2f}" if entry_price else "Ejecutada"
            },
        )
        return True

    async def get_all(
        self,
        environment: TradingEnvironment,
        *,
        limit: int = 100,
        offset: int = 0,
        status: TradeStatus | None = None,
        symbol: str | None = None,
        strategy: str | None = None,
        since: dt.datetime | None = None,
    ) -> Sequence[TradeModel]:
        stmt = self._base(environment)
        if status is not None:
            stmt = stmt.where(TradeModel.status == status.value)
        if symbol:
            stmt = stmt.where(TradeModel.symbol == symbol.upper())
        if strategy:
            stmt = stmt.where(TradeModel.strategy == strategy)
        if since is not None:
            stmt = stmt.where(TradeModel.opened_at >= since)
        stmt = stmt.order_by(TradeModel.opened_at.desc()).limit(limit).offset(offset)
        return (await self._s.execute(stmt)).scalars().all()

    async def get_closed(
        self, environment: TradingEnvironment, *, limit: int = 5000
    ) -> Sequence[TradeModel]:
        stmt = (
            self._base(environment)
            .where(TradeModel.status == TradeStatus.CLOSED.value)
            .order_by(TradeModel.closed_at)
        )
        if limit:
            stmt = stmt.limit(limit)
        return (await self._s.execute(stmt)).scalars().all()

    async def count_open(self, environment: TradingEnvironment) -> int:
        stmt = select(func.count()).select_from(TradeModel).where(
            and_(
                TradeModel.environment == _env_value(environment),
                TradeModel.status == TradeStatus.OPEN.value,
            )
        )
        return int((await self._s.execute(stmt)).scalar_one())

    async def total_open_exposure(self, environment: TradingEnvironment) -> float:
        stmt = select(func.coalesce(func.sum(TradeModel.qty * TradeModel.entry_price), 0.0)).where(
            and_(
                TradeModel.environment == _env_value(environment),
                TradeModel.status == TradeStatus.OPEN.value,
            )
        )
        return float((await self._s.execute(stmt)).scalar_one() or 0.0)

    async def exposure_by_sector(self, environment: TradingEnvironment) -> dict[str, float]:
        stmt = (
            select(TradeModel.sector, func.sum(TradeModel.qty * TradeModel.entry_price))
            .where(
                and_(
                    TradeModel.environment == _env_value(environment),
                    TradeModel.status == TradeStatus.OPEN.value,
                )
            )
            .group_by(TradeModel.sector)
        )
        rows = (await self._s.execute(stmt)).all()
        return {str(r[0] or "UNKNOWN"): float(r[1] or 0.0) for r in rows}

    async def exposure_by_symbol(self, environment: TradingEnvironment) -> dict[str, float]:
        stmt = (
            select(TradeModel.symbol, func.sum(TradeModel.qty * TradeModel.entry_price))
            .where(
                and_(
                    TradeModel.environment == _env_value(environment),
                    TradeModel.status == TradeStatus.OPEN.value,
                )
            )
            .group_by(TradeModel.symbol)
        )
        rows = (await self._s.execute(stmt)).all()
        return {str(r[0]): float(r[1] or 0.0) for r in rows}

    async def delete(self, trade_id: int, environment: TradingEnvironment) -> bool:
        stmt = delete(TradeModel).where(
            and_(TradeModel.id == trade_id, TradeModel.environment == _env_value(environment))
        )
        result = await self._s.execute(stmt)
        return bool(result.rowcount)

    async def events(self, trade_id: int, environment: TradingEnvironment) -> Sequence[TradeEventModel]:
        stmt = (
            select(TradeEventModel)
            .where(
                and_(
                    TradeEventModel.trade_id == trade_id,
                    TradeEventModel.environment == _env_value(environment),
                )
            )
            .order_by(TradeEventModel.created_at)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def add_event(
        self,
        trade_id: int,
        environment: TradingEnvironment,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> TradeEventModel:
        event = TradeEventModel(
            trade_id=trade_id,
            environment=_env_value(environment),
            event_type=event_type,
            payload=payload or {},
        )
        self._s.add(event)
        await self._s.flush()
        return event

    async def strategy_stats(self, environment: TradingEnvironment) -> dict[str, dict[str, Any]]:
        return await self._group_stats(environment, TradeModel.strategy)

    async def symbol_stats(self, environment: TradingEnvironment) -> dict[str, dict[str, Any]]:
        return await self._group_stats(environment, TradeModel.symbol)

    async def setup_stats(self, environment: TradingEnvironment) -> dict[str, dict[str, Any]]:
        return await self._group_stats(environment, TradeModel.setup)

    async def regime_stats(self, environment: TradingEnvironment) -> dict[str, dict[str, Any]]:
        return await self._group_stats(environment, TradeModel.regime)

    async def sector_stats(self, environment: TradingEnvironment) -> dict[str, dict[str, Any]]:
        return await self._group_stats(environment, TradeModel.sector)

    async def _group_stats(
        self, environment: TradingEnvironment, column: Any
    ) -> dict[str, dict[str, Any]]:
        stmt = (
            select(
                column,
                func.count(TradeModel.id),
                func.sum(func.iif(TradeModel.pnl > 0, 1, 0)),
                func.sum(func.iif(TradeModel.pnl <= 0, 1, 0)),
                func.avg(TradeModel.r_multiple),
                func.sum(TradeModel.pnl),
            )
            .where(
                and_(
                    TradeModel.environment == _env_value(environment),
                    TradeModel.status == TradeStatus.CLOSED.value,
                )
            )
            .group_by(column)
        )
        rows = (await self._s.execute(stmt)).all()
        out: dict[str, dict[str, Any]] = {}
        for key, total, wins, losses, avg_r, total_pnl in rows:
            out[str(key or "UNKNOWN")] = {
                "total": int(total or 0),
                "wins": int(wins or 0),
                "losses": int(losses or 0),
                "win_rate": (float(wins or 0) / float(total)) * 100.0 if total else 0.0,
                "avg_r": float(avg_r or 0.0),
                "total_pnl": float(total_pnl or 0.0),
            }
        return out

    async def closed_since(
        self, environment: TradingEnvironment, since: dt.datetime
    ) -> Sequence[TradeModel]:
        stmt = (
            self._base(environment)
            .where(TradeModel.status == TradeStatus.CLOSED.value)
            .where(TradeModel.closed_at >= since)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def filter_closed(
        self,
        environment: TradingEnvironment,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
        setup: str | None = None,
        regime: str | None = None,
        sector: str | None = None,
        timeframe: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        opened_after: dt.datetime | None = None,
    ) -> Sequence[TradeModel]:
        stmt = (
            self._base(environment)
            .where(TradeModel.status == TradeStatus.CLOSED.value)
            .order_by(TradeModel.closed_at)
        )
        if symbol:
            stmt = stmt.where(TradeModel.symbol == symbol.upper())
        if strategy:
            stmt = stmt.where(TradeModel.strategy == strategy)
        if setup:
            stmt = stmt.where(TradeModel.setup == setup)
        if regime:
            stmt = stmt.where(TradeModel.regime == regime)
        if sector:
            stmt = stmt.where(TradeModel.sector == sector)
        if timeframe:
            stmt = stmt.where(TradeModel.timeframe == timeframe)
        if min_score is not None:
            stmt = stmt.where(TradeModel.score >= min_score)
        if max_score is not None:
            stmt = stmt.where(TradeModel.score <= max_score)
        if opened_after is not None:
            stmt = stmt.where(TradeModel.opened_at >= opened_after)
        return (await self._s.execute(stmt)).scalars().all()


class SignalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    def _base(self, environment: TradingEnvironment) -> Select[Any]:
        return select(SignalModel).where(SignalModel.environment == _env_value(environment))

    async def create(self, signal: SignalModel) -> SignalModel:
        self._s.add(signal)
        await self._s.flush()
        return signal

    async def exists_fingerprint(
        self, environment: TradingEnvironment, fingerprint: str, *, minutes: int = 180
    ) -> bool:
        since = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=minutes)
        stmt = select(func.count()).select_from(SignalModel).where(
            and_(
                SignalModel.environment == _env_value(environment),
                SignalModel.fingerprint == fingerprint,
                SignalModel.created_at >= since,
            )
        )
        return int((await self._s.execute(stmt)).scalar_one() or 0) > 0

    async def recent(
        self,
        environment: TradingEnvironment,
        *,
        limit: int = 50,
        hours: int = 48,
        symbol: str | None = None,
    ) -> Sequence[SignalModel]:
        since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
        stmt = (
            self._base(environment)
            .where(SignalModel.created_at >= since)
            .order_by(SignalModel.created_at.desc())
            .limit(limit)
        )
        if symbol:
            stmt = stmt.where(SignalModel.symbol == symbol.upper())
        return (await self._s.execute(stmt)).scalars().all()

    async def get(self, signal_id: int, environment: TradingEnvironment) -> SignalModel | None:
        stmt = self._base(environment).where(SignalModel.id == signal_id)
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def get_latest_by_fingerprint(
        self, environment: TradingEnvironment, fingerprint: str
    ) -> SignalModel | None:
        stmt = (
            self._base(environment)
            .where(SignalModel.fingerprint == fingerprint)
            .order_by(SignalModel.created_at.desc())
        )
        return (await self._s.execute(stmt)).scalars().first()

    async def set_accepted(
        self, signal_id: int, environment: TradingEnvironment, trade_id: int | None = None
    ) -> None:
        await self._s.execute(
            update(SignalModel)
            .where(and_(SignalModel.id == signal_id, SignalModel.environment == _env_value(environment)))
            .values(accepted=True, trade_id=trade_id)
        )

    async def set_rejected(
        self, signal_id: int, environment: TradingEnvironment, reason: str
    ) -> None:
        await self._s.execute(
            update(SignalModel)
            .where(and_(SignalModel.id == signal_id, SignalModel.environment == _env_value(environment)))
            .values(accepted=False, rejected_reason=reason)
        )

    async def mark_notified(self, signal_id: int, environment: TradingEnvironment) -> None:
        await self._s.execute(
            update(SignalModel)
            .where(and_(SignalModel.id == signal_id, SignalModel.environment == _env_value(environment)))
            .values(notified=True)
        )

    async def outcome_for_fingerprint(
        self, environment: TradingEnvironment, fingerprint: str
    ) -> Sequence[TradeModel]:
        """What actually happened to past signals that looked like this one."""
        subq = select(SignalModel.trade_id).where(
            and_(
                SignalModel.environment == _env_value(environment),
                SignalModel.fingerprint == fingerprint,
                SignalModel.trade_id.is_not(None),
            )
        )
        stmt = (
            select(TradeModel)
            .where(
                and_(
                    TradeModel.environment == _env_value(environment),
                    TradeModel.id.in_(subq),
                )
            )
            .order_by(TradeModel.closed_at)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def count(self, environment: TradingEnvironment) -> int:
        stmt = select(func.count()).select_from(SignalModel).where(
            SignalModel.environment == _env_value(environment)
        )
        return int((await self._s.execute(stmt)).scalar_one() or 0)


class DailyPnlRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert_day(
        self, environment: TradingEnvironment, trade_date: dt.date
    ) -> DailyPnlModel:
        key = trade_date.isoformat()
        stmt = select(DailyPnlModel).where(
            and_(
                DailyPnlModel.environment == _env_value(environment),
                DailyPnlModel.trade_date == key,
            )
        )
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        if row is None:
            row = DailyPnlModel(environment=_env_value(environment), trade_date=key)
            self._s.add(row)
            await self._s.flush()
        return row

    async def get(self, environment: TradingEnvironment, trade_date: dt.date) -> DailyPnlModel | None:
        stmt = select(DailyPnlModel).where(
            and_(
                DailyPnlModel.environment == _env_value(environment),
                DailyPnlModel.trade_date == trade_date.isoformat(),
            )
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def realized_sum(
        self, environment: TradingEnvironment, since: dt.date, until: dt.date | None = None
    ) -> float:
        stmt = select(func.coalesce(func.sum(DailyPnlModel.realized_pnl), 0.0)).where(
            and_(
                DailyPnlModel.environment == _env_value(environment),
                DailyPnlModel.trade_date >= since.isoformat(),
            )
        )
        if until is not None:
            stmt = stmt.where(DailyPnlModel.trade_date <= until.isoformat())
        return float((await self._s.execute(stmt)).scalar_one() or 0.0)

    async def recent(self, environment: TradingEnvironment, *, days: int = 90) -> Sequence[DailyPnlModel]:
        since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        stmt = (
            select(DailyPnlModel)
            .where(
                and_(
                    DailyPnlModel.environment == _env_value(environment),
                    DailyPnlModel.trade_date >= since,
                )
            )
            .order_by(DailyPnlModel.trade_date)
        )
        return (await self._s.execute(stmt)).scalars().all()


class OrderAuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def log(self, record: OrderAuditModel) -> OrderAuditModel:
        self._s.add(record)
        await self._s.flush()
        return record

    async def recent(
        self, environment: TradingEnvironment, *, limit: int = 50
    ) -> Sequence[OrderAuditModel]:
        stmt = (
            select(OrderAuditModel)
            .where(OrderAuditModel.environment == _env_value(environment))
            .order_by(OrderAuditModel.created_at.desc())
            .limit(limit)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def find_by_idempotency_key(
        self, environment: TradingEnvironment, key: str
    ) -> OrderAuditModel | None:
        stmt = select(OrderAuditModel).where(
            and_(
                OrderAuditModel.environment == _env_value(environment),
                OrderAuditModel.idempotency_key == key,
                OrderAuditModel.status == "SUBMITTED",
            )
        )
        return (await self._s.execute(stmt)).scalars().first()

    async def client_order_exists(
        self, client_order_id: str, environment: TradingEnvironment | None = None
    ) -> bool:
        conditions = [OrderAuditModel.client_order_id == client_order_id]
        if environment is not None:
            conditions.append(OrderAuditModel.environment == _env_value(environment))
        stmt = select(func.count()).select_from(OrderAuditModel).where(and_(*conditions))
        return int((await self._s.execute(stmt)).scalar_one() or 0) > 0

    async def idempotency_key_exists(
        self, environment: TradingEnvironment, key: str
    ) -> bool:
        """True when this exact order intent was already claimed in ``environment``.

        Any status counts (PENDING included): a PENDING claim means a previous
        attempt reached the broker submission step, so replaying the same intent
        is exactly what duplicate protection must stop.
        """
        stmt = select(func.count()).select_from(OrderAuditModel).where(
            and_(
                OrderAuditModel.environment == _env_value(environment),
                OrderAuditModel.idempotency_key == key,
            )
        )
        return int((await self._s.execute(stmt)).scalar_one() or 0) > 0

    async def open_intents_for_symbol(
        self, environment: TradingEnvironment, symbol: str
    ) -> Sequence[OrderAuditModel]:
        stmt = (
            select(OrderAuditModel)
            .where(
                and_(
                    OrderAuditModel.environment == _env_value(environment),
                    OrderAuditModel.symbol == symbol.upper(),
                    OrderAuditModel.status.in_(("PENDING", "SUBMITTED")),
                )
            )
            .order_by(OrderAuditModel.created_at.desc())
        )
        return (await self._s.execute(stmt)).scalars().all()


class AppStateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, key: str, default: Any = None) -> Any:
        stmt = select(AppStateModel).where(AppStateModel.key == key)
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return row.value if row is not None else default

    async def set(self, key: str, value: Any) -> None:
        stmt = select(AppStateModel).where(AppStateModel.key == key)
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        if row is None:
            self._s.add(AppStateModel(key=key, value=value))
        else:
            row.value = value
            row.updated_at = dt.datetime.now(dt.UTC)
        await self._s.flush()

    async def all_state(self) -> dict[str, Any]:
        stmt = select(AppStateModel)
        rows = (await self._s.execute(stmt)).scalars().all()
        return {r.key: r.value for r in rows}

    async def delete(self, key: str) -> None:
        await self._s.execute(delete(AppStateModel).where(AppStateModel.key == key))


class ProtectionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert(
        self, environment: TradingEnvironment, symbol: str, **values: Any
    ) -> PositionProtectionModel:
        symbol = symbol.upper()
        stmt = select(PositionProtectionModel).where(
            and_(
                PositionProtectionModel.environment == _env_value(environment),
                PositionProtectionModel.symbol == symbol,
            )
        )
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        if row is None:
            row = PositionProtectionModel(environment=_env_value(environment), symbol=symbol)
            self._s.add(row)
        for key, val in values.items():
            if hasattr(row, key):
                setattr(row, key, val)
        row.updated_at = dt.datetime.now(dt.UTC)
        await self._s.flush()
        return row

    async def get(
        self, environment: TradingEnvironment, symbol: str
    ) -> PositionProtectionModel | None:
        stmt = select(PositionProtectionModel).where(
            and_(
                PositionProtectionModel.environment == _env_value(environment),
                PositionProtectionModel.symbol == symbol.upper(),
            )
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def all_for(self, environment: TradingEnvironment) -> Sequence[PositionProtectionModel]:
        stmt = select(PositionProtectionModel).where(
            PositionProtectionModel.environment == _env_value(environment)
        )
        return (await self._s.execute(stmt)).scalars().all()

    async def record_initial_stop(
        self, environment: TradingEnvironment, symbol: str, stop_price: float | None
    ) -> bool:
        """Freeze the entry risk the first time it is known; never rewrite it.

        ``upsert`` cannot be used here: it would overwrite the baseline every time
        the stop is ratcheted, which is exactly the value that must stay put.
        Returns True when a value was actually written.
        """
        row = await self.get(environment, symbol)
        if row is None or row.initial_stop_price is not None or stop_price is None:
            return False
        row.initial_stop_price = float(stop_price)
        row.updated_at = dt.datetime.now(dt.UTC)
        await self._s.flush()
        return True

    async def delete(self, environment: TradingEnvironment, symbol: str) -> None:
        await self._s.execute(
            delete(PositionProtectionModel).where(
                and_(
                    PositionProtectionModel.environment == _env_value(environment),
                    PositionProtectionModel.symbol == symbol.upper(),
                )
            )
        )


class SettingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get(self, key: str, scope: str = "GLOBAL", default: Any = None) -> Any:
        stmt = select(SettingModel).where(
            and_(SettingModel.scope == scope, SettingModel.key == key)
        )
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return row.value if row is not None else default

    async def set(self, key: str, value: Any, scope: str = "GLOBAL", updated_by: int | None = None) -> None:
        stmt = select(SettingModel).where(
            and_(SettingModel.scope == scope, SettingModel.key == key)
        )
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        if row is None:
            self._s.add(SettingModel(scope=scope, key=key, value=value, updated_by=updated_by))
        else:
            row.value = value
            row.updated_at = dt.datetime.now(dt.UTC)
            row.updated_by = updated_by
        await self._s.flush()

    async def all_settings(self, scope: str = "GLOBAL") -> dict[str, Any]:
        stmt = select(SettingModel).where(SettingModel.scope == scope)
        rows = (await self._s.execute(stmt)).scalars().all()
        return {r.key: r.value for r in rows}

    async def delete(self, key: str, scope: str = "GLOBAL") -> None:
        await self._s.execute(
            delete(SettingModel).where(and_(SettingModel.scope == scope, SettingModel.key == key))
        )


__all__ = [
    "AppStateRepository",
    "DailyPnlRepository",
    "OrderAuditRepository",
    "ProtectionRepository",
    "SettingRepository",
    "SignalRepository",
    "TradeRepository",
]
