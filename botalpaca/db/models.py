"""SQLAlchemy ORM models.

Design rules enforced here:

1. Every financial row carries a non-null ``environment`` column. There is no
   code path that queries trade/stat tables without filtering on it, so PAPER
   and REAL can never bleed into each other.
2. Secrets are never persisted.
3. Indicator snapshots are stored as JSON so the journal can answer
   "why did you recommend this?" months later without a schema migration.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from botalpaca.domain import MarketRegime, OrderType, TradeStatus


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[str]: JSON}


class TradeModel(Base):
    """One trade, always scoped to exactly one environment."""

    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_env_symbol", "environment", "symbol"),
        Index("ix_trades_env_status", "environment", "status"),
        Index("ix_trades_env_strategy", "environment", "strategy"),
        Index("ix_trades_env_opened", "environment", "opened_at"),
        Index("ix_trades_env_closed", "environment", "closed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, index=True)

    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    setup: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="1D")
    sector: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    regime: Mapped[str] = mapped_column(String(24), nullable=False, default=MarketRegime.UNKNOWN.value)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default=TradeStatus.OPEN.value)

    qty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)

    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rr: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    r_multiple: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mfe: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mae: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mfe_r: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mae_r: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    opened_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    # None while the entry order is still waiting to fill (market closed); set
    # once Alpaca reports a fill. Kept out of ``status`` on purpose so pending
    # trades still count towards exposure limits.
    filled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    indicators: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    confluences: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    entry_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_reason_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    entry_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exit_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    stop_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    take_profit_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trailing_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    has_stop: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    time_stop_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )

    events: Mapped[list[TradeEventModel]] = relationship(
        back_populates="trade", cascade="all, delete-orphan", lazy="selectin"
    )


class TradeEventModel(Base):
    """Append-only journal of every state change for a trade."""

    __tablename__ = "trade_events"
    __table_args__ = (Index("ix_trade_events_trade", "trade_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(
        ForeignKey("trades.id", ondelete="CASCADE"), nullable=False
    )
    environment: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    trade: Mapped[TradeModel] = relationship(back_populates="events")


class SignalModel(Base):
    """Every signal the engine produces, accepted or not.

    Recording rejected signals is what allows the learning engine to answer
    "was this a good call we skipped?" without hindsight bias in the P&L.
    """

    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signals_env_symbol", "environment", "symbol"),
        Index("ix_signals_env_created", "environment", "created_at"),
        Index("ix_signals_fingerprint", "fingerprint"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(8), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    # Higher-timeframe confirmation timeframe, kept so multi-timeframe
    # statistics can be grouped by it. ``None`` means "unknown / not checked".
    htf_timeframe: Mapped[str | None] = mapped_column(String(8), nullable=True)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    setup: Mapped[str] = mapped_column(String(32), nullable=False)
    quality: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rr: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    entry: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stop: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    target: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[str] = mapped_column(String(24), nullable=False, default=MarketRegime.UNKNOWN.value)
    sector: Mapped[str | None] = mapped_column(String(48), nullable=True)
    tradable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    non_tradable_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    accepted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    rejected_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    trade_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    confluences: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    breakdown: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    indicators: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    fingerprint: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    notified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class DailyPnlModel(Base):
    """Realized P&L per environment per day, the basis of loss limits."""

    __tablename__ = "daily_pnl"
    __table_args__ = (UniqueConstraint("environment", "trade_date", name="uq_daily_env_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    win_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    loss_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gross_profit: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    gross_loss: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class OrderAuditModel(Base):
    """Immutable audit trail of every order intent and broker response."""

    __tablename__ = "order_audit"
    __table_args__ = (
        Index("ix_order_audit_env_created", "environment", "created_at"),
        Index("ix_order_audit_client_id", "client_order_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(8), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    # ``symbol``/``side`` are nullable: an audit row for "cancel order <id>" or
    # "release kill switch" legitimately has neither until the broker replies.
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    order_type: Mapped[str] = mapped_column(String(16), nullable=False, default=OrderType.MARKET.value)
    order_class: Mapped[str | None] = mapped_column(String(16), nullable=True)
    qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confirmed_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    response: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class AppStateModel(Base):
    """Durable key/value runtime state (active env, kill switch, counters).

    Stored in SQLite rather than memory so a Fly.io restart cannot silently
    revert the trader into a different environment or a disabled kill switch.
    """

    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class PositionProtectionModel(Base):
    """Current protective orders per position, for reconciliation and recovery."""

    __tablename__ = "position_protection"
    __table_args__ = (UniqueConstraint("environment", "symbol", name="uq_protection_env_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    initial_stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    """The stop the trade was opened with, written once and never rewritten.

    ``stop_price`` moves as the position is protected, so it cannot serve as the
    denominator of an R multiple: a trade already ratcheted to break-even would
    measure its own profit as risk. This is the frozen baseline every R
    comparison is made against.
    """
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    trail_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    trail_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    take_profit_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trailing_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    break_even_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    order_class: Mapped[str | None] = mapped_column(String(16), nullable=True)
    time_stop_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class SettingModel(Base):
    """Runtime-tunable settings changed from Telegram /config."""

    __tablename__ = "settings_override"
    __table_args__ = (UniqueConstraint("scope", "key", name="uq_setting_scope_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(16), nullable=False, default="GLOBAL")
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_by: Mapped[int | None] = mapped_column(Integer, nullable=True)


__all__ = [
    "AppStateModel",
    "Base",
    "DailyPnlModel",
    "OrderAuditModel",
    "PositionProtectionModel",
    "SettingModel",
    "SignalModel",
    "TradeEventModel",
    "TradeModel",
]
