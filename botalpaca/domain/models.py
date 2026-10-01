"""Core domain models.

Pydantic models used across layers. These carry no framework-specific
attributes so they can be created in tests, persisted, serialized, and
rendered for Telegram without extra mapping layers.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ExitReason,
    MarketRegime,
    OrderClass,
    OrderSide,
    OrderType,
    Quality,
    SetupType,
    SignalDirection,
    TimeInForce,
    TradeStatus,
    TradingEnvironment,
)


class Domain(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid", populate_by_name=True)


class Bar(Domain):
    """A single OHLCV candle."""

    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Quote(Domain):
    timestamp: dt.datetime
    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def spread_pct(self) -> float:
        if self.mid <= 0:
            return 0.0
        return self.spread / self.mid * 100.0


class IndicatorSet(Domain):
    """Raw indicator values for one symbol/timeframe."""

    timeframe: str
    close: float
    ema_9: float | None = None
    ema_20: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    ema_100: float | None = None
    ema_200: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    adx_14: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    obv: float | None = None
    roc_10: float | None = None
    vwap: float | None = None
    volume_sma_20: float | None = None
    rel_volume: float | None = None
    highest_20: float | None = None
    lowest_20: float | None = None
    highest_52w: float | None = None
    lowest_52w: float | None = None


class TrendState(Domain):
    direction: SignalDirection | None = None
    ema_alignment: int = 0  # -4..+4 stacked EMA order strength
    price_above_ema20: bool | None = None
    price_above_ema50: bool | None = None
    price_above_ema200: bool | None = None
    ema20_slope: float | None = None
    ema50_slope: float | None = None
    golden_cross: bool | None = None
    death_cross: bool | None = None
    structure: str | None = None  # HH/HL, LH/LL classification
    strength: float = 0.0  # 0..1
    notes: list[str] = Field(default_factory=list)


class MomentumState(Domain):
    rsi: float | None = None
    rsi_state: str | None = None  # overshoot/neutral/overbought
    macd_hist: float | None = None
    macd_state: str | None = None
    adx: float | None = None
    adx_state: str | None = None
    roc: float | None = None
    divergence: str | None = None
    score: float = 0.0
    notes: list[str] = Field(default_factory=list)


class VolumeState(Domain):
    rel_volume: float | None = None
    volume_state: str | None = None  # spike/normal/dry
    obv_slope: float | None = None
    obv_confirming: bool | None = None
    score: float = 0.0
    notes: list[str] = Field(default_factory=list)


class VolatilityState(Domain):
    atr: float | None = None
    atr_pct: float | None = None
    bb_width: float | None = None
    bb_width_pctile: float | None = None
    squeeze: bool | None = None
    expansion: bool | None = None
    regime: MarketRegime = MarketRegime.UNKNOWN
    score: float = 0.0
    notes: list[str] = Field(default_factory=list)


class StructureLevel(Domain):
    price: float
    kind: Literal["support", "resistance"]
    touches: int = 1
    last_touch: dt.datetime | None = None


class StructureState(Domain):
    supports: list[StructureLevel] = Field(default_factory=list)
    resistances: list[StructureLevel] = Field(default_factory=list)
    breakout: bool = False
    breakout_direction: SignalDirection | None = None
    retest: bool = False
    in_range: bool = False
    gap_up: bool = False
    gap_down: bool = False
    notes: list[str] = Field(default_factory=list)


class MarketContext(Domain):
    """Broad market / sector / benchmark context."""

    benchmark: str = "SPY"
    benchmark_trend: SignalDirection | None = None
    benchmark_regime: MarketRegime = MarketRegime.UNKNOWN
    benchmark_change_pct: float | None = None
    sector: str | None = None
    sector_trend: SignalDirection | None = None
    rs_vs_benchmark: float | None = None  # relative strength %
    market_volatility_pct: float | None = None
    correlation_to_benchmark: float | None = None
    is_market_open: bool = False
    notes: list[str] = Field(default_factory=list)


class TechnicalSnapshot(Domain):
    """Everything the scoring engine needs for one symbol."""

    symbol: str
    timeframe: str
    as_of: dt.datetime
    indicators: IndicatorSet
    trend: TrendState
    momentum: MomentumState
    volume: VolumeState
    volatility: VolatilityState
    structure: StructureState
    context: MarketContext = Field(default_factory=MarketContext)
    quote: Quote | None = None
    data_quality: float = 0.0  # 0..1 confidence in the data itself
    # Higher-timeframe verdict, populated by the scanner when that data is
    # available. ``None`` means "unknown", and multi-timeframe strategies
    # abstain rather than assuming agreement.
    htf_direction: SignalDirection | None = None
    htf_timeframe: str | None = None
    htf_alignment: int | None = None


class StrategySignal(Domain):
    """A single strategy's verdict on a symbol."""

    strategy: SetupType
    direction: SignalDirection
    score: float  # 0..100 from this strategy alone
    entry: float
    stop: float
    timeframe: str = "1D"
    targets: list[float] = Field(default_factory=list)
    rr: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    invalidation: str | None = None
    confluences: list[str] = Field(default_factory=list)
    historical_win_rate: float | None = None
    historical_sample: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(Domain):
    trend: float = 0.0
    momentum: float = 0.0
    volume: float = 0.0
    volatility: float = 0.0
    structure: float = 0.0
    breakout: float = 0.0
    multi_timeframe: float = 0.0
    market: float = 0.0
    sector: float = 0.0
    liquidity: float = 0.0
    rr: float = 0.0
    risk: float = 0.0
    strategy_history: float = 0.0
    symbol_history: float = 0.0
    setup_history: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class Opportunity(Domain):
    """A scored, tradable (or not) setup."""

    symbol: str
    timeframe: str
    environment: TradingEnvironment
    as_of: dt.datetime
    direction: SignalDirection
    quality: Quality
    score: float  # 0..100
    entry: float
    stop: float
    target: float
    rr: float
    strategy: SetupType
    setup: SetupType
    breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    confluences: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    invalidation: str | None = None
    atr: float | None = None
    regime: MarketRegime = MarketRegime.UNKNOWN
    sector: str | None = None
    historical: dict[str, Any] = Field(default_factory=dict)
    signals: list[StrategySignal] = Field(default_factory=list)
    tradable: bool = True
    non_tradable_reason: str | None = None
    fingerprint: str = ""


class TradePlan(Domain):
    """A concrete, risk-sized, execution-ready trade proposal."""

    symbol: str
    environment: TradingEnvironment
    direction: SignalDirection
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    qty: float = 0.0
    notional: float | None = None
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    trail_percent: float | None = None
    trail_price: float | None = None
    order_class: OrderClass = OrderClass.SIMPLE
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    rr: float = 0.0
    score: float = 0.0
    strategy: SetupType | None = None
    setup: SetupType | None = None
    timeframe: str = "1D"
    opportunity_fingerprint: str | None = None
    notes: list[str] = Field(default_factory=list)


class RiskAssessment(Domain):
    approved: bool
    reasons: list[str] = Field(default_factory=list)
    blocks: list[str] = Field(default_factory=list)
    risk_per_trade_pct: float = 0.0
    max_risk_amount: float = 0.0
    suggested_qty: float = 0.0
    suggested_notional: float | None = None
    stop_distance_pct: float = 0.0
    estimated_slippage: float = 0.0
    exposure_pct: float = 0.0
    score: float = 0.0


class AccountSnapshot(Domain):
    environment: TradingEnvironment
    account_id: str
    account_number: str = ""
    status: str = "unknown"
    currency: str = "USD"
    equity: float = 0.0
    last_equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    non_marginable_buying_power: float = 0.0
    daytrading_buying_power: float = 0.0
    regt_buying_power: float = 0.0
    long_market_value: float = 0.0
    short_market_value: float = 0.0
    portfolio_value: float = 0.0
    daytrade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False
    trade_suspended: bool = False
    shorting_enabled: bool = False
    as_of: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class PositionSnapshot(Domain):
    environment: TradingEnvironment
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    unrealized_intraday_pl: float = 0.0
    unrealized_intraday_plpc: float = 0.0
    change_today: float = 0.0
    qty_available: float = 0.0
    exchange: str = ""
    asset_class: str = "us_equity"

    @property
    def direction(self) -> SignalDirection:
        return SignalDirection.LONG if self.qty >= 0 else SignalDirection.SHORT


class OrderLeg(Domain):
    symbol: str
    qty: float
    side: OrderSide
    order_type: OrderType


class OrderState(Domain):
    """Normalized representation of an Alpaca order."""

    id: str
    client_order_id: str | None = None
    environment: TradingEnvironment
    symbol: str
    qty: float | None = None
    notional: float | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    side: OrderSide
    order_type: OrderType
    time_in_force: TimeInForce
    order_class: OrderClass = OrderClass.SIMPLE
    status: str
    limit_price: float | None = None
    stop_price: float | None = None
    trail_percent: float | None = None
    trail_price: float | None = None
    extended_hours: bool = False
    created_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None
    legs: list[OrderLeg] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class FillRecord(Domain):
    environment: TradingEnvironment
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    timestamp: dt.datetime | None = None
    fee: float = 0.0


class TradeRecord(Domain):
    """A trade as tracked internally, scoped to exactly one environment."""

    id: int | None = None
    environment: TradingEnvironment
    symbol: str
    direction: SignalDirection
    strategy: SetupType
    setup: SetupType
    timeframe: str
    status: TradeStatus = TradeStatus.OPEN
    qty: float = 0.0
    entry_price: float = 0.0
    stop_price: float | None = None
    target_price: float | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None
    score: float = 0.0
    rr: float = 0.0
    risk_amount: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    atr: float | None = None
    regime: MarketRegime = MarketRegime.UNKNOWN
    sector: str | None = None
    volume: float | None = None
    opened_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    closed_at: dt.datetime | None = None
    duration_seconds: int | None = None
    indicators: dict[str, Any] = Field(default_factory=dict)
    confluences: list[str] = Field(default_factory=list)
    entry_reason: str | None = None
    exit_reason_note: str | None = None
    entry_order_id: str | None = None
    exit_order_id: str | None = None
    client_order_id: str | None = None
    stop_order_id: str | None = None
    take_profit_order_id: str | None = None
    trailing_order_id: str | None = None
    has_stop: bool = False
    time_stop_at: dt.datetime | None = None
    metadata_: dict[str, Any] = Field(default_factory=dict, alias="metadata")


class ProtectionState(Domain):
    """The live protective orders attached to one position."""

    symbol: str
    environment: TradingEnvironment
    has_stop: bool = False
    has_take_profit: bool = False
    has_trailing: bool = False
    stop_price: float | None = None
    take_profit_price: float | None = None
    trail_percent: float | None = None
    trail_price: float | None = None
    stop_order_id: str | None = None
    take_profit_order_id: str | None = None
    trailing_order_id: str | None = None
    break_even_active: bool = False
    time_stop_at: dt.datetime | None = None
    order_class: OrderClass | None = None
    notes: list[str] = Field(default_factory=list)


class StatisticalSummary(Domain):
    """Aggregate performance for a group of trades."""

    label: str
    sample_size: int
    win_rate: float
    profit_factor: float | None = None
    expectancy_r: float | None = None
    average_r: float | None = None
    average_win: float | None = None
    average_loss: float | None = None
    max_drawdown_r: float | None = None
    recovery_factor: float | None = None
    sharpe: float | None = None
    max_consecutive_wins: int | None = None
    max_consecutive_losses: int | None = None
    total_pnl: float = 0.0
    avg_mfe_r: float | None = None
    avg_mae_r: float | None = None
    avg_hold_minutes: float | None = None
    is_significant: bool = False
    caveat: str | None = None


class SymbolStats(Domain):
    symbol: str
    environment: TradingEnvironment
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_r: float
    best_setup: str | None = None
    best_setup_win_rate: float | None = None
    weak_condition: str | None = None
    caveat: str | None = None


class HealthStatus(Domain):
    database_ok: bool
    alpaca_ok: bool
    active_environment: TradingEnvironment
    scheduler_ok: bool
    last_scan_at: dt.datetime | None = None
    last_reconcile_at: dt.datetime | None = None
    kill_switch: bool = False
    uptime_seconds: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AccountSnapshot",
    "Bar",
    "FillRecord",
    "HealthStatus",
    "IndicatorSet",
    "MarketContext",
    "MomentumState",
    "Opportunity",
    "OrderLeg",
    "OrderState",
    "PositionSnapshot",
    "ProtectionState",
    "Quality",
    "Quote",
    "RiskAssessment",
    "ScoreBreakdown",
    "StatisticalSummary",
    "StrategySignal",
    "StructureLevel",
    "StructureState",
    "SymbolStats",
    "TechnicalSnapshot",
    "TradePlan",
    "TradeRecord",
    "TrendState",
    "VolatilityState",
    "VolumeState",
]
