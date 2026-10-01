"""Shared fixtures and test doubles.

Every broker/data client is duck-typed, so no test ever touches the network.
"""

from __future__ import annotations

import datetime as dt
import os
import random
from typing import Any

import pytest

os.environ.setdefault("TELEGRAM_ALLOWED_USER_IDS", "111,222")
os.environ.setdefault("ALPACA_PAPER_API_KEY", "TESTKEY")
os.environ.setdefault("ALPACA_PAPER_SECRET_KEY", "TESTSECRET")
os.environ.setdefault("ALPACA_LIVE_API_KEY", "TESTLIVEKEY")
os.environ.setdefault("ALPACA_LIVE_SECRET_KEY", "TESTLIVESECRET")

from botalpaca.config.settings import Settings  # noqa: E402
from botalpaca.db.session import Database  # noqa: E402
from botalpaca.domain.enums import (  # noqa: E402
    MarketRegime,
    OrderClass,
    OrderSide,
    OrderType,
    Quality,
    SetupType,
    SignalDirection,
    TimeInForce,
    TradingEnvironment,
)
from botalpaca.domain.models import (  # noqa: E402
    Bar,
    IndicatorSet,
    MomentumState,
    Opportunity,
    PositionSnapshot,
    ScoreBreakdown,
    StructureLevel,
    StructureState,
    TechnicalSnapshot,
    TrendState,
    VolatilityState,
    VolumeState,
)

UTC = dt.UTC


# --------------------------------------------------------------------------------------
# Settings / database
# --------------------------------------------------------------------------------------
@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}",
        database_migrate=False,
        database_auto_create=True,
        active_trading_environment=TradingEnvironment.PAPER,
        telegram_allowed_user_ids=[111],
    )


@pytest.fixture
async def database(settings) -> Database:
    db = Database(
        settings.database_url,
        create_all=True,
        migrate=False,
    )
    await db.start()
    try:
        yield db
    finally:
        await db.stop()


# --------------------------------------------------------------------------------------
# Synthetic market data
# --------------------------------------------------------------------------------------
def make_bars(
    n: int = 260,
    start: float = 100.0,
    drift: float = 0.0015,
    noise: float = 1.2,
    seed: int = 7,
    timeframe_minutes: int = 1440,
) -> list[Bar]:
    """Deterministic synthetic OHLCV series."""
    rng = random.Random(seed)
    price = start
    now = dt.datetime(2024, 1, 1, tzinfo=UTC)
    bars: list[Bar] = []
    for i in range(n):
        price = max(1.0, price * (1.0 + drift + rng.gauss(0.0, noise / 100.0)))
        high = price * (1.0 + abs(rng.gauss(0, 0.004)))
        low = price * (1.0 - abs(rng.gauss(0, 0.004)))
        open_ = (high + low) / 2.0
        volume = 1_000_000 + rng.gauss(0, 120_000)
        bars.append(
            Bar(
                timestamp=now + dt.timedelta(minutes=timeframe_minutes * i),
                open=round(open_, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(price, 4),
                volume=max(1000.0, round(volume)),
            )
        )
    return bars


def flat_bars(n: int = 260, price: float = 100.0) -> list[Bar]:
    now = dt.datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Bar(
            timestamp=now + dt.timedelta(days=i),
            open=price,
            high=price + 0.05,
            low=price - 0.05,
            close=price,
            volume=1_000_000.0,
        )
        for i in range(n)
    ]


def make_indicators(**overrides: Any) -> IndicatorSet:
    base: dict[str, Any] = dict(
        timeframe="1D",
        close=100.0,
        ema_9=101.0,
        ema_20=100.5,
        ema_21=100.4,
        ema_50=99.0,
        ema_100=97.0,
        ema_200=95.0,
        sma_20=100.5,
        sma_50=99.0,
        rsi_14=58.0,
        macd=1.2,
        macd_signal=0.8,
        macd_hist=0.4,
        plus_di=25.0,
        minus_di=15.0,
        adx_14=30.0,
        atr_14=2.5,
        atr_pct=2.5,
        bb_upper=104.0,
        bb_middle=100.0,
        bb_lower=96.0,
        obv=1_200_000.0,
        roc_10=2.0,
        vwap=100.2,
        rel_volume=1.2,
        highest_20=104.0,
        lowest_20=96.0,
    )
    base.update(overrides)
    return IndicatorSet(**base)


def make_snapshot(
    symbol: str = "TEST",
    *,
    htf: SignalDirection | None = SignalDirection.LONG,
    data_quality: float = 0.9,
    **ind_over: Any,
) -> TechnicalSnapshot:
    ind = make_indicators(**ind_over)
    now = dt.datetime(2024, 6, 1, tzinfo=UTC)
    return TechnicalSnapshot(
        symbol=symbol,
        timeframe="1D",
        as_of=now,
        indicators=ind,
        trend=TrendState(
            direction=SignalDirection.LONG,
            ema_alignment=3,
            structure="HH_HL",
            strength=0.8,
        ),
        momentum=MomentumState(
            rsi=58.0,
            rsi_state="BULLISH",
            macd_hist=0.4,
            macd_state="BULLISH",
            adx=30.0,
            adx_state="STRONG_TREND",
            roc=2.0,
            divergence="NONE",
            score=0.75,
        ),
        volume=VolumeState(
            rel_volume=1.2,
            volume_state="NORMAL",
            obv_slope=0.5,
            obv_confirming=True,
            score=0.6,
        ),
        volatility=VolatilityState(
            atr=ind.atr_14,
            atr_pct=ind.atr_pct,
            bb_width=8.0,
            bb_width_pctile=60.0,
            squeeze=False,
            expansion=True,
            regime=MarketRegime.TRENDING_UP,
            score=0.6,
        ),
        structure=StructureState(
            supports=[StructureLevel(price=96.0, kind="support", touches=2)],
            resistances=[StructureLevel(price=104.0, kind="resistance", touches=2)],
            breakout=True,
            breakout_direction=SignalDirection.LONG,
            retest=True,
            in_range=False,
        ),
        data_quality=data_quality,
        htf_direction=htf,
        htf_timeframe="1W" if htf is not None else None,
        htf_alignment=1 if htf is not None else None,
    )


def make_opportunity(
    *,
    environment: TradingEnvironment = TradingEnvironment.PAPER,
    symbol: str = "AAPL",
    score: float = 82.0,
    rr: float = 2.0,
    entry: float = 100.0,
    stop: float = 98.0,
    target: float = 104.0,
    tradable: bool = True,
    quality: Quality = Quality.ALTA,
    direction: SignalDirection = SignalDirection.LONG,
    strategy: SetupType = SetupType.PULLBACK,
) -> Opportunity:
    return Opportunity(
        symbol=symbol,
        timeframe="1D",
        environment=environment,
        as_of=dt.datetime.now(UTC),
        direction=direction,
        quality=quality,
        score=score,
        entry=entry,
        stop=stop,
        target=target,
        rr=rr,
        strategy=strategy,
        setup=strategy,
        breakdown=ScoreBreakdown(trend=17, momentum=11, rr=7),
        confluences=["trend", "momentum"],
        reasons=["EMA stack bullish"],
        atr=2.5,
        regime=MarketRegime.TRENDING_UP,
        sector="TECHNOLOGY",
        historical={},
        tradable=tradable,
        non_tradable_reason=None if tradable else "blocked",
        fingerprint="fp-test-AAPL",
    )


# --------------------------------------------------------------------------------------
# Test doubles for the broker
# --------------------------------------------------------------------------------------
class FakeClock:
    def __init__(self, is_open: bool = True) -> None:
        self.is_open = is_open
        self.timestamp = dt.datetime(2024, 6, 3, 14, 30, tzinfo=UTC)
        self.next_open = self.timestamp
        self.next_close = self.timestamp + dt.timedelta(hours=6)

    def is_market_open(self) -> bool:
        return self.is_open


def fake_account_raw(environment: TradingEnvironment = TradingEnvironment.PAPER, **over: Any) -> Any:
    from types import SimpleNamespace

    base: dict[str, Any] = dict(
        id="acc-1",
        account_number="010203ABCD",
        status="ACTIVE",
        crypto_status="ACTIVE",
        currency="USD",
        buying_power=100_000.0,
        regt_buying_power=100_000.0,
        daytrading_buying_power=100_000.0,
        non_marginable_buying_power=50_000.0,
        cash=100_000.0,
        accrued_fees=0.0,
        pending_transfer_out=0.0,
        pending_transfer_in=0.0,
        portfolio_value=100_000.0,
        pattern_day_trader=False,
        trading_blocked=False,
        transfers_blocked=False,
        account_blocked=False,
        created_at=dt.datetime(2023, 1, 1, tzinfo=UTC),
        trade_suspended_by_user=False,
        multiplier=1.0,
        shorting_enabled=True,
        equity=100_000.0,
        last_equity=99_000.0,
        long_market_value=0.0,
        short_market_value=0.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        last_maintenance_margin=0.0,
        sma=0.0,
        daytrade_count=0,
        options_buying_power=0.0,
        options_approved_level=0,
        options_trading_level=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def fake_position_raw(
    symbol: str = "AAPL",
    qty: float = 10.0,
    *,
    entry: float = 100.0,
    current: float = 102.0,
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        asset_id=f"asset-{symbol}",
        symbol=symbol,
        exchange="NASDAQ",
        asset_class="us_equity",
        asset_marginable=True,
        avg_entry_price=entry,
        qty=qty,
        side="long" if qty >= 0 else "short",
        market_value=abs(qty) * current,
        cost_basis=abs(qty) * entry,
        unrealized_pl=abs(qty) * (current - entry) * (1 if qty >= 0 else -1),
        unrealized_plpc=(current - entry) / entry * (1 if qty >= 0 else -1),
        unrealized_intraday_pl=0.0,
        unrealized_intraday_plpc=0.0,
        current_price=current,
        lastday_price=entry,
        change_today=0.0,
        swap_rate=None,
        avg_entry_swap_rate=None,
        usd=None,
        qty_available=qty,
    )


def fake_order_raw(
    order_id: str = "order-1",
    *,
    symbol: str = "AAPL",
    status: str = "accepted",
    order_type: str = "market",
    side: str = "buy",
    qty: float = 10.0,
    filled_qty: float = 10.0,
    filled_avg_price: float = 100.0,
    limit_price: float | None = None,
    stop_price: float | None = None,
    order_class: str = "simple",
    legs: list[Any] | None = None,
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        id=order_id,
        client_order_id=f"cid-{order_id}",
        created_at=dt.datetime(2024, 6, 3, 14, 30, tzinfo=UTC),
        updated_at=dt.datetime(2024, 6, 3, 14, 30, tzinfo=UTC),
        submitted_at=dt.datetime(2024, 6, 3, 14, 30, tzinfo=UTC),
        filled_at=None,
        expired_at=None,
        expires_at=None,
        canceled_at=None,
        failed_at=None,
        replaced_at=None,
        replaced_by=None,
        replaces=None,
        asset_id=f"asset-{symbol}",
        symbol=symbol,
        asset_class="us_equity",
        notional=None,
        qty=qty,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        order_class=order_class,
        order_type=order_type,
        type=order_type,
        side=side,
        time_in_force="day",
        limit_price=limit_price,
        stop_price=stop_price,
        status=status,
        extended_hours=False,
        legs=legs or [],
        trail_percent=None,
        trail_price=None,
        hwm=None,
        position_intent=None,
        ratio_qty=None,
    )


class FakeTradingClient:
    """Duck-typed stand-in for :class:`botalpaca.execution.AlpacaTradingClient`."""

    def __init__(self, environment: TradingEnvironment = TradingEnvironment.PAPER) -> None:
        self.environment_name = environment
        self.environment = environment
        self.base_url = "paper" if environment.is_paper else "live"
        self.submitted: list[Any] = []
        self.cancelled: list[str] = []
        self.replaced: list[tuple[str, Any]] = []
        self.closed: list[tuple[str, dict[str, Any]]] = []
        self.orders: dict[str, Any] = {}
        self.positions: list[Any] = []
        self.account = fake_account_raw(environment)
        self.closed_called = False
        self.submit_error: Exception | None = None
        self._counter = 0
        self.closed_calls: list[dict[str, Any]] = []

    # -- helpers ----------------------------------------------------------------
    def _next_id(self, prefix: str = "order") -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    # -- reads (async, like the real AlpacaTradingClient) ---------------------
    async def get_account(self):
        return self.account

    async def verify(self):
        return self.account

    async def get_account_configurations(self):
        return {}

    async def get_all_positions(self):
        return list(self.positions)

    async def get_position(self, symbol: str):
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    async def get_clock(self):
        return FakeClock()

    async def get_calendar(self, *args, **kwargs):
        return []

    async def get_portfolio_history(self, *args, **kwargs):
        return []

    async def get_asset(self, symbol: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            id=f"asset-{symbol}",
            asset_class="us_equity",
            exchange="NASDAQ",
            symbol=symbol,
            name=symbol,
            status="active",
            tradable=True,
            marginable=True,
            shortable=True,
            easy_to_borrow=True,
            fractionable=True,
            min_order_size=1.0,
            min_trade_increment=0.0001,
            price_increment=0.01,
            maintenance_margin_requirement=30.0,
            attributes=[],
        )

    async def get_all_assets(self, *args, **kwargs):
        return [await self.get_asset("AAPL")]

    # -- writes ----------------------------------------------------------------
    async def submit_order(self, request):
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append(request)
        raw = fake_order_raw(
            self._next_id(),
            symbol=request.symbol,
            order_type=getattr(request, "order_type", None)
            or _enum_str(getattr(request, "type", "market")),
            side=_enum_str(getattr(request, "side", "buy")),
            qty=float(getattr(request, "qty", 0) or 0),
            order_class=_enum_str(getattr(request, "order_class", "simple")),
        )
        self.orders[raw.id] = raw
        return raw

    async def get_order_by_id(self, order_id: str):
        if order_id not in self.orders:
            from botalpaca.execution import BrokerError

            raise BrokerError(f"order {order_id} not found", status_code=404)
        return self.orders[order_id]

    async def get_order_by_client_id(self, client_order_id: str):
        for raw in self.orders.values():
            if raw.client_order_id == client_order_id:
                return raw
        from botalpaca.execution import BrokerError

        raise BrokerError("not found", status_code=404)

    async def get_orders(self, **kwargs):
        """Mirror Alpaca: ``status="open"`` hides cancelled/filled orders."""
        status = str(getattr(kwargs.get("status", "open"), "value", kwargs.get("status", "open")))
        orders = list(self.orders.values())
        if status in ("open", "QueryOrderStatus.OPEN"):
            return [o for o in orders if getattr(o, "status", "new") in ("new", "accepted", "pending_new")]
        return orders

    async def cancel_order_by_id(self, order_id: str):
        self.cancelled.append(order_id)
        raw = self.orders.get(order_id)
        if raw is not None:
            raw.status = "canceled"
        return raw or fake_order_raw(order_id, status="canceled")

    async def cancel_orders(self, *, symbol: str | None = None):
        ids = [o.id for o in self.orders.values() if symbol is None or o.symbol == symbol]
        for i in ids:
            self.cancelled.append(i)
        return [await self.get_order_by_id(i) for i in ids]

    async def replace_order_by_id(self, order_id: str, request):
        self.replaced.append((order_id, request))
        raw = self.orders.get(order_id)
        if raw is None:
            raw = fake_order_raw(order_id)
            self.orders[order_id] = raw
        raw.limit_price = getattr(request, "limit_price", None)
        raw.stop_price = getattr(request, "stop_price", None)
        raw.trail_percent = getattr(request, "trail", None)
        raw.qty = getattr(request, "qty", None) or raw.qty
        raw.replaced_at = dt.datetime.now(UTC)
        return raw

    async def close_position(self, symbol: str, **kwargs):
        self.closed.append((symbol, kwargs))
        self.closed_calls.append(dict(kwargs, symbol=symbol))
        return fake_order_raw(self._next_id("close"), symbol=symbol, side="sell")

    async def close_all_positions(self, *, cancel_orders: bool = True):
        for p in list(self.positions):
            self.closed.append((p.symbol, {}))
        self.closed_called = True

    def close(self) -> None:
        return None


def _enum_str(value: Any) -> str:
    return getattr(value, "value", value)


# --------------------------------------------------------------------------------------
# Fake market data service
# --------------------------------------------------------------------------------------
class FakeMarketService:
    def __init__(self, bars: list[Bar] | None = None) -> None:
        self.bars = bars if bars is not None else make_bars()
        self.clock = FakeClock()
        self.calls: list[tuple[str, str]] = []
        self.quote_price = self.bars[-1].close if self.bars else 100.0

    async def get_bars(self, symbol: str, timeframe: str = "1D", **kwargs):
        self.calls.append((symbol, timeframe))
        return list(self.bars)

    async def get_bars_multi(self, symbols, timeframe: str = "1D", **kwargs):
        return {s: list(self.bars) for s in symbols}

    async def get_quote(self, symbol: str):
        from botalpaca.domain.models import Quote

        return Quote(
            symbol=symbol,
            timestamp=dt.datetime(2024, 6, 3, 14, 30, tzinfo=UTC),
            bid_price=self.quote_price - 0.01,
            ask_price=self.quote_price + 0.01,
            bid_size=100.0,
            ask_size=100.0,
        )

    async def get_quotes(self, symbols):
        return [await self.get_quote(s) for s in symbols]

    async def get_last_price(self, symbol: str) -> float:
        return self.quote_price

    async def get_snapshot(self, symbol: str):
        from types import SimpleNamespace

        return SimpleNamespace(symbol=symbol)

    async def get_clock(self):
        return self.clock

    async def is_market_open(self) -> bool:
        return self.clock.is_open

    async def avg_dollar_volume(self, symbol: str, timeframe: str = "1D", *, bars: int = 30) -> float:
        window = self.bars[-bars:]
        if not window:
            return 0.0
        return sum(b.close * b.volume for b in window) / len(window)

    async def close(self) -> None:
        return None


class FakePortfolioService:
    """Duck-typed stand-in for :class:`botalpaca.portfolio.PortfolioService`."""

    def __init__(
        self,
        client: FakeTradingClient,
        *,
        equity: float = 100_000.0,
        positions: list[PositionSnapshot] | None = None,
    ) -> None:
        self.client = client
        self.equity = equity
        self._positions = positions or []

    def _account(self):
        self.client.account.equity = self.equity
        self.client.account.portfolio_value = self.equity
        self.client.account.buying_power = self.equity * 2
        return self.client.account

    async def get_account(self):
        return self._account()

    async def verify_connection(self):
        return self._account()

    async def get_positions(self) -> list[PositionSnapshot]:
        if self._positions:
            return list(self._positions)
        from botalpaca.execution.mapping import to_position

        return [to_position(p, self.client.environment_name) for p in self.client.positions]

    async def get_position(self, symbol: str):
        for p in await self.get_positions():
            if p.symbol == symbol:
                return p
        return None

    async def get_orders(self, **kwargs):
        from botalpaca.execution.mapping import to_order_state

        return [to_order_state(o, self.client.environment_name) for o in await self.client.get_orders()]

    async def get_open_orders(self, **kwargs):
        return await self.get_orders(**kwargs)

    async def get_order(self, order_id: str):
        from botalpaca.execution.mapping import to_order_state

        return to_order_state(
            await self.client.get_order_by_id(order_id), self.client.environment_name
        )

    async def get_asset_info(self, symbol: str):
        from types import SimpleNamespace

        return SimpleNamespace(**vars(await self.client.get_asset(symbol)))

    async def get_clock(self):
        return await self.client.get_clock()

    async def portfolio_history(self, *args, **kwargs):
        return []

    async def exposures(self):
        positions = await self.get_positions()
        long_mv = sum(p.market_value for p in positions if p.qty >= 0)
        short_mv = sum(abs(p.market_value) for p in positions if p.qty < 0)
        return {
            "environment": self.client.environment_name,
            "as_of": dt.datetime.now(UTC),
            "equity": self.equity,
            "cash": self.equity,
            "buying_power": self.equity * 2,
            "long_market_value": long_mv,
            "short_market_value": short_mv,
            "gross_exposure": long_mv + short_mv,
            "gross_exposure_pct": (long_mv + short_mv) / self.equity * 100 if self.equity else 0.0,
            "net_exposure_pct": (long_mv - short_mv) / self.equity * 100 if self.equity else 0.0,
            "unrealized_pl": sum(p.unrealized_pl for p in positions),
            "position_count": len(positions),
            "positions": positions,
        }


def make_position(
    symbol: str = "AAPL",
    *,
    qty: float = 10.0,
    entry: float = 100.0,
    current: float = 102.0,
    environment: TradingEnvironment = TradingEnvironment.PAPER,
) -> PositionSnapshot:
    return PositionSnapshot(
        environment=environment,
        symbol=symbol,
        qty=qty,
        side="long" if qty >= 0 else "short",
        avg_entry_price=entry,
        current_price=current,
        market_value=abs(qty) * current,
        cost_basis=abs(qty) * entry,
        unrealized_pl=abs(qty) * (current - entry) * (1 if qty >= 0 else -1),
        unrealized_plpc=(current - entry) / entry * (1 if qty >= 0 else -1),
        qty_available=qty,
    )


@pytest.fixture(autouse=True)
def _no_sleep():
    """Keep the monitor loops fast and deterministic where they are exercised."""
    yield


__all__ = [
    "UTC",
    "FakeMarketService",
    "FakePortfolioService",
    "FakeTradingClient",
    "OrderClass",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "flat_bars",
    "fake_account_raw",
    "fake_order_raw",
    "fake_position_raw",
    "make_bars",
    "make_indicators",
    "make_opportunity",
    "make_position",
    "make_snapshot",
]
