"""Market data access.

Alpaca's ``alpaca-py`` data clients are synchronous, so every call is wrapped
in ``asyncio.to_thread`` and executed with a timeout plus retry-with-backoff.
This keeps the whole application async while never blocking the event loop on
a network round trip.

Market data is shared across PAPER and REAL (prices are the same feed); only
*financial* state is isolated. The credential bundle used here is therefore the
one of the active environment, purely so the data client is configured the same
way as the trading client.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence
from typing import Any

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient

from botalpaca.config import AlpacaEnvironmentConfig, get_logger
from botalpaca.domain import Bar, Quote, TradingEnvironment

logger = get_logger(__name__)

__all__ = [
    "MarketDataService",
    "TimeframeSpec",
    "parse_timeframe",
    "to_bar",
    "to_quote",
]

#: Headroom over the theoretical span, so weekends, holidays and halts do not
#: leave the last few bars short of the requested count.
_HISTORY_HEADROOM = 2.5


def _default_start(bar_minutes: int, limit: int) -> dt.datetime:
    """Earliest timestamp needed to obtain ``limit`` bars of this timeframe.

    Alpaca's ``/v2/stocks/bars`` endpoint does NOT honour ``limit`` on its own:
    a request with ``limit=300`` and no ``start`` returns a single bar. Every
    call must therefore carry an explicit window, so this derives one from the
    bar size and the requested count. Without it the whole scanner silently
    analysed nothing.
    """
    minutes = max(1, bar_minutes) * max(1, limit) * _HISTORY_HEADROOM
    return dt.datetime.now(dt.UTC) - dt.timedelta(minutes=minutes)

# Minutes per unit, used to decide how much history a timeframe needs.
_UNIT_MINUTES: dict[str, int] = {
    TimeFrameUnit.Minute.value: 1,
    TimeFrameUnit.Hour.value: 60,
    TimeFrameUnit.Day.value: 60 * 24,
    TimeFrameUnit.Week.value: 60 * 24 * 7,
    TimeFrameUnit.Month.value: 60 * 24 * 30,
}


class TimeframeSpec(tuple):
    """``(amount, unit)`` with Alpaca's enums, plus a human label."""

    __slots__ = ()

    @property
    def amount(self) -> int:
        return self[0]

    @property
    def unit(self) -> TimeFrameUnit:
        return self[1]

    @property
    def label(self) -> str:
        amount, unit = self
        if unit is TimeFrameUnit.Minute:
            return f"{amount}m"
        if unit is TimeFrameUnit.Hour:
            return f"{amount}h"
        if unit is TimeFrameUnit.Day:
            return f"{amount}d"
        if unit is TimeFrameUnit.Week:
            return f"{amount}w"
        return f"{amount}mo"

    @property
    def minutes(self) -> int:
        amount, unit = self
        return amount * _UNIT_MINUTES[unit.value]


def parse_timeframe(timeframe: str) -> TimeframeSpec:
    """Parse ``1D``/``15Min``/``4h``/``1w`` into Alpaca's enums.

    Raises ``ValueError`` for anything the Alpaca API does not support, so an
    invalid timeframe can never silently produce an empty bar series.
    """
    raw = timeframe.strip().lower()
    if not raw:
        raise ValueError("timeframe must not be empty")

    for suffix, unit in (
        ("mo", TimeFrameUnit.Month),
        ("w", TimeFrameUnit.Week),
        ("d", TimeFrameUnit.Day),
        ("h", TimeFrameUnit.Hour),
        ("min", TimeFrameUnit.Minute),
        ("m", TimeFrameUnit.Minute),
    ):
        if raw.endswith(suffix):
            digits = raw[: -len(suffix)]
            if digits.isdigit():
                amount = int(digits)
                if amount < 1:
                    raise ValueError(f"timeframe amount must be >= 1: {timeframe!r}")
                return TimeframeSpec((amount, unit))

    if raw.isdigit():
        return TimeframeSpec((int(raw), TimeFrameUnit.Minute))

    raise ValueError(f"unsupported timeframe: {timeframe!r}")


def to_bar(raw: Any) -> Bar:
    return Bar(
        timestamp=_as_utc(raw.timestamp),
        open=float(raw.open),
        high=float(raw.high),
        low=float(raw.low),
        close=float(raw.close),
        volume=float(raw.volume),
    )


def to_quote(raw: Any) -> Quote:
    return Quote(
        timestamp=_as_utc(raw.timestamp),
        symbol=str(raw.symbol),
        bid=float(raw.bid_price),
        ask=float(raw.ask_price),
        bid_size=float(raw.bid_size or 0),
        ask_size=float(raw.ask_size or 0),
    )


def _as_utc(value: dt.datetime | None) -> dt.datetime:
    if value is None:
        return dt.datetime.now(dt.UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


class MarketDataService:
    """Async facade over Alpaca's historical and latest-quote endpoints."""

    def __init__(
        self,
        config: AlpacaEnvironmentConfig,
        *,
        environment: TradingEnvironment,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        feed: DataFeed = DataFeed.IEX,
        trading_client: TradingClient | None = None,
    ) -> None:
        self.environment = environment
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.feed = feed
        config.require_configured()
        self._api_key = config.api_key.get_secret_value()
        self._secret_key = config.secret_key.get_secret_value()
        self._data_url = config.data_url or None
        # Lazily built so importing this module never opens a socket.
        self._data_client: StockHistoricalDataClient | None = None
        self._trading_client = trading_client

    # -- client construction -------------------------------------------------

    def _client(self) -> StockHistoricalDataClient:
        if self._data_client is None:
            self._data_client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                url_override=self._data_url,
            )
        return self._data_client

    def close(self) -> None:
        """Drop cached clients so a credential rotation takes effect."""
        self._data_client = None
        self._trading_client = None

    # -- primitive calls -----------------------------------------------------

    async def _run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking Alpaca call off the event loop with retry/backoff."""
        attempt = 0
        delay = self.backoff_seconds
        last_exc: Exception | None = None
        while attempt <= self.max_retries:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs), timeout=self.timeout_seconds
                )
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
            attempt += 1
            if attempt > self.max_retries:
                break
            logger.warning(
                "market_data.retry",
                attempt=attempt,
                delay=delay,
                error=str(last_exc),
                environment=self.environment.value,
            )
            await asyncio.sleep(delay)
            delay *= 2
        raise DataRequestError(f"Alpaca data call failed after {attempt} attempts: {last_exc}")

    # -- public API ----------------------------------------------------------

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1D",
        *,
        limit: int = 300,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        adjustment: Adjustment = Adjustment.SPLIT,
    ) -> list[Bar]:
        """Historical OHLCV bars, oldest first.

        Returns an empty list when Alpaca has no bars for the window; callers
        decide whether that is a data-quality problem.
        """
        tf = parse_timeframe(timeframe)
        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=TimeFrame(tf.amount, tf.unit),
            limit=limit,
            start=start if start is not None else _default_start(tf.minutes, limit),
            end=end,
            adjustment=adjustment,
            feed=self.feed,
        )
        client = self._client()
        payload = await self._run(client.get_stock_bars, request)
        bars = [to_bar(b) for b in payload[symbol.upper()]]
        bars.sort(key=lambda b: b.timestamp)
        return bars

    async def get_bars_multi(
        self, symbols: Sequence[str], timeframe: str = "1D", *, limit: int = 300
    ) -> dict[str, list[Bar]]:
        """Bars for several symbols in one request (Alpaca batches these)."""
        if not symbols:
            return {}
        upper = [s.upper() for s in symbols]
        tf = parse_timeframe(timeframe)
        request = StockBarsRequest(
            symbol_or_symbols=upper,
            timeframe=TimeFrame(tf.amount, tf.unit),
            limit=limit,
            start=_default_start(tf.minutes, limit),
            adjustment=Adjustment.SPLIT,
            feed=self.feed,
        )
        client = self._client()
        payload = await self._run(client.get_stock_bars, request)
        result: dict[str, list[Bar]] = {}
        for symbol, raw_bars in payload.items():
            bars = sorted((to_bar(b) for b in raw_bars), key=lambda b: b.timestamp)
            result[symbol.upper()] = bars
        return result

    async def get_quote(self, symbol: str) -> Quote | None:
        """Latest NBBO quote, or ``None`` when the symbol has no quote."""
        client = self._client()
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol.upper(), feed=self.feed)
        try:
            payload = await self._run(client.get_stock_latest_quote, request)
        except DataRequestError:
            logger.warning("market_data.quote.unavailable", symbol=symbol.upper())
            return None
        raw = payload[symbol.upper()]
        if raw is None:
            return None
        return to_quote(raw)

    async def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        if not symbols:
            return {}
        upper = [s.upper() for s in symbols]
        client = self._client()
        request = StockLatestQuoteRequest(symbol_or_symbols=upper, feed=self.feed)
        payload = await self._run(client.get_stock_latest_quote, request)
        out: dict[str, Quote] = {}
        for symbol, raw in payload.items():
            if raw is not None:
                out[symbol.upper()] = to_quote(raw)
        return out

    async def get_last_price(self, symbol: str) -> float | None:
        """Latest trade price, falling back to the daily-bar close."""
        client = self._client()
        upper = symbol.upper()
        try:
            trade = await self._run(
                client.get_stock_latest_trade,
                StockLatestTradeRequest(symbol_or_symbols=upper, feed=self.feed),
            )
        except DataRequestError:
            trade = None
        raw_trade = trade.get(upper) if isinstance(trade, dict) else None
        if raw_trade is not None and raw_trade.price:
            return float(raw_trade.price)

        snapshot = await self.get_snapshot(upper)
        if snapshot and snapshot.daily_bar:
            return float(snapshot.daily_bar.close)
        return None

    async def get_snapshot(self, symbol: str) -> Any | None:
        client = self._client()
        upper = symbol.upper()
        try:
            payload = await self._run(
                client.get_stock_snapshot, StockSnapshotRequest(symbol_or_symbols=upper)
            )
        except DataRequestError:
            return None
        return payload.get(upper) if isinstance(payload, dict) else None

    async def get_clock(self) -> Any:
        """Market clock. Requires the trading endpoint, hence a trading client."""
        client = self._trading_client
        if client is None:
            client = TradingClient(
                self._api_key,
                self._secret_key,
                paper=self.environment.is_paper,
                url_override=self._trading_url(),
            )
            self._trading_client = client
        return await self._run(client.get_clock)

    def _trading_url(self) -> str | None:
        if self.environment.is_real:
            return "https://api.alpaca.markets"
        return "https://paper-api.alpaca.markets"

    async def is_market_open(self) -> bool:
        try:
            clock = await self.get_clock()
        except DataRequestError:
            return False
        return bool(getattr(clock, "is_open", False))

    async def avg_dollar_volume(self, symbol: str, timeframe: str = "1D", *, bars: int = 30) -> float:
        """Average dollar volume over ``bars`` candles, used for liquidity checks."""
        history = await self.get_bars(symbol, timeframe, limit=bars)
        if not history:
            return 0.0
        window = history[-bars:]
        total = sum(b.close * b.volume for b in window)
        return total / len(window)


class DataRequestError(RuntimeError):
    """Raised when Alpaca market data cannot be retrieved after retries."""


__all__.append("DataRequestError")
