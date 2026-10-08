"""The bar window has to end at the present, not ten months in the past.

Alpaca answers a request that carries both ``start`` and ``limit`` with the
OLDEST ``limit`` bars of that window. The market service asked for 300 daily
bars inside a 750-day window, so the 300 bars it received started in December
2024 and ended on 2025-11-28. Every scanner run since then analysed a snapshot
more than ten months stale: the entry price it proposed was that frozen close,
and the same "opportunity" came back on every run.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from botalpaca.config import AlpacaEnvironmentConfig
from botalpaca.domain import TradingEnvironment
from botalpaca.market.service import MarketDataService

UTC = dt.UTC


def _raw(day: dt.datetime, close: float) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=day, open=close, high=close, low=close, close=close, volume=1e6
    )


class _AlpacaLikeClient:
    """Mimics the one behaviour that caused the outage.

    ``start`` plus ``limit`` keeps the oldest bars of the window. Without
    ``limit`` it returns the whole window. That is the real API contract, and
    reproducing it here is what makes these tests worth having.
    """

    def __init__(self, bars: list[SimpleNamespace]) -> None:
        self.bars = bars
        self.requests: list[SimpleNamespace] = []

    def get_stock_bars(self, request: SimpleNamespace) -> dict[str, list]:
        """Synchronous on purpose: the real client is, and ``_run`` wraps it."""
        self.requests.append(request)
        window = [b for b in self.bars if request.start.timestamp() <= b.timestamp.timestamp()]
        window.sort(key=lambda b: b.timestamp)
        if request.limit is not None:
            return {"AAPL": window[: request.limit]}
        return {"AAPL": window}


def _service(bars: list[SimpleNamespace]) -> tuple[MarketDataService, _AlpacaLikeClient]:
    config = AlpacaEnvironmentConfig(api_key="paper-key", secret_key="paper-secret")
    service = MarketDataService(config, environment=TradingEnvironment.PAPER)
    client = _AlpacaLikeClient(bars)
    # The real client is built lazily so importing never opens a socket; seeding
    # the attribute directly keeps the socket closed in tests.
    service._data_client = client  # noqa: SLF001
    return service, client


def _trading_days(count: int, *, last: dt.datetime) -> list[SimpleNamespace]:
    """``count`` weekday closes ending on the last one."""
    days: list[dt.datetime] = []
    day = last
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= dt.timedelta(days=1)
    days.reverse()
    return [_raw(d, 100.0 + i) for i, d in enumerate(days)]


@pytest.mark.asyncio
async def test_the_window_ends_at_the_last_bar_that_exists():
    """The newest bar returned has to be the newest bar available."""
    last = dt.datetime.now(UTC) - dt.timedelta(hours=20)
    bars = _trading_days(300, last=last)
    service, _ = _service(bars)

    result = await service.get_bars("AAPL", timeframe="1D", limit=300)

    assert len(result) == 300
    assert result[-1].timestamp.date() == last.date(), (
        "el escaner se esta analizando un mercado que ya no existe: "
        f"la ultima vela que recibio es del {result[-1].timestamp.date()}"
    )


@pytest.mark.asyncio
async def test_a_short_window_still_honours_the_limit():
    """Trimming must not pad the result with bars older than the window."""
    last = dt.datetime.now(UTC) - dt.timedelta(hours=20)
    bars = _trading_days(300, last=last)
    service, _ = _service(bars)

    result = await service.get_bars("AAPL", timeframe="1D", limit=50)

    assert len(result) == 50
    assert result[-1].timestamp.date() == last.date()
    assert result[0].timestamp > bars[-51].timestamp


@pytest.mark.asyncio
async def test_the_multi_request_behaves_the_same_way():
    """The batched path had the identical defect and must not drift back."""
    last = dt.datetime.now(UTC) - dt.timedelta(hours=20)
    bars = _trading_days(300, last=last)
    service, _ = _service(bars)

    result = await service.get_bars_multi(["aapl"], timeframe="1D", limit=300)

    assert result["AAPL"][-1].timestamp.date() == last.date()


@pytest.mark.asyncio
async def test_the_request_does_not_ask_alpaca_to_pick_the_old_end():
    """Regression on the cause itself, not just on its symptom."""
    last = dt.datetime.now(UTC) - dt.timedelta(hours=20)
    service, client = _service(_trading_days(300, last=last))

    await service.get_bars("AAPL", timeframe="1D", limit=300)

    assert client.requests[0].limit is None, (
        "volver a mandar limit junto con start hace que Alpaca devuelva "
        "las barras mas antiguas de la ventana"
    )
