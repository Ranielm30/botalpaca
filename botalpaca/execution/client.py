"""Per-environment Alpaca trading client.

One :class:`TradingClient` instance exists per :class:`TradingEnvironment`, and
each is constructed only from that environment's own credentials. There is no
shared client and no shared state, which is what makes the PAPER/REAL
separation structural rather than conventional.

The client is synchronous (``alpaca-py`` REST); every method here is async and
runs the blocking call off the event loop with timeout + retry-with-backoff.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from alpaca.common import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.models import Asset, Order, Position, TradeAccount
from alpaca.trading.requests import GetOrdersRequest

from botalpaca.config import AlpacaEnvironmentConfig, get_logger
from botalpaca.domain import TradingEnvironment

logger = get_logger(__name__)


def order_status_value(order: Any) -> str:
    """The lowercase status name of an Alpaca order, enum or plain string."""
    return str(getattr(order, "status", "")).split(".")[-1].strip().lower()

__all__ = ["AlpacaTradingClient", "BrokerError"]

# Errors worth retrying: transient network/5xx conditions.
_RETRYABLE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "ServerError",
        "ConnectTimeout",
        "ReadTimeout",
    }
)


class BrokerError(RuntimeError):
    """A broker-side failure that is not retryable (rejected order, auth...)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AlpacaTradingClient:
    """Async wrapper around one Alpaca trading account."""

    def __init__(
        self,
        config: AlpacaEnvironmentConfig,
        *,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        config.require_configured()
        self.environment = config.environment
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._api_key = config.api_key.get_secret_value()
        self._secret_key = config.secret_key.get_secret_value()
        self._base_url = config.base_url or (
            "https://api.alpaca.markets"
            if config.environment.is_real
            else "https://paper-api.alpaca.markets"
        )
        self._client: TradingClient | None = None

    @property
    def environment_name(self) -> TradingEnvironment:
        return self.environment

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def raw(self) -> TradingClient:
        """The underlying sync client (used by the market data service)."""
        if self._client is None:
            self._client = TradingClient(
                self._api_key,
                self._secret_key,
                paper=self.environment.is_paper,
                url_override=self._base_url,
            )
        return self._client

    def close(self) -> None:
        """Drop the cached REST client so credentials can be reloaded."""
        self._client = None

    async def _run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Off-loop execution with timeout and exponential backoff."""
        attempt = 0
        delay = self.backoff_seconds
        last_exc: Exception | None = None
        while attempt <= self.max_retries:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs), timeout=self.timeout_seconds
                )
            except APIError as exc:
                # 429 and 5xx are the only API errors worth retrying; anything
                # else (422 rejection, 403 auth) is a decision, not a blip.
                status = getattr(exc, "status_code", None)
                if type(exc).__name__ in _RETRYABLE_NAMES or (status is not None and status >= 500):
                    last_exc = exc
                else:
                    raise BrokerError(
                        f"Alpaca rejected the request: {exc}", status_code=status
                    ) from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
            attempt += 1
            if attempt > self.max_retries:
                break
            logger.warning(
                "alpaca.retry",
                attempt=attempt,
                delay=delay,
                error=str(last_exc),
                environment=self.environment.value,
            )
            await asyncio.sleep(delay)
            delay *= 2
        raise BrokerError(
            f"Alpaca call failed after {attempt} attempts: {last_exc}"
        )

    # -- account -------------------------------------------------------------

    async def get_account(self) -> TradeAccount:
        return await self._run(self.raw.get_account)  # type: ignore[no-any-return]

    async def verify(self) -> TradeAccount:
        """Confirm the credentials authenticate and the account responds.

        Used by the ``/modo`` switch: it must prove the new environment is live
        before the switch is reported as complete.
        """
        account = await self.get_account()
        logger.info(
            "alpaca.verified",
            environment=self.environment.value,
            account_status=getattr(account, "status", None),
        )
        return account

    async def get_account_configurations(self) -> Any:
        return await self._run(self.raw.get_account_configurations)

    # -- positions -----------------------------------------------------------

    async def get_all_positions(self) -> list[Position]:
        return list(await self._run(self.raw.get_all_positions))

    async def get_position(self, symbol: str) -> Position | None:
        positions = await self.get_all_positions()
        target = symbol.upper()
        for pos in positions:
            if str(pos.symbol).upper() == target:
                return pos
        return None

    async def close_position(self, symbol: str, *, qty: str | None = None, percentage: str | None = None) -> Order:
        from alpaca.trading.requests import ClosePositionRequest

        request = ClosePositionRequest(qty=qty, percentage=percentage)
        return await self._run(self.raw.close_position, symbol.upper(), request)  # type: ignore[no-any-return]

    async def close_all_positions(self, *, cancel_orders: bool = True) -> list[Order]:
        return list(await self._run(self.raw.close_all_positions, cancel_orders=cancel_orders))

    # -- orders --------------------------------------------------------------

    async def submit_order(self, request: Any) -> Order:
        return await self._run(self.raw.submit_order, request)  # type: ignore[no-any-return]

    async def get_order_by_id(self, order_id: str) -> Order | None:
        return await self._run(self.raw.get_order_by_id, order_id)  # type: ignore[no-any-return]

    async def get_order_by_client_id(self, client_order_id: str) -> Order | None:
        return await self._run(self.raw.get_order_by_client_id, client_order_id)  # type: ignore[no-any-return]

    async def get_orders(
        self,
        *,
        status: QueryOrderStatus = QueryOrderStatus.OPEN,
        limit: int = 100,
        nested: bool = False,
        side: str | None = None,
        symbols: Sequence[str] | None = None,
    ) -> list[Order]:
        request = GetOrdersRequest(
            status=status,
            limit=limit,
            nested=nested,
            side=side,
            symbols=list(symbols) if symbols else None,
        )
        return list(await self._run(self.raw.get_orders, request))

    async def get_order_for_symbol(self, symbol: str) -> Order | None:
        orders = await self.get_orders(status=QueryOrderStatus.OPEN, symbols=[symbol.upper()])
        return orders[0] if orders else None

    async def cancel_order_by_id(self, order_id: str) -> None:
        await self._run(self.raw.cancel_order_by_id, order_id)

    async def cancel_orders(self, *, symbol: str | None = None) -> None:
        """Cancel open orders, optionally narrowed to one symbol.

        Alpaca's ``cancel_orders`` takes no arguments at all and always cancels
        everything, so a symbol filter has to be resolved here by cancelling each
        open order individually. Passing the symbol through used to raise
        ``TypeError`` and broke every cancel path, ``/cancelar`` included.
        """
        if symbol is None:
            await self._run(self.raw.cancel_orders)
            return

        target = symbol.upper()
        orders = await self.get_orders(status=QueryOrderStatus.OPEN, symbols=[target])
        for order in orders:
            try:
                await self.cancel_order_by_id(order.id)
            except Exception:  # noqa: BLE001 - one bad order must not stop the rest
                logger.warning("cancel_failed", order_id=order.id, symbol=target)

    async def replace_order_by_id(self, order_id: str, request: Any) -> Order:
        return await self._run(self.raw.replace_order_by_id, order_id, request)  # type: ignore[no-any-return]

    # -- assets / clock ------------------------------------------------------

    async def get_asset(self, symbol: str) -> Asset | None:
        try:
            return await self._run(self.raw.get_asset, symbol.upper())  # type: ignore[no-any-return]
        except BrokerError as exc:
            if exc.status_code == 404:
                return None
            raise

    async def get_all_assets(self) -> list[Asset]:
        return list(await self._run(self.raw.get_all_assets))

    async def get_clock(self) -> Any:
        return await self._run(self.raw.get_clock)

    async def get_calendar(self, start: Any = None, end: Any = None) -> list[Any]:
        return list(await self._run(self.raw.get_calendar, start, end))

    async def get_portfolio_history(self, request: Any) -> Any:
        return await self._run(self.raw.get_portfolio_history, request)
