"""Layer 10 — Portfolio / account manager.

All reads go through a single :class:`AlpacaTradingClient` that is bound to ONE
environment. The manager never accepts an ``environment`` override for reads:
it can only ever see the account it was constructed for. That makes cross
environment leakage structurally impossible.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from botalpaca.config.logging import get_logger
from botalpaca.domain import (
    AccountSnapshot,
    OrderState,
    PositionSnapshot,
    TradingEnvironment,
)
from botalpaca.execution import AlpacaTradingClient, BrokerError
from botalpaca.execution.mapping import to_account, to_order_state, to_position

log = get_logger(__name__)


class PortfolioService:
    """Read-only view over one Alpaca account."""

    def __init__(self, client: AlpacaTradingClient) -> None:
        self._client = client

    @property
    def environment(self) -> TradingEnvironment:
        return self._client.environment

    async def get_account(self) -> AccountSnapshot:
        raw = await self._client.get_account()
        return to_account(raw, self.environment)

    async def verify_connection(self) -> AccountSnapshot:
        """Auth check used by ``/modo`` and ``/status``: one real account call."""
        raw = await self._client.verify()
        snapshot = to_account(raw, self.environment)
        log.info(
            "portfolio.verified",
            environment=self.environment.value,
            account_id=snapshot.account_id,
            status=snapshot.status,
        )
        return snapshot

    async def get_positions(self) -> list[PositionSnapshot]:
        raws = await self._client.get_all_positions()
        return [to_position(raw, self.environment) for raw in raws]

    async def get_position(self, symbol: str) -> PositionSnapshot | None:
        raw = await self._client.get_position(symbol)
        return to_position(raw, self.environment) if raw is not None else None

    async def get_orders(
        self,
        *,
        status: str = "open",
        limit: int = 100,
        nested: bool = True,
        side: str | None = None,
        symbols: Sequence[str] | None = None,
    ) -> list[OrderState]:
        raws = await self._client.get_orders(
            status=status, limit=limit, nested=nested, side=side, symbols=symbols
        )
        return [to_order_state(raw, self.environment) for raw in raws]

    async def get_open_orders(self, *, nested: bool = True) -> list[OrderState]:
        return await self.get_orders(status="open", limit=200, nested=nested)

    async def get_order(self, order_id: str) -> OrderState | None:
        raw = await self._client.get_order_by_id(order_id)
        return to_order_state(raw, self.environment) if raw is not None else None

    async def get_asset_info(self, symbol: str) -> dict[str, object]:
        asset = await self._client.get_asset(symbol)
        if asset is None:
            return {}
        return {
            "symbol": asset.symbol,
            "name": asset.name,
            "status": str(getattr(asset.status, "value", asset.status)),
            "tradable": asset.tradable,
            "marginable": asset.marginable,
            "shortable": asset.shortable,
            "easy_to_borrow": asset.easy_to_borrow,
            "fractionable": asset.fractionable,
            "exchange": asset.exchange,
            "min_order_size": asset.min_order_size,
            "min_trade_increment": asset.min_trade_increment,
            "price_increment": asset.price_increment,
            "maintenance_margin_requirement": asset.maintenance_margin_requirement,
        }

    async def get_clock(self) -> dict[str, object]:
        clock = await self._client.get_clock()
        return {
            "is_open": bool(clock.is_open),
            "next_open": clock.next_open,
            "next_close": clock.next_close,
            "timestamp": getattr(clock, "timestamp", None),
        }

    async def portfolio_history(self, request: object) -> object:
        return await self._client.get_portfolio_history(request)

    async def exposures(self) -> dict[str, object]:
        """Aggregate exposure view used by the risk engine and ``/portfolio``."""
        account = await self.get_account()
        positions = await self.get_positions()
        equity = account.equity or account.portfolio_value or 0.0
        long_value = sum(p.market_value for p in positions if p.qty > 0)
        short_value = sum(abs(p.market_value) for p in positions if p.qty < 0)
        unrealized = sum(p.unrealized_pl for p in positions)
        return {
            "environment": self.environment.value,
            "as_of": dt.datetime.now(dt.UTC),
            "equity": equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "long_market_value": long_value,
            "short_market_value": short_value,
            "gross_exposure": long_value + short_value,
            "gross_exposure_pct": ((long_value + short_value) / equity * 100.0) if equity > 0 else 0.0,
            "net_exposure_pct": ((long_value - short_value) / equity * 100.0) if equity > 0 else 0.0,
            "unrealized_pl": unrealized,
            "position_count": len(positions),
            "positions": positions,
        }


__all__ = ["PortfolioService", "BrokerError"]
