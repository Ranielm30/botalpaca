"""Layer 7 — Risk management engine.

Every order must pass through :meth:`RiskEngine.assess`. Anything that returns
``approved=False`` MUST NOT be executed. There is no override path except an
explicit, logged ``force`` which the Telegram layer never uses.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from botalpaca.analysis.market_context import sector_for_symbol
from botalpaca.config import RiskSettings, get_settings
from botalpaca.config.logging import get_logger
from botalpaca.db import DailyPnlRepository, Database, TradeRepository
from botalpaca.domain import (
    AccountSnapshot,
    Opportunity,
    RiskAssessment,
    SignalDirection,
    TradingEnvironment,
)
from botalpaca.portfolio import PortfolioService

log = get_logger(__name__)

MIN_EQUITY_USD = 100.0


@dataclass(slots=True)
class OpenExposure:
    """Portfolio state the risk engine reasons about."""

    symbol: str
    sector: str
    notional: float


@dataclass(slots=True)
class PortfolioRiskContext:
    environment: TradingEnvironment
    equity: float
    buying_power: float
    cash: float
    open_positions: list[OpenExposure] = field(default_factory=list)
    correlations: dict[str, float] = field(default_factory=dict)
    account: AccountSnapshot | None = None
    #: Sum of ``risk_amount`` across open trades (the drawdown currently at risk).
    open_risk: float = 0.0

    @property
    def total_notional(self) -> float:
        return sum(p.notional for p in self.open_positions)

    def sector_notional(self, sector: str) -> float:
        return sum(p.notional for p in self.open_positions if p.sector == sector)

    def symbol_notional(self, symbol: str) -> float:
        return sum(p.notional for p in self.open_positions if p.symbol == symbol)


def _status_is_active(status: object) -> bool:
    """Whether an Alpaca account status means "you may trade".

    ``str(AccountStatus.ACTIVE)`` is ``"AccountStatus.ACTIVE"`` on Python 3.11+,
    not ``"ACTIVE"``. Comparing the repr against "ACTIVE" therefore fails for
    every account, including healthy ones, and silently blocks 100% of entries
    while printing the very status it rejected.
    """
    if status is None:
        return True
    value = getattr(status, "value", status)
    text = str(value).strip().upper()
    # Tolerate a prefixed repr such as "AccountStatus.ACTIVE".
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text in {"ACTIVE", ""}


class RiskEngine:
    """Pre-trade risk validation and position sizing."""

    def __init__(
        self,
        database: Database,
        portfolio: PortfolioService,
        *,
        settings: RiskSettings | None = None,
    ) -> None:
        self._db = database
        self._portfolio = portfolio
        self._settings = settings or get_settings().risk

    @property
    def settings(self) -> RiskSettings:
        return self._settings

    # ------------------------------------------------------------------ sizing

    def calculate_position_size(
        self,
        *,
        equity: float,
        entry: float,
        stop: float,
        risk_pct: float | None = None,
        max_qty_cap: float | None = None,
    ) -> tuple[float, float, float]:
        """Return ``(qty, risk_amount, stop_distance_pct)``.

        ``qty`` is 0.0 when the geometry makes sizing impossible.
        """
        if entry <= 0 or stop <= 0:
            return 0.0, 0.0, 0.0
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0, 0.0, 0.0
        pct = self._settings.max_risk_per_trade_pct if risk_pct is None else risk_pct
        risk_amount = equity * pct / 100.0
        qty = risk_amount / stop_distance
        if max_qty_cap is not None:
            qty = min(qty, max_qty_cap)
        qty = math.floor(qty)
        stop_distance_pct = stop_distance / entry * 100.0
        return float(qty), risk_amount, stop_distance_pct

    # ---------------------------------------------------------------- context

    async def build_context(
        self,
        environment: TradingEnvironment,
        *,
        correlations: dict[str, float] | None = None,
    ) -> PortfolioRiskContext:
        account = await self._portfolio.get_account()
        positions = await self._portfolio.get_positions()
        exposure = [
            OpenExposure(
                symbol=p.symbol,
                sector=sector_for_symbol(p.symbol),
                notional=abs(p.market_value),
            )
            for p in positions
            if p.qty != 0
        ]
        equity = account.equity or account.portfolio_value or 0.0
        async with self._db.session() as session:
            trades = TradeRepository(session)
            open_trades = await trades.get_open_positions(environment)
        open_risk = sum(float(t.risk_amount or 0.0) for t in open_trades)
        return PortfolioRiskContext(
            environment=environment,
            equity=equity,
            buying_power=account.buying_power,
            cash=account.cash,
            open_positions=exposure,
            correlations=dict(correlations or {}),
            account=account,
            open_risk=open_risk,
        )

    # ------------------------------------------------------------- assessment

    async def assess(
        self,
        opportunity: Opportunity,
        context: PortfolioRiskContext,
        *,
        average_dollar_volume: float | None = None,
        spread_pct: float | None = None,
    ) -> RiskAssessment:
        s = self._settings
        equity = context.equity
        blocks: list[str] = []
        reasons: list[str] = []

        if opportunity.environment != context.environment:
            blocks.append(
                f"Entorno de la oportunidad ({opportunity.environment.value}) "
                f"no coincide con el activo ({context.environment.value})"
            )

        if equity < MIN_EQUITY_USD:
            blocks.append(f"Equity insuficiente (${equity:,.2f})")
            return RiskAssessment(approved=False, blocks=blocks, reasons=reasons)

        if not opportunity.tradable:
            blocks.append(f"Oportunidad no operable: {opportunity.non_tradable_reason or 'sin motivo'}")

        # --- kill-switch-style account level blocks -------------------------
        if context.account is not None:
            acct = context.account
            if acct.trading_blocked:
                blocks.append("Cuenta bloqueada para trading (trading_blocked)")
            if acct.account_blocked:
                blocks.append("Cuenta bloqueada (account_blocked)")
            if acct.trade_suspended:
                blocks.append("Trading suspendido por el usuario en Alpaca")
            if not _status_is_active(acct.status):
                value = getattr(acct.status, "value", acct.status)
                blocks.append(f"Estado de cuenta no activo: {value}")

        # --- loss limits ----------------------------------------------------
        await self._check_loss_limits(
            context.environment, equity, context.open_risk, blocks, reasons
        )

        # --- position count -------------------------------------------------
        max_positions = s.max_open_positions
        same_symbol = any(p.symbol == opportunity.symbol for p in context.open_positions)
        if len(context.open_positions) >= max_positions and not same_symbol:
            blocks.append(f"Máximo de posiciones abiertas alcanzado ({len(context.open_positions)}/{max_positions})")

        # --- geometry -------------------------------------------------------
        entry = opportunity.entry
        stop = opportunity.stop
        target = opportunity.target
        stop_distance_pct = abs(entry - stop) / entry * 100.0 if entry > 0 else 0.0
        if entry <= 0 or stop <= 0 or target <= 0:
            blocks.append("Niveles de entrada/stop/target inválidos")
        elif stop == entry:
            blocks.append("Stop igual a la entrada: riesgo cero indefinido")
        else:
            correct_side = (
                (stop < entry < target)
                if opportunity.direction is SignalDirection.LONG
                else (stop > entry > target)
            )
            if not correct_side:
                blocks.append("Stop/target no coherentes con la dirección de la señal")

        if stop_distance_pct < s.min_stop_distance_pct:
            blocks.append(
                f"Stop demasiado cercano ({stop_distance_pct:.2f}% < {s.min_stop_distance_pct}%)"
            )
        if stop_distance_pct > s.max_stop_distance_pct:
            blocks.append(
                f"Stop demasiado lejano ({stop_distance_pct:.2f}% > {s.max_stop_distance_pct}%)"
            )

        # --- R:R ------------------------------------------------------------
        if opportunity.rr < s.min_rr:
            blocks.append(f"R:R insuficiente ({opportunity.rr:.2f} < {s.min_rr})")

        # --- sizing ---------------------------------------------------------
        risk_pct = min(
            s.max_risk_per_trade_pct,
            self._max_risk_pct_remaining(context, blocks),
        )
        max_risk_amount = equity * risk_pct / 100.0
        qty, risk_amount, _ = self.calculate_position_size(
            equity=equity, entry=entry, stop=stop, risk_pct=risk_pct
        )
        if qty <= 0:
            blocks.append("El tamaño calculado es 0 acciones (riesgo por acción demasiado alto)")
            qty = 0.0
        notional = qty * entry
        if qty > 0 and notional < s.min_order_notional:
            blocks.append(
                f"Notional ${notional:,.2f} por debajo del mínimo (${s.min_order_notional:,.2f})"
            )
        if qty > 0 and notional > s.max_order_notional:
            allowed = math.floor(s.max_order_notional / entry)
            if allowed >= 1:
                reasons.append(
                    f"Tamaño recortado de {qty:.0f} a {allowed:.0f} acciones por notional máximo"
                )
                qty = float(allowed)
                notional = qty * entry
                risk_amount = abs(entry - stop) * qty
            else:
                blocks.append("Notional máximo por operación insuficiente para 1 acción")

        # --- exposure caps: trim, do not reject ---------------------------
        # A tight stop implies a large notional for a small monetary risk
        # (1% of equity with a 2% stop is 50% of equity). Rather than
        # refusing an otherwise sound trade, the size is reduced to respect
        # the concentration and sector caps. Risk per trade drops with it,
        # which is the intended behaviour of a fixed-fractional sizer.
        sector = opportunity.sector or sector_for_symbol(opportunity.symbol)
        if qty > 0 and equity > 0:
            caps = {
                "concentración por operación": equity * s.max_concentration_pct / 100.0,
                f"exposición del sector {sector}": max(
                    0.0, equity * s.max_sector_exposure_pct / 100.0 - context.sector_notional(sector)
                ),
                "exposición total": max(
                    0.0, equity * s.max_total_exposure_pct / 100.0 - context.total_notional
                ),
            }
            for label, cap in caps.items():
                if notional > cap:
                    allowed = math.floor(cap / entry)
                    if allowed >= 1:
                        reasons.append(
                            f"Tamaño recortado de {qty:.0f} a {allowed:.0f} acciones "
                            f"por el límite de {label} (${cap:,.2f})"
                        )
                        qty = float(allowed)
                        notional = qty * entry
                        risk_amount = abs(entry - stop) * qty
                    else:
                        blocks.append(
                            f"El límite de {label} no permite ni 1 acción (${cap:,.2f})"
                        )

        if qty <= 0:
            max_risk_amount = 0.0

        # --- buying power ---------------------------------------------------
        if qty > 0 and notional > context.buying_power:
            blocks.append(
                f"Buying power insuficiente (${notional:,.2f} > ${context.buying_power:,.2f})"
            )

        # --- exposure -------------------------------------------------------
        exposure_pct = 0.0
        if qty > 0 and equity > 0:
            total = context.total_notional + notional
            exposure_pct = total / equity * 100.0
            if total > context.equity * s.max_total_exposure_pct / 100.0:
                blocks.append(
                    f"Exposición total excedida ({exposure_pct:.1f}% > {s.max_total_exposure_pct}%)"
                )
            # Concentration and sector caps were already applied as trims
            # above; re-check after trimming so a cap of zero (already fully
            # allocated) still surfaces as a block rather than a silent pass.
            if notional > equity * s.max_concentration_pct / 100.0:
                blocks.append(
                    f"Concentración por operación excedida "
                    f"(${notional:,.2f} > {s.max_concentration_pct}% de equity)"
                )

            sector_total = context.sector_notional(sector) + notional
            if sector_total > equity * s.max_sector_exposure_pct / 100.0:
                blocks.append(
                    f"Exposición por sector {sector} excedida "
                    f"({sector_total / equity * 100:.1f}% > {s.max_sector_exposure_pct}%)"
                )

            correlated = self._correlated_exposure(context, opportunity.symbol) + notional
            if correlated > equity * s.max_correlated_exposure_pct / 100.0:
                blocks.append(
                    f"Exposición correlacionada excedida "
                    f"({correlated / equity * 100:.1f}% > {s.max_correlated_exposure_pct}%)"
                )

        # --- liquidity / spread --------------------------------------------
        if average_dollar_volume is not None and average_dollar_volume < s.min_liquidity_avg_dollar_volume:
            blocks.append(
                f"Liquidez insuficiente (volumen medio ${average_dollar_volume:,.0f} < "
                f"${s.min_liquidity_avg_dollar_volume:,.0f})"
            )
        if spread_pct is not None and spread_pct > s.max_spread_pct:
            blocks.append(f"Spread demasiado amplio ({spread_pct:.3f}% > {s.max_spread_pct}%)")

        slippage_bps = self._estimate_slippage(spread_pct, opportunity.atr, entry)
        if slippage_bps > s.max_slippage_estimate_bps:
            blocks.append(f"Slippage estimado alto ({slippage_bps:.0f} bps)")
        if opportunity.atr is not None and entry > 0:
            atr_pct = opportunity.atr / entry * 100.0
            if atr_pct > 4.0:
                reasons.append(f"Volatilidad alta (ATR {atr_pct:.1f}% del precio): reduce el tamaño")
            elif atr_pct < 0.3:
                reasons.append("Volatilidad muy baja: cuidado con stops demasiado ajustados")

        # --- volatility / gap risk -----------------------------------------
        if opportunity.atr is not None and entry > 0 and stop_distance_pct > 0:
            if opportunity.atr / entry * 100.0 * 2.0 < stop_distance_pct:
                reasons.append("El stop está a más de 2 ATR: riesgo de ruido elevado")

        score = self._score(opportunity, qty, risk_amount, equity, exposure_pct)

        approved = not blocks
        assessment = RiskAssessment(
            approved=approved,
            reasons=reasons,
            blocks=blocks,
            risk_per_trade_pct=risk_pct,
            max_risk_amount=max_risk_amount,
            suggested_qty=qty,
            suggested_notional=qty * entry if qty > 0 else None,
            stop_distance_pct=stop_distance_pct,
            estimated_slippage=slippage_bps,
            exposure_pct=exposure_pct,
            score=score,
        )
        log.info(
            "risk.assessed",
            environment=context.environment.value,
            symbol=opportunity.symbol,
            approved=approved,
            qty=assessment.suggested_qty,
            score=assessment.score,
            blocks=blocks,
        )
        return assessment

    # ------------------------------------------------------------------ limits

    async def _check_loss_limits(
        self,
        environment: TradingEnvironment,
        equity: float,
        open_risk: float,
        blocks: list[str],
        reasons: list[str],
    ) -> None:
        s = self._settings
        if equity <= 0:
            return
        today = dt.date.today()
        async with self._db.session() as session:
            daily = DailyPnlRepository(session)
            day_pnl = await daily.realized_sum(environment, today, today)
            week_pnl = await daily.realized_sum(environment, today - dt.timedelta(days=6))
            month_pnl = await daily.realized_sum(environment, today - dt.timedelta(days=29))

        for label, pnl, limit in (
            ("diario", day_pnl, s.max_daily_loss_pct),
            ("semanal", week_pnl, s.max_weekly_loss_pct),
            ("mensual", month_pnl, s.max_monthly_loss_pct),
        ):
            if pnl >= 0:
                continue
            loss_pct = abs(pnl) / equity * 100.0
            if loss_pct >= limit:
                blocks.append(f"Límite de pérdida {label} alcanzado ({loss_pct:.2f}% >= {limit}%)")
            elif loss_pct >= limit * 0.7:
                reasons.append(f"Acercándose al límite de pérdida {label} ({loss_pct:.2f}% de {limit}%)")

        drawdown_budget = equity * s.max_drawdown_pct / 100.0
        if open_risk > drawdown_budget:
            blocks.append(
                f"Riesgo abierto ${open_risk:,.2f} "
                f"({open_risk / equity * 100:.2f}%) supera el drawdown máximo "
                f"({s.max_drawdown_pct}%)"
            )
        elif open_risk > drawdown_budget * 0.7:
            reasons.append(
                f"Riesgo abierto en {open_risk / equity * 100:.1f}% del equity "
                f"(máximo {s.max_drawdown_pct}%)"
            )

    # ----------------------------------------------------------------- helpers

    def _max_risk_pct_remaining(
        self, context: PortfolioRiskContext, blocks: list[str]
    ) -> float:
        """Cap per-trade risk so total open risk stays inside the drawdown budget."""
        s = self._settings
        equity = context.equity
        if equity <= 0:
            return 0.0
        budget = equity * s.max_drawdown_pct / 100.0
        open_risk = context.open_risk
        remaining = budget - open_risk
        if remaining <= 0:
            blocks.append("Presupuesto de riesgo abierto agotado")
            return 0.0
        pct = remaining / equity * 100.0
        return min(s.max_risk_per_trade_pct, pct)

    def _correlated_exposure(self, context: PortfolioRiskContext, symbol: str) -> float:
        target_corr = context.correlations.get(symbol, 0.0)
        total = 0.0
        for p in context.open_positions:
            if p.symbol == symbol:
                total += p.notional
                continue
            corr = context.correlations.get(p.symbol, 0.0)
            if target_corr and corr and abs(target_corr * corr) >= self._settings.correlation_threshold:
                total += p.notional
            elif abs(corr) >= self._settings.correlation_threshold:
                total += p.notional
        return total

    def _estimate_slippage(
        self, spread_pct: float | None, atr: float | None, entry: float
    ) -> float:
        """Half-spread + one ATR-in-pct penalty, expressed in basis points."""
        half_spread_bps = (spread_pct or 0.0) / 2.0 * 100.0
        atr_bps = 0.0
        if atr is not None and entry > 0:
            atr_bps = atr / entry * 100.0 * 100.0 * 0.05
        return round(half_spread_bps + atr_bps, 1)

    @staticmethod
    def _score(
        opportunity: Opportunity,
        qty: float,
        risk_amount: float,
        equity: float,
        exposure_pct: float,
    ) -> float:
        """0..100 health score for the trade as sized (not the setup score)."""
        if qty <= 0 or equity <= 0:
            return 0.0
        score = 60.0
        score += min(20.0, opportunity.score / 5.0)
        risk_pct = risk_amount / equity * 100.0
        score += max(0.0, 10.0 - abs(risk_pct - 1.0) * 10.0)
        score -= max(0.0, exposure_pct - 50.0) * 0.3
        return round(max(0.0, min(100.0, score)), 1)


__all__ = ["OpenExposure", "PortfolioRiskContext", "RiskEngine"]
