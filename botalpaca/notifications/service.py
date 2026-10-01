"""Layer 15 — Notification engine.

All Telegram output funnels through here so that formatting, rate limiting and
delivery errors are handled in exactly one place. Renderers build the text; the
service decides whether it is allowed to send it and records the outcome.
"""

from __future__ import annotations

import datetime as dt
import html
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from botalpaca.config.logging import get_logger
from botalpaca.domain import (
    AccountSnapshot,
    Opportunity,
    OrderState,
    PositionSnapshot,
    RiskAssessment,
    StatisticalSummary,
    TradePlan,
    TradingEnvironment,
)

log = get_logger(__name__)

Sender = Callable[[str, object], Awaitable[None]]


def env_badge(environment: TradingEnvironment) -> str:
    """``🟢 ALPACA PAPER`` / ``🔴 ALPACA REAL``.

    The emoji alone is not enough: the environment name must be readable text in
    every message so a PAPER screenshot can never be mistaken for REAL.
    """
    return f"{environment.badge} {environment.label}"


def pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/d"
    return f"{value:+.{digits}f}%"


def money(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/d"
    return f"${value:,.{digits}f}"


def pct_auto(value: float | None, digits: int = 2) -> str:
    """Format a value Alpaca may report as a ratio (0.021) or a percent (2.1)."""
    if value is None:
        return "n/d"
    if abs(value) < 1.0:
        value = value * 100.0
    return f"{value:+.{digits}f}%"


def _esc(text: str) -> str:
    return html.escape(str(text))


def bar(value: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(100.0, value)) / 100.0 * width))
    return "█" * filled + "░" * (width - filled)


@dataclass
class NotificationBudget:
    """Simple per-hour ceiling so a bad scan cannot flood the chat."""

    max_per_hour: int = 20
    sent: list[dt.datetime] = field(default_factory=list)

    def allow(self, now: dt.datetime | None = None) -> bool:
        now = now or dt.datetime.now(dt.UTC)
        cutoff = now - dt.timedelta(hours=1)
        self.sent = [t for t in self.sent if t > cutoff]
        if len(self.sent) >= self.max_per_hour:
            return False
        self.sent.append(now)
        return True


# --------------------------------------------------------------------- render


def render_opportunity(opportunity: Opportunity, *, environment: TradingEnvironment) -> str:
    """Compact card used by /analizar and the background market monitor."""
    lines = [
        f"{env_badge(environment)} <b>{opportunity.symbol}</b> · {opportunity.strategy.value}",
        f"Dirección: {opportunity.direction.name}",
        f"TF: {opportunity.timeframe} · Régimen: {opportunity.regime.value}",
        f"Precio: {money(opportunity.entry)}",
        f"Stop: {money(opportunity.stop)}  |  Target: {money(opportunity.target)}",
        f"R:R: {opportunity.rr:.2f}",
        f"Score: <b>{opportunity.score:.0f}</b>/100 · Calidad: {opportunity.quality.value}",
    ]
    if opportunity.confluences:
        lines.append("Confluencias: " + ", ".join(opportunity.confluences[:6]))
    if opportunity.reasons:
        lines.append("Razones: " + "; ".join(opportunity.reasons[:3]))
    hist = opportunity.historical or {}
    sample = int(hist.get("sample_size") or 0)
    if sample:
        lines.append(
            f"Histórico similar: n={sample}, win rate {float(hist.get('win_rate') or 0):.0%}, "
            f"R medio {float(hist.get('average_r') or 0):+.2f}"
        )
    if not opportunity.tradable:
        lines.append(f"⛔ NO OPERABLE: {opportunity.non_tradable_reason or 'reglas de riesgo'}")
    return "\n".join(lines)


def render_trade_plan(
    plan: TradePlan,
    assessment: RiskAssessment,
    *,
    environment: TradingEnvironment,
) -> str:
    entry_ref = money(plan.limit_price) if plan.limit_price is not None else "a mercado"
    lines = [
        f"{env_badge(environment)} <b>Plan de operación</b>",
        f"Símbolo: {plan.symbol} · Dirección: {plan.direction.name} · Tipo: {plan.order_type.value}",
        f"Cantidad: {plan.qty:g}",
        f"Entrada: {entry_ref}",
        f"Stop: {money(plan.stop_loss)}",
        f"Target: {money(plan.take_profit)}",
        f"R:R: {plan.rr:.2f}",
        f"Riesgo: {money(assessment.max_risk_amount)} ({assessment.risk_per_trade_pct:.2f}% del equity)",
    ]
    if assessment.blocks:
        lines.append("⛔ Bloqueos de riesgo: " + "; ".join(assessment.blocks))
    return "\n".join(lines)


def render_real_warning(plan: TradePlan, assessment: RiskAssessment) -> str:
    """The mandatory pre-REAL card. Text is fixed by the operating rules."""
    direction = "LONG" if plan.direction.name == "LONG" else "SHORT"
    return "\n".join(
        [
            "🔴 <b>ALPACA REAL</b>",
            "<i>Esta operación utilizará dinero real.</i>",
            "",
            f"Símbolo: <b>{plan.symbol}</b>",
            f"Dirección: <b>{direction}</b>",
            f"Cantidad: <b>{plan.qty:g}</b>",
            f"Entrada: {money(plan.limit_price) if plan.limit_price else 'a mercado'}",
            f"Stop: {money(plan.stop_loss)}",
            f"Target: {money(plan.take_profit)}",
            f"Riesgo: {money(assessment.max_risk_amount)} "
            f"({assessment.risk_per_trade_pct:.2f}% del equity)",
            f"R:R: {plan.rr:.2f}",
        ]
    )


def render_account(account: AccountSnapshot) -> str:
    return "\n".join(
        [
            f"{env_badge(account.environment)} <b>Cuenta Alpaca</b>",
            f"Cuenta: {getattr(account, 'account_number', None) or account.account_id}",
            f"Estado: {account.status}",
            f"Equity: {money(account.equity)}",
            f"Cash: {money(account.cash)}",
            f"Buying power: {money(account.buying_power)}",
            f"Valor de posiciones long: {money(account.long_market_value)}",
            f"Portfolio value: {money(account.portfolio_value)}",
            f"Day trades hoy: {account.daytrade_count}"
            + (" · PDT" if account.pattern_day_trader else ""),
        ]
    )


def render_positions(positions: Sequence[PositionSnapshot], environment: TradingEnvironment) -> str:
    if not positions:
        return f"{env_badge(environment)} No hay posiciones abiertas."
    lines = [f"{env_badge(environment)} <b>Posiciones ({len(positions)})</b>"]
    for p in positions:
        lines.append(
            f"\n• <b>{p.symbol}</b> {p.qty:g} @ {money(p.avg_entry_price)} → {money(p.current_price)}"
        )
        lines.append(f"  P&L {money(p.unrealized_pl)} ({pct_auto(p.unrealized_plpc)})")
    total = sum(p.unrealized_pl for p in positions)
    lines.append(f"\nP&L no realizado total: <b>{money(total)}</b>")
    return "\n".join(lines)


def render_orders(orders: Sequence[OrderState], environment: TradingEnvironment) -> str:
    if not orders:
        return f"{env_badge(environment)} No hay órdenes."
    lines = [f"{env_badge(environment)} <b>Órdenes ({len(orders)})</b>"]
    for o in orders[:25]:
        size = o.qty if o.qty is not None else o.notional
        parts = [
            f"• {o.symbol} {o.side.value} {size if size is not None else 0:g} "
            f"{o.order_type.value} · {o.status}"
        ]
        if o.limit_price:
            parts.append(f"@ {money(o.limit_price)}")
        if o.stop_price:
            parts.append(f"stop {money(o.stop_price)}")
        if o.trail_percent:
            parts.append(f"trail {o.trail_percent:.2f}%")
        lines.append("".join(parts))
    return "\n".join(lines)


def render_summary(summary: StatisticalSummary, *, environment: TradingEnvironment | None = None) -> str:
    title = f"<b>{_esc(summary.label)}</b> · n={summary.sample_size}"
    lines = [
        title if environment is None else f"{env_badge(environment)} {title}",
        f"Win rate: {summary.win_rate:.0%}"
        + ("" if summary.is_significant else "  (muestra pequeña)"),
    ]
    if summary.profit_factor is not None:
        lines.append(f"Profit factor: {summary.profit_factor:.2f}")
    if summary.expectancy_r is not None:
        lines.append(f"Expectancy: {summary.expectancy_r:+.2f}R")
    if summary.average_r is not None:
        lines.append(f"R medio: {summary.average_r:+.2f}R")
    if summary.max_drawdown_r is not None:
        lines.append(f"Máx drawdown: {summary.max_drawdown_r:.2f}R")
    if summary.sharpe is not None:
        lines.append(f"Sharpe (aprox): {summary.sharpe:.2f}")
    if summary.avg_hold_minutes is not None:
        lines.append(f"Duración media: {summary.avg_hold_minutes:.0f} min")
    if summary.caveat:
        lines.append(f"<i>{_esc(summary.caveat)}</i>")
    return "\n".join(lines)


def render_position_alert(
    position: PositionSnapshot,
    *,
    previous_score: float,
    current_score: float,
    changes: Sequence[str],
    environment: TradingEnvironment,
) -> str:
    """Example required by the spec, e.g. score 82 → 61 with the reasons."""
    lines = [
        f"⚠️ <b>{position.symbol}</b> perdió momentum.",
        f"Score: {previous_score:.0f} → {current_score:.0f}",
    ]
    lines.extend(f"• {_esc(c)}" for c in changes)
    lines.append(f"Posición {pct_auto(position.unrealized_plpc)}")
    lines.append(f"{env_badge(environment)} No se cierra automáticamente sin regla configurada.")
    return "\n".join(lines)


# -------------------------------------------------------------------- service


class NotificationService:
    """Owns the Telegram sender and the hourly alert budget."""

    def __init__(self, sender: Sender, *, max_per_hour: int = 20) -> None:
        self._sender = sender
        self.budget = NotificationBudget(max_per_hour=max_per_hour)

    async def send(
        self, text: str, keyboard: object = None, *, force: bool = False
    ) -> bool:
        if not force and not self.budget.allow():
            log.info("notifications.throttled")
            return False
        try:
            await self._sender(text, keyboard)
        except Exception:  # noqa: BLE001 - never let a notification kill a worker
            log.exception("notifications.send_failed")
            return False
        return True

    async def send_alert(self, text: str, keyboard: object = None) -> bool:
        return await self.send(text, keyboard, force=False)


__all__ = [
    "NotificationBudget",
    "NotificationService",
    "Sender",
    "bar",
    "env_badge",
    "money",
    "pct",
    "pct_auto",
    "render_account",
    "render_orders",
    "render_position_alert",
    "render_positions",
    "render_opportunity",
    "render_real_warning",
    "render_summary",
    "render_trade_plan",
]
