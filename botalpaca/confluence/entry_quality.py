"""Entry quality explanation.

Answers "why should I take this trade?" in a form a human can audit: the
trigger, the confirmation, the invalidation, the numbers, and the historical
precedent. Every field is derived from data already computed; nothing here
invents a justification.
"""

from __future__ import annotations

from botalpaca.domain import Opportunity, RiskAssessment, TechnicalSnapshot

__all__ = ["EntryQualityExplainer", "format_entry_quality"]

# Component display labels, ordered as they appear in the explanation.
COMPONENT_LABELS: dict[str, str] = {
    "trend": "Tendencia",
    "momentum": "Momentum",
    "volume": "Volumen",
    "volatility": "Volatilidad",
    "structure": "Estructura",
    "breakout": "Breakout",
    "multi_timeframe": "Multi-timeframe",
    "market": "Mercado",
    "sector": "Sector",
    "liquidity": "Liquidez",
    "rr": "R:R",
    "risk": "Riesgo",
    "strategy_history": "Hist. estrategia",
    "symbol_history": "Hist. símbolo",
    "setup_history": "Hist. setup",
}


class EntryQualityExplainer:
    """Builds the structured explanation shown before any order."""

    def build(
        self,
        opportunity: Opportunity,
        snapshot: TechnicalSnapshot,
        risk: RiskAssessment | None = None,
    ) -> dict[str, object]:
        """Assemble every field needed for the pre-trade explanation."""
        direction = opportunity.direction.value
        lines: list[str] = []
        blocks: list[str] = []

        # -- Why enter -----------------------------------------------------
        trigger = f"{opportunity.symbol} {direction} en {opportunity.timeframe}"
        if opportunity.reasons:
            trigger += f" - {opportunity.reasons[0]}"
        lines.append(trigger)
        blocks.append(("POR QUE ENTRAR", lines))

        # -- What confirms -------------------------------------------------
        confirmations: list[str] = []
        for c in opportunity.confluences[:6]:
            confirmations.append(f"✔ {c}")
        if not confirmations:
            confirmations.append("⚠ sin confluencias adicionales")
        blocks.append(("QUE CONFIRMA", confirmations))

        # -- Score breakdown ----------------------------------------------
        breakdown_lines = _format_breakdown(opportunity)
        blocks.append(("SCORE " + f"{opportunity.score:.0f}/100", breakdown_lines))

        # -- Levels --------------------------------------------------------
        level_lines = [
            f"Entrada: ${opportunity.entry:,.2f}",
            f"Stop: ${opportunity.stop:,.2f}",
            f"Objetivo: ${opportunity.target:,.2f}",
            f"R:R: {opportunity.rr:.2f}",
        ]
        if opportunity.atr:
            stop_atr = abs(opportunity.entry - opportunity.stop) / opportunity.atr
            level_lines.append(f"Stop a {stop_atr:.1f}x ATR ({opportunity.atr:,.2f})")
        blocks.append(("NIVELES", level_lines))

        # -- Invalidation --------------------------------------------------
        invalidation = opportunity.invalidation or "no definida"
        blocks.append(("INVALIDACION", [f"✖ {invalidation}"]))

        # -- Risk ----------------------------------------------------------
        if risk is not None:
            risk_lines = [
                f"Riesgo: ${risk.max_risk_amount:,.2f} ({risk.risk_per_trade_pct:.2f}% del capital)",
                f"Tamaño: {risk.suggested_qty:g} acciones",
                f"Stop a {risk.stop_distance_pct:.2f}% del precio",
                f"Slippage estimado: {risk.estimated_slippage:.2f}%",
            ]
            if risk.blocks:
                risk_lines.append(f"⛔ BLOQUEADA: {'; '.join(risk.blocks)}")
            blocks.append(("RIESGO", risk_lines))

        # -- Context -------------------------------------------------------
        context_lines = [
            f"Estrategia: {opportunity.strategy.value}",
            f"Régimen: {opportunity.regime.value}",
            f"Sector: {opportunity.sector or 'desconocido'}",
            f"Calidad: {opportunity.quality.value}",
        ]
        ctx_notes = snapshot.context.notes[:3]
        context_lines.extend(f"• {n}" for n in ctx_notes)
        blocks.append(("CONTEXTO", context_lines))

        # -- Historical precedent ------------------------------------------
        hist_lines = _format_history(opportunity)
        if hist_lines:
            blocks.append(("ESTADISTICAS SIMILARES", hist_lines))

        return {
            "symbol": opportunity.symbol,
            "direction": direction,
            "quality": opportunity.quality,
            "score": opportunity.score,
            "blocks": blocks,
            "tradable": opportunity.tradable,
            "blocked_reason": opportunity.non_tradable_reason,
        }

    def render(
        self,
        opportunity: Opportunity,
        snapshot: TechnicalSnapshot,
        risk: RiskAssessment | None = None,
    ) -> str:
        """Plain-text explanation, used by /detalles and the pre-trade card."""
        data = self.build(opportunity, snapshot, risk)
        out: list[str] = [f"📋 CALIDAD DE ENTRADA - {opportunity.symbol}"]
        out.append("")
        for title, items in data["blocks"]:  # type: ignore[misc]
            out.append(f"【{title}】")
            out.extend(f"  {i}" for i in items)
            out.append("")
        if not data["tradable"]:
            out.append(f"⛔ NO OPERABLE: {data['blocked_reason']}")
        else:
            out.append(f"✅ OPERABLE - Calidad {opportunity.quality.value}")
        return "\n".join(out).strip()


def _format_breakdown(opportunity: Opportunity) -> list[str]:
    values = opportunity.breakdown.as_dict()
    parts = [v for k, v in values.items() if v > 0]
    if not parts:
        return ["  (sin componentes)"]
    best = max(parts)
    lines = []
    for key, label in COMPONENT_LABELS.items():
        value = values.get(key, 0.0)
        if value <= 0:
            continue
        bar_len = int(round((value / max(best, 1.0)) * 8))
        bar = "█" * max(1, min(8, bar_len))
        lines.append(f"  {label:<18} {value:5.1f} {bar}")
    return lines


def _format_history(opportunity: Opportunity) -> list[str]:
    hist = opportunity.historical
    if not hist:
        return []
    lines: list[str] = []
    total = hist.get("total_sample")
    if total is not None:
        lines.append(f"  Muestra: {int(total)} operaciones similares")
    for key, label in (
        ("win_rate", "Win rate"),
        ("avg_r", "R medio"),
        ("avg_loss_r", "Pérdida media"),
        ("expectancy_r", "Expectancy"),
    ):
        if hist.get(key) is not None:
            lines.append(f"  {label}: {float(hist[key]):+.2f}" if "r" in key else f"  {label}: {float(hist[key]):.1f}%")
    if hist.get("best_setup"):
        lines.append(f"  Mejor setup: {hist['best_setup']}")
    if hist.get("weak_condition"):
        lines.append(f"  Condición débil: {hist['weak_condition']}")
    if total is not None and int(total) < 20:
        lines.append("  ⚠ muestra pequeña: no concluyente")
    return lines


def format_entry_quality(
    opportunity: Opportunity,
    snapshot: TechnicalSnapshot,
    risk: RiskAssessment | None = None,
) -> str:
    return EntryQualityExplainer().render(opportunity, snapshot, risk)
