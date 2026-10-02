"""Layer 1 — Telegram interface.

This module turns the application services into Telegram messages. It holds no
business rules: every decision (risk, quality, protection) is already made by the
engine it calls. Its jobs are argument parsing, allowlist/rate-limit checks,
confirmation gates and rendering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from botalpaca.config.logging import get_logger
from botalpaca.domain import (
    Opportunity,
    PositionSnapshot,
    TradePlan,
    TradingEnvironment,
)
from botalpaca.domain.errors import (
    AuthorizationError,
    BotalpacaError,
    ConfigurationError,
    DataQualityError,
    DuplicateOrderError,
    EnvironmentMismatchError,
    KillSwitchError,
    RiskRejectedError,
    UnsupportedOrderShapeError,
)
from botalpaca.notifications import (
    env_badge,
    money,
    pct_auto,
    render_account,
    render_opportunity,
    render_orders,
    render_position_alert,
    render_positions,
    render_real_warning,
    render_summary,
    render_trade_plan,
)
from botalpaca.notifications import format as fmt
from botalpaca.security import (
    REAL_CONFIRM_TOKEN,
    CircuitOpenError,
    ConfirmationKind,
    RateLimitError,
)
from botalpaca.telegram import keyboards as kb

log = get_logger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

HELP_TEXT = """<b>botalpaca</b> — asistente de análisis y ejecución Alpaca

<b>Análisis</b>
/analizar [TICKER] — oportunidades de mayor calidad (rating, no lista)
/oportunidades — muestra las mejores del universo
/stats — estadísticas agregadas del entorno activo
/aprender — recomendaciones no vinculantes basadas en el historial

<b>Cuenta y riesgo</b>
/cuenta — equity, cash, buying power del entorno activo
/portfolio — exposición por símbolo y sector
/posiciones — posiciones abiertas con su protección
/ordenes — órdenes abiertas (incluidas las legs de bracket)
/riesgo — límites vigentes y consumo actual

<b>Operaciones</b>
/comprar TICKER [cantidad] — analiza y propone un LONG
/vender TICKER [cantidad] — analiza y propone un SHORT
/cerrar TICKER [cantidad|%] — cierra la posición
/cancelar ORDER_ID — cancela una orden abierta
/modificar ORDER_ID stop=... limit=... — modifica una orden

<b>Control</b>
/modo — PAPER ⇄ REAL (siempre con confirmación)
/monitor — activa o pausa el monitor de mercado
/config — ajustes del bot y del kill switch
/status — salud del sistema
/start — este mensaje
/help — este mensaje

<blockquote>El entorno activo se muestra siempre en cada mensaje. Ninguna señal
se ejecuta sin tu confirmación explícita.</blockquote>"""


class CommandError(BotalpacaError):
    """A user-facing error. The message is safe to send to Telegram verbatim."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.user_message = message


@dataclass(frozen=True)
class CommandResult:
    """What a command produced: text plus an optional inline keyboard."""

    text: str
    keyboard: object | None = None
    alert: bool = False


# --------------------------------------------------------------------- parsing


def parse_symbol(raw: str | None) -> str:
    if not raw:
        raise CommandError("Indica un símbolo. Ejemplo: <code>/analizar AAPL</code>")
    symbol = raw.strip().upper().lstrip("$")
    if not _SYMBOL_RE.match(symbol):
        raise CommandError(f"'{raw}' no parece un símbolo válido.")
    return symbol


def parse_symbol_list(raw: str | None, *, limit: int = 400) -> list[str]:
    if not raw:
        return []
    parts = [p.strip().upper().lstrip("$") for p in re.split(r"[,\s]+", raw) if p.strip()]
    symbols = [p for p in parts if _SYMBOL_RE.match(p)]
    if not symbols:
        raise CommandError("No encontré símbolos válidos en ese argumento.")
    return symbols[:limit]


def parse_float(raw: str | None, label: str) -> float:
    if raw is None:
        raise CommandError(f"Falta {label}.")
    try:
        value = float(str(raw).replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError as exc:
        raise CommandError(f"{label} inválido: '{raw}'.") from exc
    if value <= 0:
        raise CommandError(f"{label} debe ser mayor que cero.")
    return value


def parse_flags(raw: str | None) -> dict[str, str]:
    """Parse ``key=value`` pairs used by /modificar and /config."""
    flags: dict[str, str] = {}
    for token in (raw or "").split():
        if "=" in token:
            key, _, value = token.partition("=")
            flags[key.strip().lower()] = value.strip()
    return flags


def _user_id(update: object) -> int | None:
    user = getattr(getattr(update, "effective_user", None), "id", None)
    return int(user) if user is not None else None


class TelegramFacade:
    """All command logic, independent of the Telegram library.

    Keeping it here means the whole interface is unit-testable without a bot
    token, and the bot module stays a thin adapter.
    """

    def __init__(self, app: object) -> None:
        self.app = app
        # symbol -> latest opportunity seen, so ACEPTAR/DETALLES can act on it.
        self._opportunities: dict[str, Opportunity] = {}
        self._monitoring_enabled = False

    # ------------------------------------------------------------------ guards

    async def authorize(self, update: object) -> int:
        """Allowlist + per-user rate limit. Raises on rejection."""
        user_id = _user_id(update)
        try:
            allowed = self.app.security.allowlist.require(user_id)
        except AuthorizationError as exc:
            raise CommandError(str(exc)) from exc
        try:
            await self.app.security.rate_limiter.check(allowed)
        except RateLimitError as exc:
            raise CommandError(str(exc)) from exc
        return allowed

    async def guard(self, func, *args, **kwargs) -> CommandResult:
        """Run a command, translating every error into a readable message."""
        try:
            return await func(*args, **kwargs)
        except CommandError:
            raise
        except AuthorizationError as exc:
            raise CommandError(str(exc)) from exc
        except RateLimitError as exc:
            raise CommandError(str(exc)) from exc
        except CircuitOpenError as exc:
            raise CommandError(
                f"Circuit breaker abierto ante errores repetidos de Alpaca: {exc}"
            ) from exc
        except KillSwitchError as exc:
            raise CommandError(f"Kill switch activo: {exc}") from exc
        except EnvironmentMismatchError as exc:
            raise CommandError(f"Barrera de entorno: {exc}") from exc
        except DuplicateOrderError as exc:
            raise CommandError(f"Orden duplicada bloqueada: {exc}") from exc
        except RiskRejectedError as exc:
            reasons = "; ".join(getattr(exc, "reasons", []) or [str(exc)])
            raise CommandError(f"El motor de riesgo rechazó la operación: {reasons}") from exc
        except UnsupportedOrderShapeError as exc:
            raise CommandError(
                f"Alpaca no admite esta combinación de órdenes: {exc}"
            ) from exc
        except ConfigurationError as exc:
            raise CommandError(f"Configuración incompleta: {exc}") from exc
        except DataQualityError as exc:
            raise CommandError(f"Datos insuficientes: {exc}") from exc
        except BotalpacaError as exc:
            raise CommandError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - the chat must never see a traceback
            log.exception("telegram.command_failed")
            raise CommandError(f"Error inesperado: {exc}") from exc

    # --------------------------------------------------------------- basics

    @property
    def environment(self) -> TradingEnvironment:
        return self.app.active_environment

    async def start(self, update: object) -> CommandResult:
        await self.authorize(update)
        account = await self.app.portfolio.get_account()
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} <b>botalpaca activo</b>",
                    f"Entorno: <b>{self.environment.value}</b>",
                    f"Cuenta: {account.account_number or account.account_id}",
                    f"Equity: {money(account.equity)}",
                    f"Buying power: {money(account.buying_power)}",
                    "",
                    "Usa /help para ver todos los comandos.",
                ]
            )
        )

    async def help(self, update: object) -> CommandResult:
        await self.authorize(update)
        return CommandResult(
            f"{env_badge(self.environment)}\n\n{HELP_TEXT}"
        )

    async def status(self, update: object) -> CommandResult:
        await self.authorize(update)
        health = await self.app.health()
        kill = "🛑 ACTIVO" if health.kill_switch else "inactivo"
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(health.active_environment)} <b>Estado del sistema</b>",
                    f"Entorno activo: <b>{health.active_environment.value}</b>",
                    f"Base de datos: {'✅' if health.database_ok else '❌'}",
                    f"Alpaca: {'✅' if health.alpaca_ok else '❌'}",
                    f"Monitor de mercado: {'activo' if health.scheduler_ok else 'en espera'}",
                    f"Kill switch: {kill}",
                    f"Uptime: {health.uptime_seconds / 60:.0f} min",
                    "Auto-trading: "
                    + ("⚠️ ACTIVADO" if self.app.settings.monitoring.auto_trading_enabled else "desactivado"),
                ]
            )
        )

    # ----------------------------------------------------------------- /modo

    async def modo(self, update: object, args: str | None = None) -> CommandResult:
        """Show the current mode, or stage an explicit environment switch."""
        await self.authorize(update)

        if args:
            wanted = args.strip().upper()
            if wanted == self.environment.value:
                return await self._mode_card()
            if wanted not in {e.value for e in TradingEnvironment}:
                raise CommandError("Uso: <code>/modo</code> o <code>/modo PAPER|REAL</code>")
            return await self._stage_switch(update, TradingEnvironment(wanted))

        return await self._mode_card()

    async def _mode_card(self) -> CommandResult:
        from botalpaca.telegram.keyboards import mode_keyboard

        account = await self.app.portfolio.get_account()
        if self.environment.is_paper:
            headline = "🟢 Modo actual: ALPACA PAPER"
            note = "Ordenes simuladas. Ningún dinero real en riesgo."
        else:
            headline = "🔴 Modo actual: ALPACA REAL"
            note = "⚠️ Dinero real. Cada operación requiere confirmación adicional."
        text = "\n".join(
            [
                headline,
                f"Cuenta: {account.account_number or account.account_id}",
                f"Equity: {money(account.equity)}",
                f"Buying power: {money(account.buying_power)}",
                "",
                note,
            ]
        )
        return CommandResult(text, mode_keyboard(self.environment))

    async def _stage_switch(self, update: object, target: TradingEnvironment) -> CommandResult:
        from botalpaca.telegram.keyboards import env_confirm_keyboard

        user_id = _user_id(update) or 0
        await self.app.confirmations.request(
            kind=ConfirmationKind.ENVIRONMENT,
            user_id=user_id,
            environment=target,
            summary=f"Cambiar entorno activo a {target.value}",
        )
        if target.is_real:
            body = (
                "🔴 <b>Vas a cambiar a ALPACA REAL</b>\n"
                "A partir de aquí las órdenes usan dinero real y requieren "
                "confirmación explícita en cada operación.\n\n"
                f"Responde <code>{REAL_CONFIRM_TOKEN}</code> o pulsa el botón para confirmar."
            )
        else:
            body = (
                "🟢 <b>Vas a cambiar a ALPACA PAPER</b>\n"
                "Las órdenes siguientes serán simuladas y no afectan a la cuenta real.\n\n"
                "¿Confirmas el cambio?"
            )
        return CommandResult(body, env_confirm_keyboard(target))

    async def confirm_mode_switch(
        self, update: object, target: TradingEnvironment
    ) -> CommandResult:
        """Executed only after an explicit confirmation."""
        user_id = _user_id(update) or 0
        pending = await self.app.confirmations.consume(
            user_id=user_id, kind=ConfirmationKind.ENVIRONMENT
        )
        if pending is None or pending.environment is not target:
            raise CommandError("No hay ninguna confirmación de cambio de entorno pendiente.")
        self.app.security.allowlist.require(user_id)
        account = await self.app.switch_environment(target, confirmed_by=user_id)
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(target)} <b>Entorno activo: {target.value}</b>",
                    f"Cuenta: {account.account_number or account.account_id}",
                    f"Equity: {money(account.equity)}",
                    f"Buying power: {money(account.buying_power)}",
                    "",
                    "Autenticación verificada. No se envió ninguna orden durante el cambio.",
                ]
            )
        )

    # -------------------------------------------------------------- análisis

    async def analizar(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        symbols = parse_symbol_list(args)
        if symbols:
            return await self._analyze_symbols(symbols)
        return await self._scan_universe()

    async def _analyze_symbols(self, symbols: list[str]) -> CommandResult:
        blocks: list[str] = [
            f"{env_badge(self.environment)} <b>Análisis bajo demanda</b>",
            f"Símbolos: {', '.join(symbols)}",
        ]
        for symbol in symbols:
            try:
                outcome = await self.app.analyze_symbol(symbol)
            except BotalpacaError as exc:
                blocks.append(f"\n<b>{symbol}</b>\n⚠️ {exc}")
                continue
            if outcome is None:
                blocks.append(f"\n<b>{symbol}</b>\n⚠️ Sin datos suficientes para analizar.")
                continue
            snapshot, opportunities = outcome
            if not opportunities:
                blocks.append(
                    f"\n<b>{symbol}</b>\nSin señales en el timeframe "
                    f"{snapshot.timeframe}. Regime: {snapshot.volatility.regime.value}."
                )
                continue
            best = opportunities[0]
            self._remember(best)
            blocks.append("")
            blocks.append(render_opportunity(best, environment=self.environment))
            if best.tradable:
                blocks.append(
                    self.app.active.explainer.render(best, snapshot)
                )
        return CommandResult("\n".join(blocks))

    async def _scan_universe(self) -> CommandResult:
        result = await self.app.scan()
        best = result.best()[:5]
        header = [
            f"{env_badge(self.environment)} <b>Mejores oportunidades</b>",
            f"Analizados: {result.scanned} · Con señal: {len(result.opportunities)} · "
            f"Operables: {len(result.tradable)} · {result.duration_seconds:.1f}s",
        ]
        if not best:
            header.append("\nNinguna señal alcanza el umbral de reporte ahora mismo.")
            return CommandResult("\n".join(header))
        for opportunity in best:
            self._remember(opportunity)
            header.append("")
            header.append(render_opportunity(opportunity, environment=self.environment))
        return CommandResult("\n".join(header))

    async def oportunidades(self, update: object, args: str | None = None) -> CommandResult:
        return await self.analizar(update, args)

    async def detalles(self, update: object, symbol: str | None = None) -> CommandResult:
        """Full analysis of one symbol. Typed as `/detalles AAPL`.

        Without a symbol it falls back to the symbols already in the current
        opportunity cache, so the button version and the typed version behave
        identically. Previously this method required `symbol` positionally, so
        typing `/detalles` raised a TypeError.
        """
        await self.authorize(update)
        if not symbol:
            cached = list(self._opportunities)[:5]
            if not cached:
                raise CommandError(
                    "Uso: <code>/detalles SIMBOLO</code>\nEjemplo: <code>/detalles AAPL</code>"
                )
            lines = [fmt.header("Oportunidades recientes", env_badge(self.environment))]
            for symbol_name in cached:
                lines.append(f"  {fmt.BULL} {symbol_name}")
            lines.append("")
            lines.append(f"{fmt.INFO} Pulsa uno o escribe el comando completo.")
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"📊 {name}", callback_data=f"{kb.SIGNAL_DETAILS}:{name}"
                        )
                        for name in cached[:5]
                    ]
                ]
            )
            return CommandResult("\n".join(lines), keyboard)

        target = parse_symbol(symbol)
        outcome = await self.app.analyze_symbol(target)
        if outcome is None:
            raise CommandError(f"No hay datos suficientes para {target}.")
        snapshot, opportunities = outcome
        lines = [
            fmt.header(f"{target} · {snapshot.timeframe}", env_badge(self.environment)),
            fmt.row("Regimen", snapshot.volatility.regime.value),
            fmt.row("Tendencia", (snapshot.trend.direction.value if snapshot.trend.direction else "n/d")),
            fmt.row("Score maximo", f"{max((o.score for o in opportunities), default=0):.1f}"),
            fmt.row("Datos", f"{snapshot.data_quality:.0%}"),
        ]
        if not opportunities:
            lines.append("")
            lines.append(f"{fmt.INFO} Sin senales ahora mismo.")
            return CommandResult(
                "\n".join(lines), kb.signal_keyboard(target)
            )
        for opportunity in opportunities[:3]:
            self._remember(opportunity)
            lines.append("")
            lines.append(self.app.active.explainer.render(opportunity, snapshot))
        return CommandResult("\n".join(lines), kb.signal_keyboard(target))

    async def riesgo_detalle(self, update: object, symbol: str) -> CommandResult:
        await self.authorize(update)
        target = parse_symbol(symbol)
        opportunity = self._opportunities.get(target)
        if opportunity is None:
            outcome = await self.app.analyze_symbol(target)
            if outcome is None or not outcome[1]:
                raise CommandError(f"No hay una señal activa de {target} que evaluar.")
            opportunity = outcome[1][0]
            self._remember(opportunity)
        assessment = await self.app.active.risk_assessment(opportunity)
        verdict = "✅ APROBADA" if assessment.approved else "⛔ RECHAZADA"
        lines = [
            f"{env_badge(self.environment)} <b>Riesgo · {target}</b>",
            f"Veredicto: <b>{verdict}</b>",
            f"Tamaño sugerido: {assessment.suggested_qty:g} acciones",
            f"Riesgo: {money(assessment.max_risk_amount)} "
            f"({assessment.risk_per_trade_pct:.2f}% del equity)",
            f"Distancia al stop: {assessment.stop_distance_pct:.2f}%",
            f"Slippage estimado: {assessment.estimated_slippage:.2f}%",
            f"Exposición actual: {assessment.exposure_pct:.1f}%",
        ]
        if assessment.blocks:
            lines.append("\n⛔ <b>Bloqueos</b>")
            lines.extend(f"• {b}" for b in assessment.blocks)
        elif assessment.reasons:
            lines.append("\n<b>Notas</b>")
            lines.extend(f"• {r}" for r in assessment.reasons)
        return CommandResult("\n".join(lines))

    # ------------------------------------------------------------ cuenta/riesgo

    async def cuenta(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        account = await self.app.portfolio.get_account()
        return CommandResult(render_account(account))

    async def portfolio(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        exposures = await self.app.portfolio.exposures()
        positions = exposures.get("positions") or []
        sectors: dict[str, float] = {}
        from botalpaca.analysis import sector_for_symbol

        for position in positions:
            if not isinstance(position, PositionSnapshot):
                continue
            sector = sector_for_symbol(position.symbol)
            sectors[sector] = sectors.get(sector, 0.0) + abs(position.market_value)
        lines = [
            f"{env_badge(self.environment)} <b>Portfolio</b>",
            f"Equity: {money(float(exposures.get('equity') or 0))}",
            f"Buying power: {money(float(exposures.get('buying_power') or 0))}",
            f"Exposición bruta: {money(float(exposures.get('gross_exposure') or 0))} "
            f"({float(exposures.get('gross_exposure_pct') or 0):.1f}% del equity)",
            f"Exposición neta: {float(exposures.get('net_exposure_pct') or 0):.1f}%",
            f"P&L no realizado: {money(float(exposures.get('unrealized_pl') or 0))}",
            f"Posiciones: {exposures.get('position_count', 0)}",
        ]
        if sectors:
            lines.append("\n<b>Por sector</b>")
            total = sum(sectors.values()) or 1.0
            for sector, value in sorted(sectors.items(), key=lambda kv: -kv[1]):
                lines.append(f"• {sector}: {money(value)} ({value / total * 100:.0f}%)")
        return CommandResult("\n".join(lines))

    async def posiciones(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        wanted = parse_symbol_list(args)
        positions = await self.app.portfolio.get_positions()
        if wanted:
            targets = set(wanted)
            positions = [p for p in positions if p.symbol in targets]
        if not positions:
            return CommandResult(
                render_positions([], self.environment),
                kb.positions_picker_keyboard([], environment=self.environment),
            )

        protection: dict[str, object] = {}
        for position in positions:
            protection[position.symbol] = await self.app.protection.state_for_position(
                self.environment, position
            )

        entries = [(p.symbol, p.unrealized_plpc) for p in positions]
        return CommandResult(
            render_positions(positions, self.environment, protection=protection),
            kb.positions_picker_keyboard(entries, environment=self.environment),
        )

    async def position_menu(self, update: object, symbol: str) -> CommandResult:
        """Show one position in full, with the actions that apply to it."""
        await self.authorize(update)
        target = parse_symbol(symbol)
        position = await self.app.portfolio.get_position(target)
        if position is None or position.qty == 0:
            return CommandResult(
                "\n".join(
                    [
                        fmt.header("Posicion no encontrada", env_badge(self.environment)),
                        f"No hay posicion abierta de <b>{fmt.esc(target)}</b>.",
                    ]
                ),
                kb.positions_picker_keyboard([], environment=self.environment),
            )

        state = await self.app.protection.state_for_position(self.environment, position)
        guard_lines: list[str] = []
        if state.has_stop:
            guard_lines.append(fmt.kv("Stop", money(state.stop_price)))
        elif state.has_trailing:
            guard_lines.append(fmt.kv("Trailing", "activo"))
        else:
            guard_lines.append(f"  {fmt.BULL} {fmt.WARN} <b>Sin stop</b> - posicion expuesta")
        if state.has_take_profit:
            guard_lines.append(fmt.kv("Target", money(state.take_profit_price)))
        if state.break_even_active:
            guard_lines.append(fmt.kv("Break-even", "activo"))
        if state.time_stop_at:
            guard_lines.append(fmt.kv("Time stop", state.time_stop_at.strftime("%Y-%m-%d %H:%M UTC")))
        for note in state.notes[:3]:
            guard_lines.append(f"  {fmt.BULL} {fmt.esc(note)}")

        pl = position.unrealized_pl or 0.0
        lines = [
            fmt.header(target, env_badge(self.environment)),
            fmt.row("Direccion", position.side),
            fmt.row("Cantidad", f"{abs(position.qty):g}"),
            fmt.row("Entrada", money(position.avg_entry_price)),
            fmt.row("Precio actual", money(position.current_price)),
            f"{fmt.trend_glyph(position.unrealized_plpc)} P&amp;L  {fmt.ARROW}  "
            f"<b>{money(pl)}</b> ({pct_auto(position.unrealized_plpc)})",
        ]
        if guard_lines:
            lines.append(fmt.section("Proteccion"))
            lines.extend(guard_lines)
        lines.append("")
        lines.append(fmt.RULE)
        lines.append("Elige una accion:")
        return CommandResult("\n".join(lines), kb.position_actions_keyboard(target))

    async def ordenes(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        status = (args or "open").strip().lower()
        if status not in {"open", "closed", "all"}:
            status = "open"
        orders = await self.app.portfolio.get_orders(status=status, limit=50, nested=True)
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("🔄 Actualizar", callback_data=f"{kb.REFRESH}:orders"),
                    InlineKeyboardButton("📋 Abiertas", callback_data=f"{kb.REFRESH}:orders"),
                    InlineKeyboardButton("✅ Cerradas", callback_data=f"{kb.REFRESH}:closed"),
                ],
                [
                    InlineKeyboardButton("📊 Posiciones", callback_data=f"{kb.REFRESH}:pos"),
                    InlineKeyboardButton("📈 Portfolio", callback_data=f"{kb.REFRESH}:portfolio"),
                ],
            ]
        )
        header = f"Ordenes {status}" if status != "open" else "Ordenes abiertas"
        return CommandResult(render_orders(orders, self.environment, title=header), keyboard)

    async def riesgo(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        settings = self.app.settings.risk
        context = await self.app.active.risk_context()
        equity = context.equity or 0.0
        used_pct = (context.open_risk / equity * 100.0) if equity > 0 else 0.0
        lines = [
            f"{env_badge(self.environment)} <b>Límites de riesgo</b>",
            f"Riesgo por operación: {settings.max_risk_per_trade_pct:.2f}%",
            f"Pérdida diaria máx: {settings.max_daily_loss_pct:.2f}%",
            f"Pérdida semanal máx: {settings.max_weekly_loss_pct:.2f}%",
            f"Pérdida mensual máx: {settings.max_monthly_loss_pct:.2f}%",
            f"Drawdown máx: {settings.max_drawdown_pct:.2f}%",
            f"R:R mínimo: {settings.min_rr:.2f}",
            f"Máx posiciones: {settings.max_open_positions}",
            f"Exposición total máx: {settings.max_total_exposure_pct:.0f}%",
            f"Exposición por sector máx: {settings.max_sector_exposure_pct:.0f}%",
            f"Correlación máx: {settings.max_correlated_exposure_pct:.0f}%",
            f"Concentración máx: {settings.max_concentration_pct:.0f}%",
            f"Liquidez mínima: {money(settings.min_liquidity_avg_dollar_volume)}/día",
            "",
            "<b>Consumo actual</b>",
            f"Riesgo abierto: {money(context.open_risk)} ({used_pct:.2f}% del equity)",
            f"Posiciones abiertas: {len(context.open_positions)}/{settings.max_open_positions}",
            f"Equity: {money(equity)}",
            f"Buying power: {money(context.buying_power)}",
        ]
        return CommandResult("\n".join(lines))

    # ------------------------------------------------------------ operaciones

    async def comprar(self, update: object, args: str | None = None) -> CommandResult:
        return await self._propose(update, args, direction="LONG")

    async def vender(self, update: object, args: str | None = None) -> CommandResult:
        return await self._propose(update, args, direction="SHORT")

    async def _propose(self, update: object, args: str | None, *, direction: str) -> CommandResult:
        """Analyse a symbol and present a risk-approved plan for confirmation.

        Nothing is submitted here: the plan is registered as a pending
        confirmation and only executed by :meth:`confirm_trade`.
        """
        user_id = await self.authorize(update)
        tokens = (args or "").split()
        if not tokens:
            raise CommandError(
                f"Uso: <code>/{'comprar' if direction == 'LONG' else 'vender'} TICKER</code>"
            )
        symbol = parse_symbol(tokens[0])
        qty_override = parse_float(tokens[1], "cantidad") if len(tokens) > 1 else None

        outcome = await self.app.analyze_symbol(symbol)
        if outcome is None:
            raise CommandError(f"Sin datos suficientes para {symbol}.")
        snapshot, opportunities = outcome
        if not opportunities:
            raise CommandError(
                f"{symbol} no tiene ninguna señal operable en {snapshot.timeframe} ahora mismo."
            )

        opportunity = self._pick(opportunities, direction)
        if opportunity is None:
            wanted = "LONG" if direction == "LONG" else "SHORT"
            available = {o.direction.value for o in opportunities}
            raise CommandError(
                f"{symbol} no tiene señal {wanted}. Detectado: {', '.join(sorted(available)) or 'nada'}."
            )
        self._remember(opportunity)

        context = await self.app.active.risk_context()
        assessment = await self.app.active.risk.assess(opportunity, context)
        if qty_override:
            stop_distance = abs(opportunity.entry - opportunity.stop)
            risk_amount = qty_override * stop_distance
            risk_pct = (risk_amount / context.equity * 100.0) if context.equity else 0.0
            from botalpaca.domain import RiskAssessment

            assessment = RiskAssessment(
                approved=assessment.approved and not assessment.blocks,
                reasons=assessment.reasons
                + [f"Cantidad indicada por el usuario: {qty_override:g}"],
                blocks=list(assessment.blocks),
                risk_per_trade_pct=risk_pct,
                max_risk_amount=risk_amount,
                suggested_qty=qty_override,
                suggested_notional=qty_override * opportunity.entry,
                stop_distance_pct=abs(opportunity.entry - opportunity.stop)
                / opportunity.entry
                * 100.0,
                estimated_slippage=assessment.estimated_slippage,
                exposure_pct=assessment.exposure_pct,
                score=assessment.score,
            )

        if not assessment.approved:
            lines = [
                f"{env_badge(self.environment)} ⛔ <b>Operación no permitida</b>",
                f"Símbolo: {opportunity.symbol} · Dirección: {opportunity.direction.value}",
            ]
            lines.extend(f"• {b}" for b in assessment.blocks)
            return CommandResult("\n".join(lines))

        plan = self.app.active.build_plan(opportunity, assessment)
        await self._stage_trade(user_id, plan, assessment, opportunity)

        if self.environment.is_real:
            from botalpaca.telegram.keyboards import real_confirm_keyboard

            body = render_real_warning(plan, assessment)
        else:
            from botalpaca.telegram.keyboards import paper_confirm_keyboard

            body = render_trade_plan(plan, assessment, environment=self.environment)

        keyboard = (
            real_confirm_keyboard(plan.symbol)
            if self.environment.is_real
            else paper_confirm_keyboard(plan.symbol)
        )
        return CommandResult(body, keyboard)

    @staticmethod
    def _pick(opportunities: list[Opportunity], direction: str) -> Opportunity | None:
        for opportunity in opportunities:
            if opportunity.direction.value == direction and opportunity.tradable:
                return opportunity
        return None

    async def _stage_trade(
        self,
        user_id: int,
        plan: TradePlan,
        assessment: object,
        opportunity: Opportunity | None,
    ) -> None:
        await self.app.confirmations.request(
            kind=ConfirmationKind.TRADE,
            user_id=user_id,
            environment=self.environment,
            summary=f"{plan.direction.value} {plan.qty:g} {plan.symbol}",
            details={
                "plan": plan.model_dump(mode="json"),
                "fingerprint": opportunity.fingerprint if opportunity else None,
            },
        )

    async def confirm_trade(self, update: object, token: str) -> CommandResult:
        """Execute a staged plan. This is the only path that submits orders."""
        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        expected = TradingEnvironment(token.upper()) if token else self.environment
        if expected is not self.environment:
            raise CommandError(
                f"La confirmación corresponde a {expected.value} y el entorno activo es "
                f"{self.environment.value}."
            )
        pending = await self.app.confirmations.consume(
            user_id=user_id, kind=ConfirmationKind.TRADE
        )
        if pending is None:
            raise CommandError("No hay ninguna operación pendiente de confirmar.")
        if pending.environment is not self.environment:
            raise CommandError("La confirmación pertenece a otro entorno. Se descarta.")

        plan = TradePlan.model_validate(pending.details.get("plan"))
        from botalpaca.domain import RiskAssessment

        assessment = RiskAssessment.model_validate(
            pending.details.get("assessment", {"approved": True})
        )
        fingerprint = pending.details.get("fingerprint")

        result = await self.app.submit_plan(
            plan, assessment, user_id=user_id, opportunity=self._opportunities.get(plan.symbol)
        )
        if fingerprint:
            await self.app.journal.mark_signal_accepted(self.environment, fingerprint)

        order = result.order
        lines = [
            f"{env_badge(self.environment)} ✅ <b>Orden enviada</b>",
            f"Símbolo: {plan.symbol} · Dirección: {plan.direction.value}",
            f"Cantidad: {plan.qty:g}",
            f"Entrada: {money(plan.limit_price) if plan.limit_price else 'a mercado'}",
            f"Stop: {money(plan.stop_loss)}",
            f"Target: {money(plan.take_profit)}",
            f"Riesgo: {money(assessment.max_risk_amount)}",
            f"ID Alpaca: {order.id if order else 'n/d'}",
            f"Estado: {order.status if order else 'desconocido'}",
        ]
        legs = order.legs if order else []
        if legs:
            lines.append("Protección:")
            lines.extend(
                f"• {leg.order_type.value} {leg.side.value} {leg.qty:g} ({leg.symbol})"
                for leg in legs
            )
        else:
            lines.append("⚠️ Alpaca no devolvió legs de protección en la respuesta.")
        return CommandResult("\n".join(lines))

    async def cancelar_pendiente(self, update: object) -> CommandResult:
        """Discard whatever confirmation was staged, without touching orders."""
        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        await self.app.confirmations.clear(user_id)
        return CommandResult(
            f"{env_badge(self.environment)} Operación cancelada. No se envió ninguna orden."
        )

    async def rechazar(self, update: object, symbol: str) -> CommandResult:
        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        target = parse_symbol(symbol)
        opportunity = self._opportunities.get(target)
        if opportunity is None:
            return CommandResult(f"{target}: no hay señal pendiente que rechazar.")
        await self.app.journal.record_signal(opportunity, accepted=False)
        await self.app.confirmations.clear(user_id)
        self._opportunities.pop(target, None)
        log.info("telegram.signal_rejected", user_id=user_id, symbol=target)
        return CommandResult(
            f"❌ {target} rechazada. Registrada en el journal para análisis posterior."
        )

    async def cerrar(self, update: object, args: str | None = None) -> CommandResult:
        """Stage a position close behind a confirmation."""
        user_id = await self.authorize(update)
        tokens = (args or "").split()
        if not tokens:
            raise CommandError(
                "Uso: <code>/cerrar TICKER</code>, <code>/cerrar TICKER 10</code> o "
                "<code>/cerrar TICKER 50%</code>"
            )
        symbol = parse_symbol(tokens[0])
        position = await self.app.portfolio.get_position(symbol)
        if position is None or position.qty == 0:
            raise CommandError(f"No hay posición abierta de {symbol} en {self.environment.value}.")

        qty: str | None = None
        percentage: str | None = None
        if len(tokens) > 1:
            raw = tokens[1]
            if raw.endswith("%"):
                percentage = str(min(100.0, max(0.01, float(raw[:-1]))))
            else:
                qty = str(int(float(raw)))
        else:
            percentage = "100"

        await self.app.confirmations.request(
            kind=ConfirmationKind.CLOSE,
            user_id=user_id,
            environment=self.environment,
            summary=f"Cerrar {symbol}",
            details={"symbol": symbol, "qty": qty, "percentage": percentage},
        )
        from botalpaca.telegram.keyboards import confirm_keyboard

        scope = f"{percentage}%" if percentage else f"{qty} acciones"
        warning = (
            "\n🔴 <b>ALPACA REAL</b>\n<i>Esta operación utilizará dinero real.</i>"
            if self.environment.is_real
            else ""
        )
        body = "\n".join(
            [
                f"{env_badge(self.environment)} <b>Cerrar posición</b>",
                warning,
                f"Símbolo: <b>{position.symbol}</b>",
                f"Cantidad actual: {position.qty:g} @ {money(position.avg_entry_price)}",
                f"P&L actual: {money(position.unrealized_pl)}",
                f"Cerrar: {scope}",
            ]
        )
        return CommandResult(
            body, confirm_keyboard(action="cerrar", token=f"{symbol}:{qty or percentage}")
        )

    async def confirm_close(self, update: object, symbol: str, scope: str) -> CommandResult:
        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        pending = await self.app.confirmations.consume(
            user_id=user_id, kind=ConfirmationKind.CLOSE
        )
        if pending is None or pending.details.get("symbol") != symbol.upper():
            raise CommandError("No hay ningún cierre pendiente para ese símbolo.")
        await self.app.security.ensure_trading_allowed()
        result = await self.app.active.positions.close_position(
            symbol,
            qty=pending.details.get("qty"),
            percentage=pending.details.get("percentage"),
            confirmed=True,
        )
        await self.app.protection.clear(self.environment, symbol)
        order = result.order
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} ✅ <b>Cierre enviado</b>",
                    f"Símbolo: {symbol}",
                    f"Orden: {order.id if order else 'n/d'}",
                    f"Estado: {order.status if order else 'desconocido'}",
                    "Protección local limpiada.",
                ]
            )
        )

    async def cancelar(self, update: object, args: str | None = None) -> CommandResult:
        """`/cancelar` with no argument drops the pending confirmation.

        With an order id it cancels that order, and with a symbol it cancels every
        open order attached to that position. This command previously only existed
        as a button target and raised a TypeError when typed.
        """
        token = (args or "").strip()
        if not token:
            return await self.cancelar_pendiente(update)
        if token.upper() in {"PENDIENTE", "PENDING"}:
            return await self.cancelar_pendiente(update)
        return await self.cancelar_orden(update, token)

    async def cancelar_orden(self, update: object, order_id: str) -> CommandResult:
        user_id = await self.authorize(update)
        if not order_id:
            raise CommandError("Uso: <code>/cancelar ORDER_ID</code>")
        order = await self.app.portfolio.get_order(order_id)
        if order is None:
            raise CommandError(f"No encontré la orden {order_id} en {self.environment.value}.")
        await self.app.confirmations.request(
            kind=ConfirmationKind.CANCEL,
            user_id=user_id,
            environment=self.environment,
            summary=f"Cancelar {order_id}",
            details={"order_id": order_id},
        )
        from botalpaca.telegram.keyboards import confirm_keyboard

        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} <b>Cancelar orden</b>",
                    f"ID: {order_id}",
                    f"Símbolo: {order.symbol} · {order.side.value} {order.qty or order.notional}",
                    f"Tipo: {order.order_type.value} · Estado: {order.status}",
                    "",
                    "¿Confirmas la cancelación?",
                ]
            ),
            confirm_keyboard(action="cancelar_orden", token=order_id),
        )

    async def confirm_cancel(self, update: object, order_id: str) -> CommandResult:
        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        pending = await self.app.confirmations.consume(
            user_id=user_id, kind=ConfirmationKind.CANCEL
        )
        if pending is None or pending.details.get("order_id") != order_id:
            raise CommandError("No hay ninguna cancelación pendiente para esa orden.")
        await self.app.security.ensure_trading_allowed()
        await self.app.execution.cancel_order(order_id, environment=self.environment)
        return CommandResult(
            f"{env_badge(self.environment)} ✅ Orden {order_id} cancelada."
        )

    async def modificar(self, update: object, args: str | None = None) -> CommandResult:
        user_id = await self.authorize(update)
        tokens = (args or "").split()
        if not tokens:
            raise CommandError(
                "Uso: <code>/modificar ORDER_ID stop=185 limit=190 qty=5</code>"
            )
        order_id = tokens[0]
        flags = parse_flags(" ".join(tokens[1:]))
        if not flags:
            raise CommandError(
                "Indica al menos un campo: <code>stop=</code>, <code>limit=</code> o <code>qty=</code>"
            )
        order = await self.app.portfolio.get_order(order_id)
        if order is None:
            raise CommandError(f"No encontré la orden {order_id} en {self.environment.value}.")

        stop = parse_float(flags.get("stop"), "stop") if flags.get("stop") else None
        limit = parse_float(flags.get("limit"), "limit") if flags.get("limit") else None
        qty = int(parse_float(flags.get("qty"), "qty")) if flags.get("qty") else None

        await self.app.confirmations.request(
            kind=ConfirmationKind.MODIFY,
            user_id=user_id,
            environment=self.environment,
            summary=f"Modificar {order_id}",
            details={
                "order_id": order_id,
                "stop": stop,
                "limit": limit,
                "qty": qty,
            },
        )
        from botalpaca.telegram.keyboards import confirm_keyboard

        changes = []
        if stop:
            changes.append(f"stop → {stop}")
        if limit:
            changes.append(f"limit → {limit}")
        if qty:
            changes.append(f"qty → {qty}")
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} <b>Modificar orden</b>",
                    f"ID: {order_id} · {order.symbol} {order.order_type.value}",
                    f"Cambios: {', '.join(changes)}",
                    "",
                    "⚠️ Alpaca no combina stop y limit en la misma orden: se "
                    "modifica el campo indicado y el resto queda intacto.",
                    "¿Confirmas?",
                ]
            ),
            confirm_keyboard(action="modificar", token=order_id),
        )

    async def confirm_modify(self, update: object, order_id: str) -> CommandResult:
        from botalpaca.execution import OrderBuilder

        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        pending = await self.app.confirmations.consume(
            user_id=user_id, kind=ConfirmationKind.MODIFY
        )
        if pending is None or pending.details.get("order_id") != order_id:
            raise CommandError("No hay ninguna modificación pendiente para esa orden.")
        await self.app.security.ensure_trading_allowed()
        details = pending.details
        request = OrderBuilder.build_replace(
            qty=details.get("qty"),
            stop_price=details.get("stop"),
            limit_price=details.get("limit"),
        )
        order = await self.app.execution.replace_order(
            order_id, request, environment=self.environment
        )
        await self.app.protection.reconcile(self.environment, [])
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} ✅ <b>Orden modificada</b>",
                    f"ID: {order.id}",
                    f"Nuevo estado: {order.status}",
                    f"Stop: {money(order.stop_price)}",
                    f"Limit: {money(order.limit_price)}",
                ]
            )
        )

    # -------------------------------------------------------------- protección

    async def break_even(self, update: object, symbol: str) -> CommandResult:
        user_id = await self.authorize(update)
        target = parse_symbol(symbol)
        position = await self.app.portfolio.get_position(target)
        if position is None or position.qty == 0:
            raise CommandError(f"No hay posición abierta de {target}.")
        await self.app.confirmations.request(
            kind=ConfirmationKind.BREAK_EVEN,
            user_id=user_id,
            environment=self.environment,
            summary=f"Break-even {target}",
            details={"symbol": target},
        )
        from botalpaca.telegram.keyboards import confirm_keyboard

        entry = position.avg_entry_price
        buffer = self.app.settings.protection.break_even_buffer_pct
        stop = round(entry * (1 + buffer / 100.0), 2) if position.qty > 0 else round(
            entry * (1 - buffer / 100.0), 2
        )
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} <b>Mover stop a break-even</b>",
                    f"Símbolo: {target}",
                    f"Entrada: {money(entry)}",
                    f"Nuevo stop: {money(stop)} (+{buffer:.2f}% de buffer)",
                    "",
                    "¿Confirmas?",
                ]
            ),
            confirm_keyboard(action="break_even", token=target),
        )

    async def confirm_break_even(self, update: object, symbol: str) -> CommandResult:
        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        pending = await self.app.confirmations.consume(
            user_id=user_id, kind=ConfirmationKind.BREAK_EVEN
        )
        if pending is None or pending.details.get("symbol") != symbol:
            raise CommandError("No hay ningún break-even pendiente.")
        await self.app.security.ensure_trading_allowed()
        position = await self.app.portfolio.get_position(symbol)
        if position is None:
            raise CommandError(f"La posición de {symbol} ya no existe.")
        state = await self.app.protection.move_to_break_even(
            environment=self.environment, position=position
        )
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} ✅ <b>Break-even aplicado</b>",
                    f"Símbolo: {symbol}",
                    f"Stop: {money(state.stop_price)}",
                    f"Orden de stop: {state.stop_order_id or 'n/d'}",
                ]
                + [f"• {note}" for note in state.notes[:3]]
            )
        )

    async def trailing(self, update: object, symbol: str) -> CommandResult:
        user_id = await self.authorize(update)
        target = parse_symbol(symbol)
        position = await self.app.portfolio.get_position(target)
        if position is None or position.qty == 0:
            raise CommandError(f"No hay posición abierta de {target}.")
        state = await self.app.protection.state_for_position(self.environment, position)
        if state.has_trailing:
            await self.app.confirmations.request(
                kind=ConfirmationKind.TRAILING,
                user_id=user_id,
                environment=self.environment,
                summary=f"Quitar trailing {target}",
                details={"symbol": target, "enable": False},
            )
            from botalpaca.telegram.keyboards import confirm_keyboard

            return CommandResult(
                f"{env_badge(self.environment)} El trailing de {target} ya está activo "
                f"({state.trail_percent:.2f}%). ¿Quieres desactivarlo?",
                confirm_keyboard(action="trailing", token=target),
            )

        percent = self.app.settings.protection.default_trailing_percent
        await self.app.confirmations.request(
            kind=ConfirmationKind.TRAILING,
            user_id=user_id,
            environment=self.environment,
            summary=f"Trailing {target} {percent}%",
            details={"symbol": target, "enable": True},
        )
        from botalpaca.telegram.keyboards import confirm_keyboard

        body = [
            f"{env_badge(self.environment)} <b>Activar trailing stop</b>",
            f"Símbolo: {target}",
            f"Trail: {percent:.2f}%",
        ]
        if state.has_stop or state.has_take_profit:
            body.append("")
            body.append(
                "⚠️ Alpaca no permite un trailing stop como leg de un bracket. "
                "Para activar el trailing hay que cancelar primero las legs "
                "existentes y colocar el trailing. Si la colocación falla, se "
                "restaurará la protección anterior."
            )
        return CommandResult(
            "\n".join(body), confirm_keyboard(action="trailing", token=target)
        )

    async def confirm_trailing(self, update: object, symbol: str) -> CommandResult:
        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        pending = await self.app.confirmations.consume(
            user_id=user_id, kind=ConfirmationKind.TRAILING
        )
        if pending is None or pending.details.get("symbol") != symbol:
            raise CommandError("No hay ninguna acción de trailing pendiente.")
        await self.app.security.ensure_trading_allowed()
        position = await self.app.portfolio.get_position(symbol)
        if position is None:
            raise CommandError(f"La posición de {symbol} ya no existe.")
        if pending.details.get("enable"):
            state = await self.app.protection.enable_trailing_stop(
                environment=self.environment, position=position
            )
            action = "activado"
        else:
            state = await self.app.protection.disable_trailing_stop(
                environment=self.environment, position=position
            )
            action = "desactivado"
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} ✅ Trailing {action}",
                    f"Símbolo: {symbol}",
                    f"Stop actual: {money(state.stop_price)}",
                    f"Take profit: {money(state.take_profit_price)}",
                ]
                + [f"• {note}" for note in state.notes[:4]]
            )
        )

    # -------------------------------------------------------------- stats

    async def stats(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        environment = self.environment
        overall = await self.app.learning.overall(environment)
        by_strategy = await self.app.learning.by_strategy(environment)
        by_regime = await self.app.learning.by_regime(environment)
        lines = [
            f"{env_badge(environment)} <b>Estadísticas · {environment.value}</b>",
            "",
            render_summary(overall),
        ]
        if by_strategy:
            lines += ["", "<b>Por estrategia</b>"]
            ranked = sorted(
                by_strategy.values(),
                key=lambda s: (s.expectancy_r if s.expectancy_r is not None else -99.0),
                reverse=True,
            )
            for summary in ranked[:6]:
                lines.append("• " + _one_line(summary))
        if by_regime:
            lines += ["", "<b>Por régimen de mercado</b>"]
            for summary in list(by_regime.values())[:5]:
                lines.append("• " + _one_line(summary))
        lines += ["", "<i>Las muestras pequeñas no son estadísticamente significativas.</i>"]
        return CommandResult("\n".join(lines))

    async def historial(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        environment = self.environment
        symbols = parse_symbol_list(args)
        lines = [
            f"{env_badge(environment)} <b>Historial · {environment.value}</b>",
        ]
        from botalpaca.db import TradeRepository

        async with self.app.database.session() as session:
            repo = TradeRepository(session)
            for symbol in symbols:
                lines.append("")
                lines.append(f"<b>{symbol}</b>")
                trades = await repo.get_all(environment, limit=10, symbol=symbol)
                if not trades:
                    lines.append("Sin operaciones registradas.")
                for trade in trades:
                    lines.append(
                        f"• {trade.opened_at:%Y-%m-%d %H:%M} {trade.symbol} "
                        f"{trade.direction.value} {trade.status.value} · "
                        f"score {trade.score:.0f} · P&L {money(trade.pnl)} "
                        f"({trade.pnl_pct:+.2f}%) · {trade.r_multiple:+.2f}R · "
                        f"{trade.strategy}"
                    )
            if not symbols:
                trades = await repo.get_closed(environment, 15)
                if not trades:
                    lines.append("\nSin operaciones cerradas todavía.")
                for trade in trades:
                    lines.append(
                        f"• {trade.closed_at:%Y-%m-%d %H:%M} {trade.symbol} "
                        f"{trade.direction.value} · {money(trade.pnl)} "
                        f"({trade.pnl_pct:+.2f}%) · {trade.r_multiple:+.2f}R · "
                        f"salida: {trade.exit_reason}"
                    )
        return CommandResult("\n".join(lines))

    async def explicar(self, update: object, args: str | None = None) -> CommandResult:
        """Answer 'why did you recommend X?' from the persisted journal."""
        await self.authorize(update)
        symbol = parse_symbol(args)
        entries = await self.app.journal.explain(self.environment, symbol)
        if not entries:
            return CommandResult(
                f"{env_badge(self.environment)} No hay registro de {symbol} en "
                f"{self.environment.value}."
            )
        return CommandResult(_render_explanation(entries))

    async def aprender(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        recommendations = await self.app.learning.recommendations(self.environment)
        lines = [
            f"{env_badge(self.environment)} <b>Aprendizaje estadístico</b>",
            "",
        ]
        if not recommendations:
            lines.append(
                "Todavía no hay suficientes operaciones cerradas para extraer "
                "conclusiones. El sistema no inventa estadísticas: necesita datos."
            )
        else:
            lines.extend(f"• {r}" for r in recommendations)
        lines += [
            "",
            "<i>Estas recomendaciones son informative. El sistema no modifica "
            "parámetros críticos automáticamente.</i>",
        ]
        return CommandResult("\n".join(lines))

    # -------------------------------------------------------- monitor / config

    async def monitor(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        action = (args or "").strip().lower()
        if action in {"on", "activar", "start"}:
            self._monitoring_enabled = True
        elif action in {"off", "pausar", "stop"}:
            self._monitoring_enabled = False
        running = self.app.active.market_monitor.running
        state = "activo" if running else ("habilitado, esperando ciclo" if self._monitoring_enabled else "en espera")
        last = self.app.active.market_monitor.last_result
        lines = [
            f"{env_badge(self.environment)} <b>Monitor de mercado</b>",
            f"Estado: {state}",
            f"Auto-trading: {'⚠️ ACTIVADO' if self.app.settings.monitoring.auto_trading_enabled else 'desactivado (por diseño)'}",
            f"Alerta mínima: score {self.app.settings.monitoring.opportunity_alert_min_score:.0f}",
            f"Cooldown de alertas: {self.app.settings.monitoring.signal_alert_cooldown_seconds}s",
        ]
        if last is not None:
            lines.append(
                f"Último barrido: {len(last.opportunities)} señales · "
                f"{len(last.tradable)} operables · {last.duration_seconds:.1f}s"
            )
        lines += [
            "",
            "Uso: <code>/monitor on</code> · <code>/monitor off</code>",
            "<i>Las señales detectadas nunca se ejecutan sin tu confirmación.</i>",
        ]
        return CommandResult("\n".join(lines))

    async def config(self, update: object, args: str | None = None) -> CommandResult:
        await self.authorize(update)
        environment = self.environment
        tokens = (args or "").split()
        if tokens and tokens[0].lower() == "kill":
            return await self._kill_switch(update, tokens[1:])

        settings = self.app.settings
        kill, reason = await self.app.security.kill_switch_state()
        lines = [
            f"{env_badge(environment)} <b>Configuración</b>",
            f"Entorno activo: {environment.value}",
            f"Base de datos: {settings.database_url}",
            f"Zona horaria: {settings.timezone}",
            f"Timeframe por defecto: {settings.strategy.default_timeframe} "
            f"(alto {settings.strategy.higher_timeframe}, bajo {settings.strategy.lower_timeframe})",
            f"Universo: {len(settings.scanner.universe)} símbolos · benchmark {settings.scanner.benchmark}",
            f"Intervalos: scan {settings.scan_interval_seconds}s · monitor {settings.monitor_interval_seconds}s "
            f"· reconcile {settings.reconcile_interval_seconds}s",
            f"Kill switch: {'🛑 ACTIVO' if kill else 'inactivo'}"
            + (f" ({reason})" if reason else ""),
            f"Auto-trading: {'⚠️ ACTIVADO' if settings.monitoring.auto_trading_enabled else 'desactivado'}",
            f"Usuarios autorizados: {len(self.app.security.allowlist.allowed)}",
            "",
            "<b>Entornos configurados</b>",
        ]
        for env in TradingEnvironment:
            cfg = settings.alpaca(env)
            status = "✅ configurado" if cfg.is_configured else "❌ sin credenciales"
            lines.append(f"• {env.value}: {status} ({cfg.base_url or 'base url por defecto'})")
        lines += [
            "",
            "Kill switch: <code>/config kill on|off</code>",
            "No se muestran claves ni secretos en ningún mensaje.",
        ]
        return CommandResult("\n".join(lines))

    async def _kill_switch(self, update: object, args: list[str]) -> CommandResult:
        action = (args[0].lower() if args else "")
        user_id = _user_id(update) or 0
        if action in {"on", "activar"}:
            await self.app.confirmations.request(
                kind=ConfirmationKind.CLOSE,
                user_id=user_id,
                environment=self.environment,
                summary="Activar kill switch",
                details={"kill_switch": True},
            )
            from botalpaca.telegram.keyboards import confirm_keyboard

            return CommandResult(
                "🛑 <b>Activar kill switch</b>\n"
                "Bloquea el envío de cualquier orden nueva. Las posiciones y "
                "órdenes existentes siguen gestionándose.\n\n¿Confirmas?",
                confirm_keyboard(action="kill_switch", token="on"),
            )
        if action in {"off", "desactivar", "release"}:
            await self.app.security.release_kill_switch()
            return CommandResult("🟢 Kill switch desactivado. Las órdenes están habilitadas.")
        if action == "":
            active, reason = await self.app.security.kill_switch_state()
            return CommandResult(
                f"Kill switch: {'🛑 ACTIVO' if active else 'inactivo'}"
                + (f"\nMotivo: {reason}" if reason else "")
            )
        raise CommandError("Uso: <code>/config kill on</code> o <code>/config kill off</code>")

    async def confirm_kill_switch(self, update: object, token: str) -> CommandResult:
        user_id = _user_id(update) or 0
        self.app.security.allowlist.require(user_id)
        pending = await self.app.confirmations.consume(user_id=user_id)
        if pending is None or not pending.details.get("kill_switch"):
            raise CommandError("No hay ninguna activación de kill switch pendiente.")
        await self.app.security.engage_kill_switch("activado manualmente desde Telegram")
        return CommandResult(
            "🛑 <b>Kill switch ACTIVADO</b>\nNo se enviarán órdenes nuevas hasta desactivarlo."
        )

    # ------------------------------------------------------------- reconciliar

    async def reconciliar(self, update: object, args: str | None = None) -> CommandResult:
        """Force a SQLite ↔ Alpaca reconciliation on demand."""
        await self.authorize(update)
        notes = await self.app.reconcile()
        if not notes:
            return CommandResult(
                f"{env_badge(self.environment)} ✅ Reconciliación completa. "
                "La protección en SQLite coincide con Alpaca."
            )
        return CommandResult(
            "\n".join(
                [
                    f"{env_badge(self.environment)} <b>Reconciliación</b>",
                    *[f"• {n}" for n in notes],
                ]
            )
        )

    # ------------------------------------------------------------------ alerts

    async def handle_alert(self, payload: object) -> None:
        """Render a monitor alert into the allowed chat(s)."""
        opportunities = getattr(payload, "opportunities", ()) or ()
        alerts = getattr(payload, "alerts", ()) or ()
        environment = getattr(payload, "environment", self.environment)

        from botalpaca.telegram.keyboards import position_keyboard, signal_keyboard

        for opportunity in opportunities:
            self._remember(opportunity)
            text = render_opportunity(opportunity, environment=environment)
            if alerts or opportunity.tradable:
                text = (
                    "🔔 <b>Nueva oportunidad</b>\n"
                    + text
                    + "\n\n<i>No se ejecutará nada sin tu confirmación.</i>"
                )
            await self.app.notifications.send(text, signal_keyboard(opportunity.symbol))

        for alert in alerts:
            position = alert.position
            text = render_position_alert(
                position,
                previous_score=alert.previous_score,
                current_score=alert.current_score,
                changes=alert.changes,
                environment=environment,
            )
            if alert.exit_rule_triggered:
                text += f"\n\nRegla de protección activada: {alert.exit_rule_triggered}"
            await self.app.notifications.send(text, position_keyboard(position.symbol))

    def _remember(self, opportunity: Opportunity) -> None:
        self._opportunities[opportunity.symbol.upper()] = opportunity


def _one_line(summary: object) -> str:
    parts = [f"{summary.label}: n={summary.sample_size}"]
    parts.append(f"WR {summary.win_rate:.0%}")
    if summary.expectancy_r is not None:
        parts.append(f"E {summary.expectancy_r:+.2f}R")
    if summary.profit_factor is not None:
        parts.append(f"PF {summary.profit_factor:.2f}")
    if not summary.is_significant:
        parts.append("(muestra pequeña)")
    return " · ".join(parts)


def _render_explanation(entries: dict[str, Any]) -> str:
    """Turn a ``TradeJournal.explain`` payload into the /explicar answer.

    The journal returns ORM rows, so every field is read defensively: the card
    must still render when only the signal half exists.
    """
    symbol = str(entries.get("symbol", "?"))
    environment = str(entries.get("environment", "?"))
    badge = f"{TradingEnvironment(environment).badge} {TradingEnvironment(environment).label}" if environment in {"PAPER", "REAL"} else environment
    lines = [f"{badge} <b>Por qué {symbol}</b>", ""]

    signal = entries.get("signal")
    if signal is not None:
        lines.append("<b>Señal registrada</b>")
        lines.append(
            f"• {getattr(signal, 'created_at', None)} · {getattr(signal, 'strategy', '?')} "
            f"· {getattr(signal, 'direction', '?')} · score {getattr(signal, 'score', 0):.0f}"
        )
        confluences = list(getattr(signal, "confluences", []) or [])
        if confluences:
            lines.append(f"• Confluencias: {', '.join(str(c) for c in confluences)}")
        reasons = list(getattr(signal, "reasons", []) or [])
        if reasons:
            lines.append(f"• Motivo: {'; '.join(str(r) for r in reasons)}")
        lines.append("")

    trade = entries.get("trade")
    if trade is not None:
        lines.append("<b>Resultado posterior</b>")
        status = getattr(trade, "status", "?")
        if status == "OPEN":
            lines.append(
                f"• Abierta: entrada {money(getattr(trade, 'entry_price', 0))} · "
                f"stop {money(getattr(trade, 'stop_price', None) or 0)} · "
                f"target {money(getattr(trade, 'target_price', None) or 0)}"
            )
        else:
            lines.append(
                f"• Cerrada: {money(getattr(trade, 'exit_price', 0))} · "
                f"P&L {money(getattr(trade, 'pnl', 0))} · "
                f"{getattr(trade, 'r_multiple', 0):+.2f}R · salida "
                f"{getattr(trade, 'exit_reason', 'n/d')}"
            )
        lines.append("")

    count = int(entries.get("history_count", 0) or 0)
    lines.append(
        f"Operaciones históricas en {environment}: {count}"
        + (" (muestra pequeña: no estadísticamente significativo)" if count < 20 else "")
    )
    return "\n".join(lines)


__all__ = [
    "HELP_TEXT",
    "CommandError",
    "CommandResult",
    "TelegramFacade",
    "parse_flags",
    "parse_float",
    "parse_symbol",
    "parse_symbol_list",
]
