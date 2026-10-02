"""python-telegram-bot adapter.

This module is intentionally thin: it parses the transport (which command, which
button) and delegates every decision to :class:`~botalpaca.telegram.service.TelegramFacade`.
"""

from __future__ import annotations

import inspect

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application as TelegramApplication,
)
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    Defaults,
)

from botalpaca.config.logging import get_logger
from botalpaca.domain import TradingEnvironment
from botalpaca.domain.errors import AuthorizationError
from botalpaca.telegram import keyboards as kb
from botalpaca.telegram.service import CommandError, CommandResult, TelegramFacade

log = get_logger(__name__)


def _args(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    args = getattr(context, "args", None)
    return " ".join(str(a) for a in args) if args else None


def _chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int | None:
    """Resolve the chat to reply to.

    ``CallbackContext`` exposes no ``effective_chat``; the chat always comes from
    the update itself. Callback queries carry it inside ``callback_query.message``
    (or ``.inline_message_id``, which cannot be replied to as text).
    """
    chat = getattr(update, "effective_chat", None)
    if chat is not None:
        return chat.id
    callback = getattr(update, "callback_query", None)
    message = getattr(callback, "message", None) if callback is not None else None
    return getattr(message, "chat", None) and message.chat.id


async def _answer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    keyboard: object | None = None,
) -> None:
    """Send a reply, splitting on the 4096-char Telegram limit."""
    chat_id = _chat_id(update, context)
    if chat_id is None:
        log.warning("telegram.reply_skipped_no_chat")
        return
    try:
        if len(text) <= 4096:
            await context.bot.send_message(
                chat_id, text, parse_mode=ParseMode.HTML, reply_markup=keyboard
            )
            return
        # Long analyses: send the first chunk then the rest sequentially.
        head, remainder = text[:4000], text[4000:]
        await context.bot.send_message(
            chat_id, head, parse_mode=ParseMode.HTML, reply_markup=keyboard
        )
        for start in range(0, len(remainder), 4000):
            await context.bot.send_message(
                chat_id, remainder[start : start + 4000], parse_mode=ParseMode.HTML
            )
    except TelegramError:
        log.exception("telegram.send_failed")


async def _dispatch(
    update: Update, context: ContextTypes.DEFAULT_TYPE, coro_factory, **kwargs
) -> None:
    facade: TelegramFacade = context.application.bot_data["facade"]
    try:
        result: CommandResult = await facade.guard(coro_factory, update, **kwargs)
    except CommandError as exc:
        await _answer(update, context, f"⚠️ {exc.user_message}")
        return
    except AuthorizationError:
        # Never reveal whether the bot exists to unauthorized users.
        log.warning("telegram.unauthorized_attempt", user=update.effective_user)
        return
    await _answer(update, context, result.text, result.keyboard)


def _make_command(name: str, method_name: str):
    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        facade: TelegramFacade = context.application.bot_data["facade"]
        method = getattr(facade, method_name)
        # Only forward `args` when the command actually accepts it. `/start`,
        # `/help` and `/status` take no arguments, and passing them anyway made
        # every one of those commands fail with a TypeError.
        extra: dict[str, object] = {}
        if "args" in inspect.signature(method).parameters:
            extra["args"] = _args(context)
        await _dispatch(update, context, method, **extra)

    return handler


def register_handlers(bot_app: TelegramApplication, facade: TelegramFacade) -> None:
    """Attach every command and callback route to the Telegram application."""
    bot_app.bot_data["facade"] = facade

    commands = {
        "start": "start",
        "help": "help",
        "status": "status",
        "modo": "modo",
        "analizar": "analizar",
        "oportunidades": "oportunidades",
        "detalles": "detalles",
        "riesgo": "riesgo",
        "cuenta": "cuenta",
        "portfolio": "portfolio",
        "posiciones": "posiciones",
        "ordenes": "ordenes",
        "comprar": "comprar",
        "vender": "vender",
        "cerrar": "cerrar",
        "cancelar": "cancelar",
        "modificar": "modificar",
        "stats": "stats",
        "historial": "historial",
        "explicar": "explicar",
        "aprender": "aprender",
        "monitor": "monitor",
        "config": "config",
        "reconciliar": "reconciliar",
    }
    bot_app.bot_data["commands"] = commands
    for command, method in commands.items():
        bot_app.add_handler(CommandHandler(command, _make_command(command, method)))

    bot_app.add_handler(CallbackQueryHandler(_on_callback))
    bot_app.add_error_handler(_on_error)


# Telegram shows these in the command picker as soon as the user types "/", so
# nobody has to memorise the command list. Descriptions are shown by clients that
# support the newer BotCommandScope features and ignored by older ones.
COMMAND_DESCRIPTIONS: dict[str, str] = {
    "start": "Bienvenida y estado del bot",
    "help": "Índice de todos los comandos",
    "status": "Salud: base de datos, Alpaca, scheduler, kill switch",
    "modo": "Ver o cambiar entre ALPACA PAPER y ALPACA REAL",
    "analizar": "Analizar símbolos o escanear el universo",
    "oportunidades": "Últimas oportunidades detectadas",
    "detalles": "Detalle completo de un símbolo",
    "riesgo": "Límites de riesgo y exposición actual",
    "cuenta": "Equity, cash, buying power y valor de cartera",
    "portfolio": "Exposiciones y P&L no realizado",
    "posiciones": "Posiciones abiertas con su protección",
    "ordenes": "Órdenes abiertas",
    "comprar": "Proponer una compra (requiere confirmación)",
    "vender": "Proponer una venta (requiere confirmación)",
    "cerrar": "Cerrar una posición (requiere confirmación)",
    "cancelar": "Cancelar una orden o la confirmación pendiente",
    "modificar": "Reemplazar cantidad, stop o límite de una orden",
    "stats": "Estadísticas históricas del entorno activo",
    "historial": "Operaciones cerradas",
    "explicar": "Por qué el bot recomendó un símbolo",
    "aprender": "Recomendaciones del motor estadístico",
    "monitor": "Activar o pausar los monitores en segundo plano",
    "config": "Configuración efectiva (sin secretos)",
    "reconciliar": "Reconciliar SQLite contra Alpaca ahora",
}


async def publish_commands(bot: object) -> int:
    """Publish the command list to Telegram so `/` shows every command.

    Returns the number of commands published. Failures are logged and swallowed:
    a bot that cannot publish its menu must still run.
    """
    commands = [
        BotCommand(command=command, description=description)
        for command, description in COMMAND_DESCRIPTIONS.items()
    ]
    try:
        await bot.set_my_commands(commands)
    except TelegramError:
        log.warning("telegram.set_commands_failed", exc_info=True)
        return 0
    log.info("telegram.commands_published", count=len(commands))
    return len(commands)


# ------------------------------------------------------------------ callbacks


async def _on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None:
        return
    facade: TelegramFacade = context.application.bot_data["facade"]
    action, _, payload = query.data.partition(":")
    try:
        result = await facade.guard(_route, update, action, payload, facade)
    except CommandError as exc:
        await query.answer()
        await _answer(update, context, f"⚠️ {exc.user_message}")
        return
    await query.answer()
    await _answer(update, context, result.text, result.keyboard)


async def _route(
    update: Update, action: str, payload: str, facade: TelegramFacade
) -> CommandResult:
    """Dispatch one button press to the right guarded operation."""
    if action == kb.MODE_SWITCH:
        return await facade.modo(update, payload)
    if action == kb.CONFIRM_ENV:
        return await facade.confirm_mode_switch(update, TradingEnvironment(payload))
    if action == kb.CONFIRM_TRADE:
        return await facade.confirm_trade(update, payload)
    if action == kb.CANCEL_PENDING:
        return await facade.cancelar_pendiente(update)
    if action == kb.ACCEPT_SIGNAL:
        return await facade.comprar(update, payload)
    if action == kb.REJECT_SIGNAL:
        return await facade.rechazar(update, payload)
    if action == kb.SIGNAL_DETAILS:
        return await facade.detalles(update, payload)
    if action == kb.RISK_DETAIL:
        return await facade.riesgo_detalle(update, payload)
    if action == kb.STATS_DETAIL:
        return await facade.stats(update, payload)
    if action == kb.CLOSE_POSITION:
        return await facade.cerrar(update, payload)
    if action == kb.BREAK_EVEN:
        return await facade.break_even(update, payload)
    if action == kb.TRAILING:
        return await facade.trailing(update, payload)
    if action == kb.CANCEL_ORDER:
        return await facade.cancelar_orden(update, payload)
    if action == kb.REPLACE_ORDER:
        return await facade.modificar(update, payload)
    if action == "cerrar" or action == "cerrar_posicion":
        return await facade.confirm_close(update, *payload.split(":", 1))
    if action == "cancelar_orden":
        return await facade.confirm_cancel(update, payload)
    if action == "modificar":
        return await facade.confirm_modify(update, payload)
    if action == "break_even":
        return await facade.confirm_break_even(update, payload)
    if action == "trailing":
        return await facade.confirm_trailing(update, payload)
    if action == "kill_switch":
        return await facade.confirm_kill_switch(update, payload)
    if action in {"long", "short"}:
        method = facade.comprar if action == "long" else facade.vender
        return await method(update, payload)
    if action == kb.POSITION_MENU:
        return await facade.position_menu(update, payload)
    if action == kb.REFRESH:
        if payload == "pos":
            return await facade.posiciones(update, None)
        if payload == "portfolio":
            return await facade.portfolio(update, None)
        if payload == "orders":
            return await facade.ordenes(update, None)
        return await facade.posiciones(update, None)
    if action == kb.ADD_POSITION:
        return await facade._propose(update, payload, direction="long")
    if action == kb.REDUCE_POSITION:
        return await facade._propose(update, payload, direction="short")
    raise CommandError(f"Acción desconocida: {action}")


def _uid(update: Update) -> int:
    user = update.effective_user
    return int(user.id) if user else 0


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("telegram.handler_error", error=str(context.error))
    chat = getattr(update, "effective_chat", None)
    if chat is not None:
        try:
            await context.bot.send_message(
                chat.id,
                "❌ Ocurrió un error procesando el comando. Revisa los logs.",
            )
        except TelegramError:
            pass


def build_telegram_application(token: str) -> TelegramApplication:
    """Create the Telegram application with sane defaults.

    ``ApplicationBuilder`` has no ``parse_mode`` setter; the supported way to
    set a default parse mode is through ``Defaults``.
    """
    return (
        ApplicationBuilder()
        .token(token)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .concurrent_updates(True)
        .build()
    )


__all__ = [
    "build_telegram_application",
    "register_handlers",
]
