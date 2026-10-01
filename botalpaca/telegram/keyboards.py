"""Inline keyboards and callback-data helpers for the Telegram interface.

Callback data is a strict, machine-readable envelope so that a stale button can
never execute something the user did not intend. Every payload is validated
again in the handler.
"""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from botalpaca.domain import TradingEnvironment

#: Telegram limits callback_data to 64 bytes.
MAX_CALLBACK_BYTES = 64

MODE_SWITCH = "modo"
ACCEPT_SIGNAL = "aceptar"
REJECT_SIGNAL = "rechazar"
SIGNAL_DETAILS = "detalles"
CONFIRM_TRADE = "confirmar"
CANCEL_PENDING = "cancelar"
CLOSE_POSITION = "cerrar"
BREAK_EVEN = "be"
TRAILING = "trail"
CANCEL_ORDER = "cancelar_orden"
REPLACE_ORDER = "modificar"
RISK_DETAIL = "riesgo"
STATS_DETAIL = "stats"
CONFIRM_ENV = "confirmar_modo"
HELP = "ayuda"

#: Token typed by the user to confirm a live-money action.
REAL_CONFIRM_TOKEN = "REAL"


def mode_button(environment: TradingEnvironment) -> InlineKeyboardButton:
    """The opposite-environment switch button shown by /modo."""
    target = environment.other
    label = (
        "🔴 CAMBIAR A ALPACA REAL"
        if target.is_real
        else "🟢 CAMBIAR A ALPACA PAPER"
    )
    return InlineKeyboardButton(label, callback_data=f"{MODE_SWITCH}:{target.value}")


def mode_keyboard(environment: TradingEnvironment) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[mode_button(environment)]])


def signal_keyboard(symbol: str) -> InlineKeyboardMarkup:
    """ACEPTAR / RECHAZAR / DETALLES, as required for monitored signals."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ ACEPTAR", callback_data=f"{ACCEPT_SIGNAL}:{symbol}"),
                InlineKeyboardButton("❌ RECHAZAR", callback_data=f"{REJECT_SIGNAL}:{symbol}"),
            ],
            [
                InlineKeyboardButton("🔎 DETALLES", callback_data=f"{SIGNAL_DETAILS}:{symbol}"),
                InlineKeyboardButton("ℹ️ Estadísticas", callback_data=f"{STATS_DETAIL}:{symbol}"),
                InlineKeyboardButton("⚠️ Riesgo", callback_data=f"{RISK_DETAIL}:{symbol}"),
            ],
        ]
    )


def confirm_keyboard(
    *,
    action: str,
    token: str,
    cancel_label: str = "CANCELAR",
) -> InlineKeyboardMarkup:
    """Two-button confirmation row: confirm + cancel, never a single tap."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ CONFIRMAR", callback_data=f"{action}:{token}"),
                InlineKeyboardButton(f"✖ {cancel_label}", callback_data=f"{CANCEL_PENDING}:{token}"),
            ]
        ]
    )


def real_confirm_keyboard(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔴 CONFIRMAR OPERACIÓN REAL",
                    callback_data=f"{CONFIRM_TRADE}:real:{symbol}",
                ),
                InlineKeyboardButton("✖ CANCELAR", callback_data=f"{CANCEL_PENDING}:real:{symbol}"),
            ]
        ]
    )


def paper_confirm_keyboard(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ CONFIRMAR ORDEN PAPER",
                    callback_data=f"{CONFIRM_TRADE}:paper:{symbol}",
                ),
                InlineKeyboardButton("✖ CANCELAR", callback_data=f"{CANCEL_PENDING}:paper:{symbol}"),
            ]
        ]
    )


def env_confirm_keyboard(target: TradingEnvironment) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"Confirmar {target.value}",
                    callback_data=f"{CONFIRM_ENV}:{target.value}",
                ),
                InlineKeyboardButton("✖ CANCELAR", callback_data=f"{CANCEL_PENDING}:env"),
            ]
        ]
    )


def position_keyboard(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "⚖️ Break-even", callback_data=f"{BREAK_EVEN}:{symbol}"
                ),
                InlineKeyboardButton(
                    "📉 Activar trailing", callback_data=f"{TRAILING}:{symbol}"
                ),
                InlineKeyboardButton(
                    "🔴 Cerrar", callback_data=f"{CLOSE_POSITION}:{symbol}"
                ),
            ]
        ]
    )


def order_keyboard(order_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✖ Cancelar orden", callback_data=f"{CANCEL_ORDER}:{order_id}"
                ),
                InlineKeyboardButton(
                    "✏️ Modificar", callback_data=f"{REPLACE_ORDER}:{order_id}"
                ),
            ]
        ]
    )


def side_keyboard(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🟢 Comprar (LONG)", callback_data=f"long:{symbol}"
                ),
                InlineKeyboardButton(
                    "🔴 Vender (SHORT)", callback_data=f"short:{symbol}"
                ),
            ]
        ]
    )


def back_keyboard(environment: TradingEnvironment) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[mode_button(environment)]])


__all__ = [
    "ACCEPT_SIGNAL",
    "BREAK_EVEN",
    "CANCEL_ORDER",
    "CANCEL_PENDING",
    "CLOSE_POSITION",
    "CONFIRM_ENV",
    "CONFIRM_TRADE",
    "HELP",
    "MAX_CALLBACK_BYTES",
    "MODE_SWITCH",
    "REAL_CONFIRM_TOKEN",
    "REJECT_SIGNAL",
    "REPLACE_ORDER",
    "RISK_DETAIL",
    "SIGNAL_DETAILS",
    "STATS_DETAIL",
    "TRAILING",
    "back_keyboard",
    "confirm_keyboard",
    "env_confirm_keyboard",
    "mode_button",
    "mode_keyboard",
    "order_keyboard",
    "paper_confirm_keyboard",
    "position_keyboard",
    "real_confirm_keyboard",
    "side_keyboard",
    "signal_keyboard",
]
