"""Shared card formatting for Telegram.

Telegram renders every message in a proportional client font, so aligned
"columns" built with spaces or tabs look broken on a phone. The only reliable
way to get a clean, scannable layout in Telegram is a single-column list with
explicit separators and labels.

Every helper here returns Telegram HTML and escapes the values it interpolates,
so a symbol, strategy name or broker message can never inject markup.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

# Telegram caps a message at 4096 characters; we split well below that so the
# keyboard never gets detached from the last chunk of text.
MAX_MESSAGE_CHARS = 3900
# Long analyses are split into chunks of at most this size.
CHUNK_CHARS = 3800

# Horizontal rules. Telegram HTML has no <hr>, so a line of light box-drawing
# characters is the closest thing to a divider and renders identically
# everywhere.
RULE = "━━━━━━━━━━━━━━━━━━━━"
RULE_THIN = "────────────────────"

# Glyphs chosen because they are plain Unicode and never mangled by the build:
# no combining characters, no emoji variation selectors, no zero-width joiners.
BULL = "•"
ARROW = "›"
UP = "▲"
DOWN = "▼"
FLAT = "•"
BULLSEYE = "🎯"
SHIELD = "🛡"
CHART = "📊"
INFO = "ℹ"
WARN = "⚠"
ROCKET = "🚀"
LOCK = "🔒"
CHAIN = "🔗"
SCISSORS = "🔪"
FLAME = "🔥"
HAND = "👋"
EYES = "👀"
PACK = "📦"
PEN = "📝"
CROSS = "🚫"
PIN = "📌"
SIREN = "🚨"
FLAG = "\U0001F3C1"
BELL = "\U0001F514"
BRAIN = "\U0001F9E0"
ENTRY = "\U0001F3AF"
SCALE = "\u2696\uFE0F"
BULB = "\U0001F4A1"
STAR = "\u2B50"
GREEN = "\U0001F7E2"
YELLOW = "\U0001F7E8"
RED = "\U0001F534"
QUALITY_GLYPH = {"ALTA": GREEN, "MEDIA": YELLOW, "BAJA": RED}


TARGET = "🎯"
CLOCK = "🕐"


def esc(value: object) -> str:
    """Escape a value for Telegram's HTML parse mode."""
    return html.escape(str(value), quote=False)


def trend_glyph(value: float | None) -> str:
    if value is None:
        return FLAT
    if value > 0:
        return UP
    if value < 0:
        return DOWN
    return FLAT


def kv(label: str, value: object, *, indent: int = 0) -> str:
    """One ``Label: value`` line with consistent indentation."""
    pad = "  " * indent
    return f"{pad}{BULL} {esc(label)}: <b>{esc(value)}</b>"


def row(left: str, right: str) -> str:
    """A label/value pair on one line, e.g. ``P&L`` and ``+$120.50``.

    Telegram collapses runs of spaces in HTML, so the two halves are separated
    with a real separator character rather than padding.
    """
    return f"{esc(left)} {ARROW} <b>{esc(right)}</b>"


def header(title: str, environment_label: str | None = None) -> str:
    """Standard card header: optional environment badge, title, divider."""
    parts: list[str] = []
    if environment_label:
        parts.append(f"<b>{esc(environment_label)}</b>")
    parts.append(f"<b>{esc(title)}</b>")
    parts.append(RULE)
    return "\n".join(parts)


def section(title: str) -> str:
    return f"\n<b>{esc(title)}</b>"


def bullets(items: Sequence[str], *, indent: int = 1) -> list[str]:
    pad = "  " * indent
    return [f"{pad}{BULL} {esc(item)}" for item in items]


def meter(value: float, *, width: int = 10, filled: str = "█", empty: str = "░") -> str:
    """A simple progress bar. Used for confluence and risk budgets."""
    ratio = min(max(value, 0.0), 1.0)
    return filled * int(round(ratio * width)) + empty * (width - int(round(ratio * width)))


def chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split a long card on line boundaries so no field is cut in half."""
    if len(text) <= size:
        return [text]
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        addition = len(line) + 1
        if current and length + addition > size:
            chunks.append("\n".join(current))
            current = []
            length = 0
        current.append(line)
        length += addition
    if current:
        chunks.append("\n".join(current))
    return chunks


__all__ = [
    "ARROW",
    "BULL",
    "BULLSEYE",
    "CHART",
    "CHUNK_CHARS",
    "DOWN",
    "FLAT",
    "INFO",
    "MAX_MESSAGE_CHARS",
    "RULE",
    "RULE_THIN",
    "SHIELD",
    "UP",
    "WARN",
    "bullets",
    "chunk",
    "esc",
    "header",
    "kv",
    "meter",
    "row",
    "section",
    "trend_glyph",
]
