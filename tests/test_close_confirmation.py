"""The close confirmation must not be mistaken for the close command.

``/cerrar AAPL`` renders a confirmation whose button carries
``cerrar_ok:AAPL:100``.  The position menu's own button carries ``cerrar:AAPL``.

Both prefixes used to be ``cerrar``, so the menu branch matched first and fed a
confirmation token such as ``AAPL:100`` into the symbol parser. The operator got
``'AAPL:100' no parece un símbolo válido`` and could not close a position at all
-- with the bot refusing to keep their take-profit alive.
"""

from __future__ import annotations

import inspect

from botalpaca.telegram import bot as bot_module
from botalpaca.telegram import keyboards as kb
from botalpaca.telegram.service import parse_symbol


def test_the_confirmation_uses_its_own_prefix():
    assert kb.CLOSE_CONFIRM != kb.CLOSE_POSITION
    assert kb.CLOSE_CONFIRM == "cerrar_ok"


def test_a_confirmation_token_is_never_valid_as_a_symbol():
    """This is the exact string the operator saw."""
    from botalpaca.telegram.service import CommandError

    try:
        parse_symbol("AAPL:100")
    except CommandError as exc:
        assert "no parece un símbolo válido" in str(exc)
    else:
        raise AssertionError("a confirmation token must not parse as a symbol")


def test_the_close_branch_routes_the_confirmation_first():
    """Order matters: the confirmation prefix must be matched before the menu one."""
    source = inspect.getsource(bot_module)
    confirm_at = source.index("kb.CLOSE_CONFIRM")
    menu_at = source.index("if action == kb.CLOSE_POSITION:")
    assert confirm_at < menu_at, (
        "the position-menu branch swallows the confirmation button and passes "
        "its token to the symbol parser"
    )


def test_cerrar_emits_the_confirmation_prefix():
    source = inspect.getsource(
        __import__("botalpaca.telegram.service", fromlist=["TelegramFacade"]).TelegramFacade.cerrar
    )
    assert "kb.CLOSE_CONFIRM" in source
    assert 'confirm_keyboard(action="cerrar"' not in source


def test_the_menu_button_still_stages_a_close():
    markup = kb.position_actions_keyboard("AAPL")
    payloads = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert f"{kb.CLOSE_POSITION}:AAPL" in payloads


def test_the_two_payloads_are_distinguishable():
    """A symbol for the menu, symbol+scope for the confirmation."""
    menu = f"{kb.CLOSE_POSITION}:AAPL"
    confirm = f"{kb.CLOSE_CONFIRM}:AAPL:100"
    assert menu.split(":", 1)[0] != confirm.split(":", 1)[0]
    # The confirmation reaches confirm_close(symbol, scope) via exactly one split.
    _, payload = confirm.split(":", 1)
    symbol, scope = payload.split(":", 1)
    assert symbol == "AAPL"
    assert scope == "100"
