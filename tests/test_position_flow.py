"""Tests for the position-picker flow and the card formatting helpers."""

from __future__ import annotations

from botalpaca.domain.enums import OrderSide, TradingEnvironment
from botalpaca.notifications import format as fmt
from botalpaca.notifications.service import render_orders, render_positions
from botalpaca.telegram import keyboards as kb

PAPER = TradingEnvironment.PAPER
REAL = TradingEnvironment.REAL


class _Protection:
    """Minimal stand-in for a ProtectionState."""

    def __init__(self, **kw):
        self.has_stop = kw.get("has_stop", True)
        self.has_take_profit = kw.get("has_take_profit", False)
        self.has_trailing = kw.get("has_trailing", False)
        self.stop_price = kw.get("stop_price", 97.0)
        self.take_profit_price = kw.get("take_profit_price", None)
        self.trail_percent = kw.get("trail_percent", None)
        self.break_even_active = kw.get("break_even_active", False)
        self.time_stop_at = kw.get("time_stop_at", None)
        self.notes = kw.get("notes", [])


class _Position:
    def __init__(self, symbol, qty=10.0, entry=100.0, current=104.0, pl=40.0, plpc=0.04):
        self.symbol = symbol
        self.qty = qty
        self.side = OrderSide.BUY
        self.avg_entry_price = entry
        self.current_price = current
        self.market_value = current * abs(qty)
        self.unrealized_pl = pl
        self.unrealized_plpc = plpc


# -- format helpers -------------------------------------------------------------------
def test_esc_escapes_markup():
    assert fmt.esc("<b>x</b>") == "&lt;b&gt;x&lt;/b&gt;"


def test_row_has_a_visible_separator():
    out = fmt.row("P&L", "$10.00")
    assert "P&amp;L" in out
    assert fmt.ARROW in out


def test_header_always_has_a_divider():
    out = fmt.header("Titulo", "🟢 ALPACA PAPER")
    assert fmt.RULE in out
    assert "ALPACA PAPER" in out


def test_meter_is_bounded():
    assert fmt.meter(0.0).count("█") == 0
    assert fmt.meter(1.0).count("█") == 10
    assert fmt.meter(5.0).count("█") == 10


def test_chunk_splits_on_line_boundaries():
    text = "\n".join("linea " + "x" * 100 for _ in range(50))
    parts = fmt.chunk(text, 500)
    assert len(parts) > 1
    assert all(len(p) <= 500 for p in parts)
    # No line is cut in half.
    joined = "\n".join(parts)
    assert joined == text


def test_trend_glyph():
    assert fmt.trend_glyph(1.0) == fmt.UP
    assert fmt.trend_glyph(-1.0) == fmt.DOWN
    assert fmt.trend_glyph(0.0) == fmt.FLAT
    assert fmt.trend_glyph(None) == fmt.FLAT


# -- position cards --------------------------------------------------------------------
def test_render_positions_empty():
    out = render_positions([], PAPER)
    assert "Sin posiciones" in out


def test_render_positions_shows_each_block():
    positions = [_Position("AAPL"), _Position("TSLA", entry=200.0, current=210.0, pl=100.0, plpc=0.05)]
    out = render_positions(positions, PAPER)
    assert "AAPL" in out and "LONG" in out
    assert "TSLA" in out
    # The reference layout puts one divider above the accumulated total.
    assert fmt.RULE in out
    assert "Entrada:" in out and "Actual:" in out
    assert "Total acumulado" in out


def test_render_positions_marks_unprotected():
    positions = [_Position("AAPL")]
    out = render_positions(positions, PAPER, protection={"AAPL": _Protection(has_stop=False)})
    assert "SIN STOP" in out


def test_render_positions_marks_protected():
    positions = [_Position("AAPL")]
    out = render_positions(positions, PAPER, protection={"AAPL": _Protection(has_stop=True)})
    assert "stop" in out.lower()


def test_render_positions_breaks_even():
    positions = [_Position("AAPL")]
    out = render_positions(
        positions, PAPER, protection={"AAPL": _Protection(break_even_active=True)}
    )
    assert "break-even" in out


def test_a_short_position_is_labelled_short():
    """A negative quantity must render as SHORT, never as LONG.

    ``_side_label`` used to return "LONG" for anything it did not
    recognise, so a short whose ``side`` field did not match its set was
    displayed as a long. The card now reads the sign of ``qty``.
    """
    short = _Position("SBUX", qty=-59.0, entry=89.98, current=90.92,
                      pl=-55.46, plpc=-0.0616)
    short.side = OrderSide.SELL
    out = render_positions([short], PAPER)
    assert "SHORT" in out
    assert "LONG" not in out


def test_a_long_position_is_labelled_long():
    long_ = _Position("AAPL")
    out = render_positions([long_], PAPER)
    assert "LONG" in out
    assert "SHORT" not in out


# -- order cards ----------------------------------------------------------------------
def test_render_orders_empty():
    out = render_orders([], PAPER)
    assert "Sin ordenes" in out


def test_render_orders_custom_title():
    orders = []
    assert "Cerradas" in render_orders(orders, PAPER, title="Cerradas")


# -- keyboards ------------------------------------------------------------------------
def test_positions_picker_has_one_button_per_position():
    kb_ = kb.positions_picker_keyboard([("AAPL", 0.01), ("TSLA", -0.02)], environment=PAPER)
    payloads = [b.callback_data for row in kb_.inline_keyboard for b in row]
    assert f"{kb.POSITION_MENU}:AAPL" in payloads
    assert f"{kb.POSITION_MENU}:TSLA" in payloads
    # Green for a winner, red for a loser.
    labels = [b.text for row in kb_.inline_keyboard for b in row]
    assert any("🟢" in t for t in labels)
    assert any("🔴" in t for t in labels)


def test_positions_picker_has_refresh_button():
    kb_ = kb.positions_picker_keyboard([("AAPL", 0.0)], environment=PAPER)
    payloads = [b.callback_data for row in kb_.inline_keyboard for b in row]
    assert f"{kb.REFRESH}:pos" in payloads


def test_positions_picker_empty_shows_refresh():
    kb_ = kb.positions_picker_keyboard([], environment=PAPER)
    payloads = [b.callback_data for row in kb_.inline_keyboard for b in row]
    assert f"{kb.REFRESH}:pos" in payloads


def test_positions_picker_switches_environment():
    kb_ = kb.positions_picker_keyboard([("AAPL", 0.0)], environment=PAPER)
    payloads = [b.callback_data for row in kb_.inline_keyboard for b in row]
    assert f"{kb.MODE_SWITCH}:{REAL.value}" in payloads


def test_position_actions_has_all_the_operations():
    kb_ = kb.position_actions_keyboard("AAPL")
    payloads = [b.callback_data for row in kb_.inline_keyboard for b in row]
    for action in (kb.BREAK_EVEN, kb.TRAILING, kb.ADD_POSITION, kb.REDUCE_POSITION,
                   kb.REPLACE_ORDER, kb.CANCEL_ORDER, kb.CLOSE_POSITION):
        assert f"{action}:AAPL" in payloads


def test_position_actions_can_go_back():
    kb_ = kb.position_actions_keyboard("AAPL")
    payloads = [b.callback_data for row in kb_.inline_keyboard for b in row]
    assert f"{kb.REFRESH}:pos" in payloads


def test_all_callback_payloads_fit_the_telegram_limit():
    kb_ = kb.position_actions_keyboard("BRK.B")
    for row in kb_.inline_keyboard:
        for button in row:
            assert len(button.callback_data.encode()) <= kb.MAX_CALLBACK_BYTES


def test_mode_button_label_points_at_the_other_environment():
    assert "REAL" in kb.mode_button_label(PAPER)
    assert "PAPER" in kb.mode_button_label(REAL)




def test_position_menu_renders_protection_state():
    from botalpaca.notifications.service import _protection_line

    line = _protection_line(_Protection(has_stop=True, stop_price=97.0))
    assert "97" in line
