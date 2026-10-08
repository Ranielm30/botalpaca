"""The operator must be told when a position cannot be protected.

Two failures were live at the same time:

* ``_clear_inert_orders`` reported success as soon as Alpaca *accepted* the
  cancel, not when Alpaca *completed* it. Reconcile printed "ordenes inertes
  liberadas" and then failed on the very next submit with the shares still held,
  every cycle, forever.
* ``Application.reconcile`` wrote those notes to the log and nowhere else. A
  position that cannot be protected is the most important thing this process
  knows, and the operator could not see it: silence on a stuck position reads
  exactly like "everything is fine".

These tests pin both halves.
"""

from __future__ import annotations

import pytest

from botalpaca.app import Application
from botalpaca.domain.enums import TradingEnvironment
from tests.test_protection import PAPER, _setup, _stop_order, make_position

HELD = "held"


def _raw(order, status: str):
    order.status = status
    return order


async def test_clearing_an_inert_order_waits_for_the_shares_to_come_back(database):
    """The cancel must not be reported as done until the broker lets go."""
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)

    client.orders["stuck-1"] = _raw(_stop_order("stuck-1", stop=99.0), HELD)

    probes = {"n": 0}

    async def fake_live(symbol):
        # The first two probes still show the order holding, exactly as Alpaca
        # behaves between accepting and completing a cancel.
        probes["n"] += 1
        if probes["n"] <= 2:
            return [_to_state(_raw(_stop_order("stuck-1", stop=99.0), HELD))]
        return []

    engine.get_live_orders_for_symbol = fake_live  # type: ignore[method-assign]

    state = await protection.state_for_position(PAPER, position)
    assert state.inert_order_ids == ["stuck-1"]

    stuck = await protection._clear_inert_orders(PAPER, position, state)

    assert stuck == []
    assert probes["n"] >= 3, "it must poll until the shares are actually free"
    assert any("liberadas" in note for note in state.notes), state.notes


async def test_an_inert_order_that_never_frees_is_reported_as_stuck(database):
    """No false success: the caller must learn the shares are still held."""
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)
    client.orders["wedged"] = _raw(_stop_order("wedged", stop=99.0), HELD)

    async def always_held(symbol):
        return [_to_state(_raw(_stop_order("wedged", stop=99.0), HELD))]

    engine.get_live_orders_for_symbol = always_held  # type: ignore[method-assign]

    state = await protection.state_for_position(PAPER, position)
    stuck = await protection._clear_inert_orders(
        PAPER, position, state, attempts=2
    ) if "attempts" in protection._clear_inert_orders.__code__.co_varnames else None

    if stuck is None:
        # The helper does not take attempts; drive the wait to its ceiling.
        stuck = await protection._clear_inert_orders(PAPER, position, state)

    assert stuck == ["wedged"], "a cancel that never completes is stuck, not cleared"
    assert not any("liberadas" in note for note in state.notes), state.notes
    assert state.inert_order_ids == ["wedged"]


async def test_reconcile_says_so_when_inert_orders_stay_stuck(database):
    """The operator gets a loud note, not a quiet one."""
    manager, engine, client = await _setup(database)
    protection = manager
    position = make_position(entry=100.0, current=101.0)
    client.orders["wedged"] = _raw(_stop_order("wedged", stop=99.0), HELD)

    async def always_held(symbol):
        return [_to_state(_raw(_stop_order("wedged", stop=99.0), HELD))]

    engine.get_live_orders_for_symbol = always_held  # type: ignore[method-assign]

    notes = await protection.reconcile(environment=PAPER, positions=[position])

    assert any("🚨" in n and "ancladas" in n for n in notes), notes


def _to_state(raw):
    from botalpaca.execution.mapping import to_order_state

    return to_order_state(raw, PAPER)


# -- the operator's notice ------------------------------------------------------------
class _Notifications:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bool]] = []

    async def send(self, text, keyboard=None, *, force=False):
        self.sent.append((text, force))
        return True


class _App:
    _reconcile_state = None
    def __init__(self) -> None:
        self.notifications = _Notifications()
        self.active_environment = TradingEnvironment.PAPER


async def test_reconciliation_problems_reach_the_operator():
    from botalpaca.app import Application

    app = _App()
    notes = ["🚨 MSFT: 1 orden(es) inertes retienen las acciones", "🧹 MSFT: ordenes inertes liberadas"]

    await Application._notify_reconciliation(app, TradingEnvironment.PAPER, notes)

    assert len(app.notifications.sent) == 1
    text, force = app.notifications.sent[0]
    # Only the actionable ones; the housekeeping note stays in the log.
    assert "🚨 MSFT" in text
    assert "🧹" not in text
    assert "PAPER" in text
    assert force is True, "a position that cannot be protected is not chatter"


async def test_a_clean_reconciliation_says_nothing():
    from botalpaca.app import Application

    app = _App()
    notes = ["🧹 MSFT: ordenes inertes liberadas", "🛡️ AAPL: stop de emergencia creado"]

    await Application._notify_reconciliation(app, TradingEnvironment.PAPER, notes)

    assert app.notifications.sent == []


async def test_a_take_profit_close_reaches_the_operator():
    """The position is gone and the trade is recorded. Silence on that is the
    one thing an operator cannot interpret: it reads exactly like still open."""
    from botalpaca.app import Application

    app = _App()
    note = "📕 LLY: cerrada en el ledger por TARGET a 1,186.14 (P&L +182.48, R +0.08)"

    await Application._notify_reconciliation(app, TradingEnvironment.PAPER, [note])

    assert len(app.notifications.sent) == 1
    text, _ = app.notifications.sent[0]
    assert "Operaciones" in text
    assert "TARGET" in text
    assert "+182.48" in text


async def test_a_stop_close_reaches_the_operator():
    from botalpaca.app import Application

    app = _App()
    note = "📕 AVGO: cerrada en el ledger por STOP a 365.26 (P&L -350.00, R -1.00)"

    await Application._notify_reconciliation(app, TradingEnvironment.PAPER, [note])

    text, _ = app.notifications.sent[0]
    assert "STOP" in text
    assert "-350.00" in text


async def test_a_close_and_an_alarm_each_get_their_own_section():
    """A close must not be buried inside an alarm, and an alarm must not be
    diluted by routine bookkeeping."""
    from botalpaca.app import Application

    app = _App()
    notes = [
        "📕 QQQ: cerrada en el ledger por STOP a 741.40 (P&L -500.00, R -1.00)",
        "🚨 MSFT: 1 orden(es) inertes retienen las acciones",
    ]

    await Application._notify_reconciliation(app, TradingEnvironment.PAPER, notes)

    text, _ = app.notifications.sent[0]
    assert text.index("Operaciones") < text.index("Aviso de proteccion")
    assert "STOP" in text and "🚨" in text
    # The footer about held shares only belongs with the alarm.
    assert text.count("retiene las acciones") == 1


async def test_a_repeated_close_is_only_sent_once():
    from botalpaca.app import Application

    app = _App()
    note = "📕 LLY: cerrada en el ledger por TARGET a 1,186.14 (P&L +182.48, R +0.08)"

    await Application._notify_reconciliation(app, TradingEnvironment.PAPER, [note])
    await Application._notify_reconciliation(app, TradingEnvironment.PAPER, [note])

    assert len(app.notifications.sent) == 1


async def test_a_broken_notifier_never_breaks_reconciliation():
    from botalpaca.app import Application

    class _Broken:
        async def send(self, text, keyboard=None, *, force=False):
            raise RuntimeError("Telegram caido")

    app = _App()
    app.notifications = _Broken()

    # Must not raise: reconciliation runs on a scheduler.
    await Application._notify_reconciliation(
        app, TradingEnvironment.PAPER, ["❌ AAPL: no se pudo crear stop"]
    )


@pytest.mark.parametrize(
    "note",
    ["🚨 AAPL: retenida", "❌ AAPL: no se pudo crear stop automático"],
)
async def test_both_levels_of_problem_are_actionable(note):
    from botalpaca.app import Application

    app = _App()
    await Application._notify_reconciliation(app, TradingEnvironment.PAPER, [note])
    text = app.notifications.sent[0][0]
    # The card puts the symbol on its own line and the reason under it.
    symbol, _, reason = note.partition(": ")
    assert symbol in text
    assert reason in text


async def test_the_notice_never_shows_raw_broker_json():
    """The operator reads a sentence, not an Alpaca payload."""
    from botalpaca.protection.manager import readable_reason

    raw = (
        'Alpaca rejected the request: {"available":"0","code":40310000,'
        '"existing_qty":"4","held_for_orders":4,"message":"insufficient qty '
        'available for order (requested: 4, available: 0)","symbol":"MSFT"}'
    )
    reason = readable_reason(raw)
    assert "insufficient qty available" not in reason
    assert "{" not in reason and "held_for_orders" not in reason
    assert "mercado abra" in reason


def test_an_unknown_broker_error_is_still_readable():
    from botalpaca.protection.manager import readable_reason

    reason = readable_reason("something nobody predicted")
    assert "{" not in reason
    assert len(reason) <= 140


async def test_the_same_problem_is_announced_once():
    class _Notifications:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, text, keyboard=None, *, force=False):
            self.sent.append(text)
            return True

    class _App:
        _reconcile_state = None

    app = _App()
    app.notifications = _Notifications()
    notify = Application._notify_reconciliation
    notes = ["\U0001F6A8 MSFT: ancladas en Alpaca retienen las acciones"]

    await notify(app, PAPER, notes)
    await notify(app, PAPER, notes)
    await notify(app, PAPER, notes)
    assert len(app.notifications.sent) == 1, "the same problem must not repeat"


async def test_a_changed_problem_is_announced_again():
    class _Notifications:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, text, keyboard=None, *, force=False):
            self.sent.append(text)
            return True

    class _App:
        _reconcile_state = None

    app = _App()
    app.notifications = _Notifications()
    notify = Application._notify_reconciliation

    await notify(app, PAPER, ["\U0001F6A8 MSFT: ancladas en Alpaca"])
    await notify(app, PAPER, ["\U0001F6A8 NVDA: ancladas en Alpaca"])
    assert len(app.notifications.sent) == 2


async def test_the_notice_explains_what_happens_next():
    class _Notifications:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, text, keyboard=None, *, force=False):
            self.sent.append(text)
            return True

    class _App:
        _reconcile_state = None

    app = _App()
    app.notifications = _Notifications()
    await Application._notify_reconciliation(app, PAPER, ["\u274C MSFT: SIN stop"])
    text = app.notifications.sent[0]
    assert "MSFT" in text
    assert "SIN stop" in text
    assert "mercado abra" in text
