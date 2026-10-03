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

    assert any("🚨" in n and "inerte" in n for n in notes), notes


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
    assert note in app.notifications.sent[0][0]
