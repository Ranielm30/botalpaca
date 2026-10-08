"""The ledger must never disagree with the broker about who is protecting a position.

A live position with no stop in the broker has no downside protection at all,
and a row that claims a stop the broker does not hold is worse than no row:
everything downstream (break-even, trailing, autonomy, the learning loop)
reasons about a stop that does not exist.

That failure once looked real -- an offline simulation showed a position with a
live exit in the broker and ``stop=None`` in SQLite -- and it turned out the
simulation was wrong, not the bot. This file pins the invariant so the question
does not have to be reopened, and so the next person to see a ``None`` here
learns it is a genuine defect and not a harness artefact.

The broker refuses orders here the way Alpaca actually refuses them: with
``insufficient qty available ... held_for_orders``. That refusal is what drove
the release-and-restore logic in ``_merge_into_oco``, so it is the only refusal
worth testing the rollback against.
"""

from __future__ import annotations

import pytest

from botalpaca.db import Database, ProtectionRepository
from botalpaca.domain import TradingEnvironment
from botalpaca.execution.engine import ExecutionEngine
from botalpaca.protection.manager import PositionProtectionManager

from .conftest import FakeTradingClient, fake_order_raw, make_position

PAPER = TradingEnvironment.PAPER

# The message Alpaca returns when a stop still holds the shares an exit needs.
REFUSAL = "insufficient qty available for order (requested: 2, available: 0)"


class _RefusingBroker(FakeTradingClient):
    """Refuses the listed submission attempts with Alpaca's real refusal.

    ``FakeTradingClient.submit_error`` is sticky -- once set it refuses
    everything -- so it cannot express "the group failed twice and the third
    attempt landed". Refusing by call number can.
    """

    def __init__(self, refuse_nth: frozenset[int]) -> None:
        super().__init__(PAPER)
        self.refuse_nth = refuse_nth
        self.calls = 0
        self.refusals: list[int] = []

    def submit_order(self, request):
        self.calls += 1
        if self.calls in self.refuse_nth:
            self.refusals.append(self.calls)
            raise RuntimeError(REFUSAL)
        return super().submit_order(request)


def _stop_order(id_: str, stop: float = 97.0):
    return fake_order_raw(id_, symbol="AAPL", order_type="stop", side="sell",
                          stop_price=stop, status="new")


def _take_profit_order(id_: str):
    return fake_order_raw(id_, symbol="AAPL", order_type="limit", side="sell",
                          limit_price=110.0, status="new")


async def _reconcile_once(database: Database, refuse_nth: frozenset[int]):
    """Fold two loose exits into one group while the broker keeps refusing.

    Returns the live exit ids at the broker and the protection row, so the caller
    can compare the two directly.
    """
    client = _RefusingBroker(refuse_nth)
    engine = ExecutionEngine(client, database, active_environment=PAPER,
                             require_confirmation=False)
    manager = PositionProtectionManager(engine, database)
    client.orders["s1"] = _stop_order("s1")
    client.orders["t1"] = _take_profit_order("t1")

    await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    live = [o.id for o in await client.get_orders(status="open", symbols=["AAPL"])]
    async with database.session() as session:
        row = await ProtectionRepository(session).get(PAPER, "AAPL")
    return live, row, client.calls, len(client.refusals)


@pytest.mark.parametrize(
    "refuse_nth",
    [
        pytest.param(frozenset(), id="the-group-is-accepted"),
        pytest.param(frozenset({1}), id="the-group-is-refused-once"),
        pytest.param(frozenset({1, 2}), id="twice"),
        pytest.param(frozenset({1, 2, 3}), id="three-times"),
        pytest.param(frozenset({1, 2, 3, 4}), id="and-four-times"),
        pytest.param(frozenset({1, 2, 3, 4, 5}), id="the-rollback-too"),
    ],
)
@pytest.mark.asyncio
async def test_the_row_always_matches_the_broker(database: Database, refuse_nth):
    """However often the broker refuses, the recorded stop is the live stop.

    An empty attempt list still has to hold: the invariant is about the bot
    being honest, not about the failure path.
    """
    live, row, _calls, refusals = await _reconcile_once(database, refuse_nth)

    covered = bool(live)
    recorded = bool(row and row.stop_price is not None)
    assert covered == recorded, (
        f"el broker tiene salidas {live} pero la fila registra "
        f"stop={getattr(row, 'stop_price', None)} (rechazos={refusals})"
    )


@pytest.mark.parametrize(
    "refuse_nth",
    [
        pytest.param(frozenset(), id="accepted"),
        pytest.param(frozenset({1}), id="refused-once"),
        pytest.param(frozenset({1, 2, 3, 4, 5}), id="refused-all-the-way"),
    ],
)
@pytest.mark.asyncio
async def test_a_position_never_ends_up_unprotected(database: Database, refuse_nth):
    """A position left in the broker always keeps an exit, whatever the refusals.

    Releasing the live stop to attempt the group is the riskiest thing the
    protection manager does. This is the case that has to hold.
    """
    live, row, _calls, _refusals = await _reconcile_once(database, refuse_nth)

    assert live, "el bot soltó el stop y no dejó ninguna salida en pie"
    assert row is not None, "una posición viva quedó sin fila de protección"
    assert row.stop_order_id == live[0], (
        f"la fila apunta a {row.stop_order_id} pero la salida viva es {live[0]}"
    )


@pytest.mark.asyncio
async def test_a_refusal_is_always_reported_and_never_silent(database: Database):
    """Folding the exits fails loudly; that silence is what was worth removing.

    The whole reason the group is even attempted is that two independent exits
    cannot share the shares. If that attempt fails and nothing is said, the
    position looks fine in the notes while sitting in a known-bad shape.
    """
    client = _RefusingBroker(frozenset(range(1, 8)))
    engine = ExecutionEngine(client, database, active_environment=PAPER,
                             require_confirmation=False)
    manager = PositionProtectionManager(engine, database)
    client.orders["s1"] = _stop_order("s1")
    client.orders["t1"] = _take_profit_order("t1")

    notes = await manager.reconcile(
        environment=PAPER,
        positions=[make_position(symbol="AAPL", entry=100.0, current=102.0)],
        protect_missing=True,
    )

    assert client.refusals, "el broker nunca rechazó, el caso no probó nada"
    assert any("⚠️" in note for note in notes), (
        f"un fallo del grupo no se dijo en voz alta: {notes}"
    )
