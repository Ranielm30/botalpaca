"""The exit create must keep waiting while Alpaca is still releasing shares.

Cancelling an exit order only *starts* the release: the order sits in
``pending_cancel`` and a create issued in that window comes back as
``insufficient qty available ... held_for_orders: N``, naming the order that was
just cancelled. A single retry gave up inside the window, so the trailing
handover never armed and a trade that reached its trigger kept a stop the
operator believed had been replaced. Seen live on the paper account.
"""

from __future__ import annotations

import pytest

from tests.test_protection import PAPER, _setup


class _FlakyEngine:
    """Fails with the broker's own wording until the shares come back."""

    def __init__(self, engine, free_after: int):
        self._engine = engine
        self._free_after = free_after
        self.attempts = 0
        self.waits = 0

    async def submit_protective(self, *, environment, request, symbol):
        self.attempts += 1
        if self.attempts <= self._free_after:
            raise RuntimeError(
                "Alpaca rejected the request: insufficient qty available "
                "(requested: 2, available: 0, held_for_orders: 2)"
            )
        return await self._engine.submit_protective(
            environment=environment, request=request, symbol=symbol
        )


@pytest.mark.parametrize("free_after", [1, 2, 3])
async def test_the_create_is_retried_until_the_shares_come_back(database, free_after):
    """A handover that needs three tries must still succeed."""
    manager, engine, _ = await _setup(database)
    flaky = _FlakyEngine(engine, free_after=free_after)
    manager._engine = flaky
    manager._await_shares_free = _no_wait(manager, flaky)

    from botalpaca.domain.enums import SignalDirection
    from botalpaca.execution.builder import OrderBuilder

    # Built through the real builder so the shape is the production one.
    request = OrderBuilder.build_protective_stop(
        symbol="AAPL", qty=2.0, exit_side=SignalDirection.LONG, stop_price=97.0
    )

    result = await manager._submit_exit(
        environment=PAPER, symbol="AAPL", request=request
    )
    assert result.order is not None
    assert flaky.attempts == free_after + 1


async def test_it_gives_up_after_a_bounded_number_of_tries(database):
    """Never loop forever: the broker must be told the position is bare."""
    manager, engine, _ = await _setup(database)
    flaky = _FlakyEngine(engine, free_after=99)
    manager._engine = flaky
    manager._await_shares_free = _no_wait(manager, flaky)

    from botalpaca.domain.enums import SignalDirection
    from botalpaca.execution.builder import OrderBuilder

    request = OrderBuilder.build_protective_stop(
        symbol="AAPL", qty=2.0, exit_side=SignalDirection.LONG, stop_price=97.0
    )
    with pytest.raises(RuntimeError, match="insufficient qty"):
        await manager._submit_exit(
            environment=PAPER, symbol="AAPL", request=request, attempts=3
        )
    assert flaky.attempts == 3


async def test_an_unrelated_error_is_not_retried(database):
    """Only the share-release race deserves retries."""

    class _Boom:
        def __init__(self, engine):
            self._engine = engine
            self.attempts = 0

        async def submit_protective(self, **kwargs):
            self.attempts += 1
            raise RuntimeError("order class is not supported")

    manager, engine, _ = await _setup(database)
    boom = _Boom(engine)
    manager._engine = boom

    from botalpaca.domain.enums import SignalDirection
    from botalpaca.execution.builder import OrderBuilder

    request = OrderBuilder.build_protective_stop(
        symbol="AAPL", qty=2.0, exit_side=SignalDirection.LONG, stop_price=97.0
    )
    with pytest.raises(RuntimeError, match="not supported"):
        await manager._submit_exit(
            environment=PAPER, symbol="AAPL", request=request
        )
    assert boom.attempts == 1


def _no_wait(manager, flaky):
    """Replace the real poll so the test does not sleep for a minute."""

    async def wait(environment, symbol, *, ignore=()):
        flaky.waits += 1
        return True

    return wait
