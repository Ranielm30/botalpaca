"""A bracket must not carry a price Alpaca will refuse, nor a level the market
has already run past.

Two live rejections, both found by the operator tapping ACEPTAR:

    invalid take_profit.limit_price 1194.4686506181379.
    sub-penny increment does not fulfill minimum pricing criteria

    take_profit.limit_price must be >= base_price + 0.01

The first is ours: ``refine_levels_with_structure`` copied a resistance price
straight off the bar data, decimals and all. The second is staleness -- Alpaca
validates a take-profit against ``base_price``, the price the entry fills at, not
the price the analysis was built from. Move the market and the target ends up on
the wrong side; the broker then answers with a message that tells the operator
nothing about what to do.
"""

from __future__ import annotations

import pytest

from botalpaca.app import Application
from botalpaca.domain.enums import OrderClass, OrderType, SignalDirection, TradingEnvironment
from botalpaca.strategies.base import TradeLevels, refine_levels_with_structure

PAPER = TradingEnvironment.PAPER


def _structure(supports=(), resistances=()):
    from botalpaca.domain.models import StructureLevel, StructureState

    return StructureState(
        supports=[StructureLevel(price=p, kind="support", touches=t) for p, t in supports],
        resistances=[
            StructureLevel(price=p, kind="resistance", touches=t) for p, t in resistances
        ],
    )


# -- the sub-penny increment ----------------------------------------------------------
def test_a_raw_structural_price_never_reaches_the_broker():
    """A resistance carries every decimal the bar data had."""
    levels = TradeLevels(
        direction=SignalDirection.LONG, entry=105.67, stop=99.65, targets=(114.69,)
    )
    refined = refine_levels_with_structure(
        levels,
        _structure(
            supports=[(99.65432109876543, 3)],
            resistances=[(1194.4686506181379, 2)],
        ),
        atr=6.0,
        min_rr=1.5,
    )
    assert refined.stop == 99.65
    assert refined.primary_target == 1194.47, refined.primary_target
    # Two decimals is what Alpaca accepts at or above $1.00.
    for price in (refined.stop, refined.primary_target):
        assert round(price, 2) == price


def test_sub_penny_prices_keep_four_decimals():
    levels = TradeLevels(
        direction=SignalDirection.LONG, entry=0.9234, stop=0.85, targets=(1.10,)
    )
    refined = refine_levels_with_structure(
        levels, _structure(supports=[(0.8543210987, 2)]), atr=0.07, min_rr=1.5
    )
    assert round(refined.stop, 4) == refined.stop


# -- the market already ran past the target ------------------------------------------
class _App:
    """Enough of Application to exercise the guard unbound."""

    def __init__(self, price: float) -> None:
        self.price = price
        self.active = type(
            "_Active",
            (),
            {
                "market": type(
                    "_Market", (), {"get_last_price": self._price}
                )(),
            },
        )()

    async def _price(self, symbol: str) -> float:
        return self.price


def _plan(*, long: bool = True, entry=100.0, stop=95.0, target=110.0):
    from botalpaca.domain.models import TradePlan

    return TradePlan(
        symbol="AAPL",
        environment=PAPER,
        direction=SignalDirection.LONG if long else SignalDirection.SHORT,
        order_type=OrderType.MARKET,
        qty=1.0,
        order_class=OrderClass.BRACKET,
        stop_loss=stop,
        take_profit=target,
    )


async def test_a_stale_target_is_refused_before_the_broker_sees_it():
    from botalpaca.domain.errors import ValidationError

    app = _App(price=112.0)  # the market ran past the 110 target
    with pytest.raises(ValidationError, match="objetivo ya quedo"):
        await Application._check_levels_are_still_valid(app, _plan())


async def test_a_stale_stop_is_refused():
    from botalpaca.domain.errors import ValidationError

    app = _App(price=90.0)  # the market dropped below the 95 stop
    with pytest.raises(ValidationError, match="stop ya quedo"):
        await Application._check_levels_are_still_valid(app, _plan())


async def test_a_short_is_mirrored():
    from botalpaca.domain.errors import ValidationError

    app = _App(price=88.0)
    plan = _plan(long=False, entry=100.0, stop=105.0, target=90.0)
    with pytest.raises(ValidationError, match="objetivo ya quedo"):
        await Application._check_levels_are_still_valid(app, plan)


async def test_a_fresh_setup_passes_and_gets_rounded():
    app = _App(price=100.5)
    plan = _plan(stop=99.65432109876543, target=114.693456789)
    await Application._check_levels_are_still_valid(app, plan)
    assert plan.stop_loss == 99.65
    assert plan.take_profit == 114.69


async def test_no_price_means_no_blind_refusal():
    """A missing quote must not block a valid setup."""
    app = _App(price=0.0)
    plan = _plan()
    await Application._check_levels_are_still_valid(app, plan)
    assert plan.take_profit == 110.0


async def test_a_plan_without_a_take_profit_is_left_alone():
    app = _App(price=500.0)
    plan = _plan()
    plan.take_profit = None
    await Application._check_levels_are_still_valid(app, plan)
    assert plan.take_profit is None
