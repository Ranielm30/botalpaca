"""Check the refined levels still produce orders Alpaca will accept.

The R:R rework moved the stop and the target onto real structure. That is only
safe if the result is still a valid bracket: the stop on the losing side, the
target reachable, both inside the distance band the risk engine enforces, both
rounded to the broker's 2-decimal rule, and a positive risk.

This is the guard on "only improve the strategy, never damage the order".
"""

from __future__ import annotations

import pytest

from botalpaca.config import RiskSettings, StrategySettings
from botalpaca.domain import SignalDirection
from botalpaca.domain.models import StructureLevel, StructureState
from botalpaca.strategies.base import (
    MAX_STRUCTURE_STOP_BUDGET,
    TradeLevels,
    levels_from_atr,
    refine_levels_with_structure,
)

RISK = RiskSettings()


def _structure(*, supports=(), resistances=()):
    return StructureState(
        supports=[StructureLevel(price=p, kind="support", touches=t) for p, t in supports],
        resistances=[StructureLevel(price=p, kind="resistance", touches=t) for p, t in resistances],
        breakout=False,
        breakout_direction=None,
    )


def _atr_levels(direction=SignalDirection.LONG, entry=100.0, atr=2.0):
    return levels_from_atr(
        direction=direction,
        entry=entry,
        atr=atr,
        stop_multiplier=2.0,
        target_multiplier=3.0,
    )


CASES = [
    # (label, direction, entry, atr, structure)
    ("long into a clean support and resistance", SignalDirection.LONG, 100.0, 2.0,
     _structure(supports=[(96.0, 3), (91.0, 2)], resistances=[(104.0, 2), (110.0, 1)])),
    ("long with only a very tight support", SignalDirection.LONG, 100.0, 2.0,
     _structure(supports=[(99.7, 1)], resistances=[(108.0, 2)])),
    ("long with no structure at all", SignalDirection.LONG, 100.0, 2.0, _structure()),
    ("short into a resistance above", SignalDirection.SHORT, 100.0, 2.0,
     _structure(resistances=[(105.0, 3)], supports=[(92.0, 2)])),
    ("short with the mirror of a tight level", SignalDirection.SHORT, 100.0, 2.0,
     _structure(resistances=[(100.4, 1)], supports=[(90.0, 2)])),
    ("a cheap stock", SignalDirection.LONG, 4.7, 0.15,
     _structure(supports=[(4.4, 2)], resistances=[(5.2, 2)])),
    ("an expensive one", SignalDirection.LONG, 4800.0, 40.0,
     _structure(supports=[(4650.0, 2)], resistances=[(4980.0, 2)])),
]


@pytest.mark.parametrize("label,direction,entry,atr,structure", CASES, ids=[c[0] for c in CASES])
def test_the_refined_levels_are_still_an_acceptable_order(label, direction, entry, atr, structure):
    base = _atr_levels(direction, entry, atr)
    levels = refine_levels_with_structure(base, structure, atr, min_rr=RISK.min_rr)

    risk = levels.risk_per_share
    reward = levels.reward_per_share

    assert risk > 0, f"{label}: a zero-width stop would be rejected outright"
    assert reward > 0, f"{label}: no room to run"

    # The stop must sit on the losing side of the entry, never through it.
    if direction is SignalDirection.LONG:
        assert levels.stop < levels.entry, f"{label}: long stop at or above entry"
        assert levels.primary_target > levels.entry, f"{label}: long target below entry"
    else:
        assert levels.stop > levels.entry, f"{label}: short stop at or below entry"
        assert levels.primary_target < levels.entry, f"{label}: short target above entry"

    # Inside the band the risk engine enforces, so the setup is not auto-vetoed.
    distance_pct = risk / entry * 100.0
    assert RISK.min_stop_distance_pct <= distance_pct <= RISK.max_stop_distance_pct, (
        f"{label}: stop distance {distance_pct:.2f}% outside "
        f"[{RISK.min_stop_distance_pct}, {RISK.max_stop_distance_pct}]"
    )

    # The broker only accepts 2 decimals at or above $1.00.
    for level in (levels.stop, levels.primary_target, levels.entry):
        assert round(level, 2) == level, f"{label}: {level} would trip the sub-penny rule"

    assert isinstance(levels.rr, float) and levels.rr > 0


@pytest.mark.parametrize("label,direction,entry,atr,structure", CASES, ids=[c[0] for c in CASES])
def test_refinement_never_widens_the_risk_past_the_budget(label, direction, entry, atr, structure):
    base = _atr_levels(direction, entry, atr)
    levels = refine_levels_with_structure(base, structure, atr, min_rr=RISK.min_rr)
    if levels is base:
        return  # nothing usable in the structure; the strategy levels stand
    budget = base.risk_per_share * MAX_STRUCTURE_STOP_BUDGET
    assert levels.risk_per_share <= budget + 1e-9, (
        f"{label}: risk grew from {base.risk_per_share} to {levels.risk_per_share}, "
        f"past the {MAX_STRUCTURE_STOP_BUDGET}x budget"
    )


@pytest.mark.parametrize("label,direction,entry,atr,structure", CASES, ids=[c[0] for c in CASES])
def test_a_trade_plan_can_be_built_from_the_refined_levels(label, direction, entry, atr, structure):
    """The last mile: the levels must survive into an executable bracket."""
    from botalpaca.domain.enums import OrderClass, OrderType, TradingEnvironment
    from botalpaca.domain.models import TradePlan
    from botalpaca.execution.builder import OrderBuilder

    levels = refine_levels_with_structure(
        _atr_levels(direction, entry, atr), structure, atr, min_rr=RISK.min_rr
    )
    plan = TradePlan(
        symbol="TEST",
        environment=TradingEnvironment.PAPER,
        direction=direction,
        order_type=OrderType.MARKET,
        qty=10.0,
        order_class=OrderClass.BRACKET,
        stop_loss=round(levels.stop, 2),
        take_profit=round(levels.primary_target, 2),
        risk_amount=round(levels.risk_per_share * 10.0, 2),
        rr=round(levels.rr, 2),
    )
    request = OrderBuilder.build_entry(plan, client_order_id="probe")
    assert request.stop_loss is not None and request.stop_loss.stop_price > 0
    assert request.take_profit is not None and request.take_profit.limit_price > 0


def test_the_ratio_is_no_longer_a_constant():
    """The whole point: different structure must produce different ratios."""
    base = _atr_levels()
    tight = refine_levels_with_structure(
        base, _structure(resistances=[(102.0, 2)]), 2.0, min_rr=RISK.min_rr
    )
    wide = refine_levels_with_structure(
        base, _structure(resistances=[(108.0, 2)]), 2.0, min_rr=RISK.min_rr
    )
    assert base.rr == pytest.approx(1.5)
    assert tight.rr != pytest.approx(wide.rr), "the ratio still carries no information"
    assert tight.rr < base.rr and wide.rr > base.rr


def test_no_structure_leaves_the_strategy_levels_untouched():
    base = _atr_levels()
    assert refine_levels_with_structure(base, None, 2.0, min_rr=1.5) is base
    assert refine_levels_with_structure(base, _structure(), 2.0, min_rr=1.5) is base
    assert refine_levels_with_structure(base, _structure(), 0.0, min_rr=1.5) is base


def test_strategy_settings_are_unaffected_by_the_refinement():
    """No strategy knob was moved: this only relocates the levels."""
    assert StrategySettings().atr_stop_multiplier == 2.0
    assert StrategySettings().atr_target_multiplier == 3.0
