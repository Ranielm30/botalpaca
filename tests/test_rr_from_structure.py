"""The R:R has to be a measurement, not a configuration value.

``levels_from_atr`` places the stop and the target as multiples of the same ATR,
so ``(target - entry) / (entry - stop)`` collapses to
``target_multiplier / stop_multiplier`` -- 3.0 / 2.0, exactly 1.50, for every
symbol and every day. A risk filter built on that ratio could never reject
anything, while looking exactly like a filter.

``refine_levels_with_structure`` puts the levels on the chart instead. The stop
sits just beyond the nearest support the price has to defend, with a buffer of
noise around it; the target walks outwards until it pays at least ``min_rr``
for the risk taken. The ratio then measures the setup.

Two rules the numbers behind this are worth repeating. Measured results put the
profitable range for an initial stop at 2-4 ATR, so the stop is capped there:
past it the target has to travel too far to fill. And a ratio of 2:1 needs a
33% hit rate just to break even, which is the floor worth designing against --
1.5:1 was labelled "marginal" at a 45% hit rate.
"""

from __future__ import annotations

import pytest

from botalpaca.domain.enums import SignalDirection
from botalpaca.strategies.base import (
    MAX_STOP_ATR,
    STRUCTURE_STOP_BUFFER_ATR,
    TradeLevels,
    levels_from_atr,
    refine_levels_with_structure,
)

LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT
MIN_RR = 2.0


class _Level:
    def __init__(self, price: float, touches: int = 1):
        self.price = price
        self.touches = touches


class _Structure:
    def __init__(self, supports=(), resistances=()):
        self.supports = [_Level(p, t) for p, t in supports]
        self.resistances = [_Level(p, t) for p, t in resistances]


def _atr_levels(direction=LONG, entry=100.0, atr=2.0):
    return levels_from_atr(
        direction, entry, atr, stop_multiplier=2.0, target_multiplier=3.0
    )


# -- the tautology, demonstrated ------------------------------------------------------
def test_the_atr_levels_always_report_the_same_ratio():
    """The reason this module exists: identical ratio on wildly different charts."""
    ratios = set()
    for atr in (0.5, 2.0, 25.0):
        for entry in (3.0, 100.0, 4800.0):
            levels = _atr_levels(entry=entry, atr=atr)
            ratios.add(round(levels.rr, 4))
    assert ratios == {1.5}


def test_the_atr_ratio_ignores_the_market_entirely():
    cheap = _atr_levels(atr=0.5)
    expensive = _atr_levels(atr=25.0)
    assert cheap.rr == expensive.rr
    assert cheap.risk_per_share != expensive.risk_per_share


# -- the stop: nearest support, plus room for the wick -------------------------------
def test_the_stop_sits_beyond_the_nearest_support_with_a_noise_buffer():
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(96.0, 2)]), 2.0, min_rr=MIN_RR
    )
    # ATR is 2.0, so the buffer is half of it: the stop goes below the level,
    # not on it, so the wick has to travel before the support is even tested.
    assert refined.stop == pytest.approx(96.0 - STRUCTURE_STOP_BUFFER_ATR * 2.0)
    assert refined.stop < 96.0


def test_the_nearest_support_wins_not_the_deepest():
    """Widening the risk buys no information and only makes the ratio harder.

    Both levels here are wide enough to sit behind, so the choice between them
    is the one the floor does not decide.
    """
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(96.0, 3), (95.5, 1)]), 2.0, min_rr=MIN_RR
    )
    assert refined.stop == pytest.approx(96.0 - STRUCTURE_STOP_BUFFER_ATR * 2.0)


def test_a_support_too_near_to_sit_behind_is_not_used():
    """A level resting right under the entry is noise, not a floor.

    Adopting it put the stop half an ATR away -- inside the range a single
    ordinary day crosses -- and then wrecked the position size on top of that.
    The ATR stop stands instead: it is the same 2 ATR the level would have had
    to beat to be worth taking.
    """
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(99.8, 3)]), 2.0, min_rr=MIN_RR
    )
    assert refined.stop == pytest.approx(levels.stop)


def test_the_stop_never_reaches_past_the_measured_band():
    """Past 2-4 ATR the target has to travel far enough to stop filling."""
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(50.0, 5)]), 2.0, min_rr=MIN_RR
    )
    assert levels.entry - refined.stop <= MAX_STOP_ATR * 2.0 + 1e-9


# -- the target: it has to pay for the risk -------------------------------------------
def test_the_target_walks_outwards_until_the_ratio_clears():
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(resistances=[(104.0, 2), (120.0, 2)]), 2.0, min_rr=MIN_RR
    )
    # 104 is only 4 away against 4 of risk: that is 1.0, not enough. 120 pays.
    assert refined.primary_target == pytest.approx(120.0)
    assert refined.rr >= MIN_RR


def test_the_nearest_resistance_is_used_when_nothing_on_the_chart_pays():
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(resistances=[(104.0, 2)]), 2.0, min_rr=MIN_RR
    )
    # Real structure the market has defended beats a projection nobody reaches.
    assert refined.primary_target == pytest.approx(104.0)
    assert refined.rr < MIN_RR


def test_a_bare_chart_projects_the_target_the_risk_demands():
    levels = _atr_levels()
    refined = refine_levels_with_structure(levels, _Structure(), 2.0, min_rr=MIN_RR)
    assert refined.rr >= MIN_RR


def test_the_rr_gate_can_still_reject_a_setup():
    """If the next obstacle does not pay, the setup genuinely does not."""
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(96.0, 2)], resistances=[(100.5, 2)]),
        2.0, min_rr=MIN_RR,
    )
    assert refined.rr < MIN_RR


def test_the_ratio_varies_between_setups():
    tight = refine_levels_with_structure(
        _atr_levels(), _Structure(supports=[(97.0, 2)], resistances=[(104.0, 2)]),
        2.0, min_rr=MIN_RR,
    )
    generous = refine_levels_with_structure(
        _atr_levels(), _Structure(supports=[(96.0, 2)], resistances=[(120.0, 2)]),
        2.0, min_rr=MIN_RR,
    )
    assert tight.rr != generous.rr


# -- shorts are mirrored ---------------------------------------------------------------
def test_a_short_uses_resistance_for_the_stop_and_support_for_the_target():
    levels = _atr_levels(direction=SHORT, entry=100.0)
    refined = refine_levels_with_structure(
        levels,
        _Structure(supports=[(90.0, 2)], resistances=[(105.0, 2)]),
        2.0,
        min_rr=MIN_RR,
    )
    # Above the entry, because a short is defended from overhead, plus the buffer.
    assert refined.stop == pytest.approx(105.0 + STRUCTURE_STOP_BUFFER_ATR * 2.0)
    # Below the entry, because that is where the profit is.
    assert refined.primary_target == pytest.approx(90.0)
    # 10 of reward against 6 of risk: below 2:1, and the function says so
    # instead of reaching for a target the chart never offered.
    assert refined.rr == pytest.approx(10.0 / 6.0)
    assert refined.rr < MIN_RR


# -- it never makes things worse --------------------------------------------------------
def test_without_structure_the_levels_are_untouched():
    levels = _atr_levels()
    assert refine_levels_with_structure(levels, None, 2.0, min_rr=MIN_RR) is levels


def test_a_broken_stop_never_survives_the_refinement():
    """Whatever comes out, the stop must stay on the losing side of the entry."""
    for entry in (10.0, 100.0, 1000.0):
        for direction in (LONG, SHORT):
            levels = levels_from_atr(
                direction, entry, 1.0, stop_multiplier=2.0, target_multiplier=3.0
            )
            refined = refine_levels_with_structure(
                levels,
                _Structure(
                    supports=[(entry * 0.5, 2), (entry * 0.99, 2)],
                    resistances=[(entry * 1.5, 2), (entry * 1.01, 2)],
                ),
                1.0,
                min_rr=MIN_RR,
            )
            if direction is LONG:
                assert refined.stop < refined.entry
                assert refined.primary_target > refined.entry
            else:
                assert refined.stop > refined.entry
                assert refined.primary_target < refined.entry
            assert refined.rr > 0


def test_the_result_is_frozen():
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(96.0, 2)], resistances=[(110.0, 2)]),
        2.0, min_rr=MIN_RR,
    )
    assert isinstance(refined, TradeLevels)
    with pytest.raises(Exception):
        refined.stop = 1.0  # type: ignore[misc]
