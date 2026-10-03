"""The R:R has to be a measurement, not a configuration value.

``levels_from_atr`` places the stop and the target as multiples of the same ATR,
so ``(target - entry) / (entry - stop)`` collapses to
``target_multiplier / stop_multiplier`` -- 3.0 / 2.0, exactly 1.50, for every
symbol and every day. A risk filter built on that ratio could never reject
anything, while looking exactly like a filter.

``refine_levels_with_structure`` puts the levels on the chart instead: the stop
just beyond a tested support, the target at the nearest resistance. The ratio
then measures the setup.
"""

from __future__ import annotations

import pytest

from botalpaca.domain.enums import SignalDirection
from botalpaca.strategies.base import (
    TradeLevels,
    levels_from_atr,
    refine_levels_with_structure,
)

LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT
MIN_RR = 1.5


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
    # The risk is wildly different, yet the ratio is identical.
    assert cheap.risk_per_share != expensive.risk_per_share


# -- the refinement reads the chart --------------------------------------------------
def test_the_stop_goes_just_beyond_a_tested_support():
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(96.0, 2)]), 2.0, min_rr=MIN_RR
    )
    assert refined.stop == pytest.approx(96.0)
    # Beyond the level, so it only fires when the support actually fails.
    assert refined.stop < 96.0 + 1e-9


def test_the_target_goes_to_the_nearest_resistance():
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(resistances=[(104.0, 2), (120.0, 2)]), 2.0, min_rr=MIN_RR
    )
    assert refined.primary_target == pytest.approx(104.0)


def test_a_well_tested_level_is_preferred_over_a_bare_one():
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels,
        _Structure(supports=[(99.0, 1), (96.0, 3)]),
        2.0,
        min_rr=MIN_RR,
    )
    # 99 was touched once; 96 has been tested three times and is the real floor.
    assert refined.stop == pytest.approx(96.0)


def test_a_support_inside_the_noise_band_is_ignored():
    """A stop 0.1 ATR away gets taken out on an ordinary day."""
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(99.8, 3)]), 2.0, min_rr=MIN_RR, min_stop_atr=0.5
    )
    assert refined.stop == pytest.approx(levels.stop)


# -- the filter can finally reject ----------------------------------------------------
def test_the_ratio_now_varies_between_setups():
    tight = refine_levels_with_structure(
        _atr_levels(), _Structure(supports=[(97.0, 2)], resistances=[(104.0, 2)]),
        2.0, min_rr=MIN_RR,
    )
    generous = refine_levels_with_structure(
        _atr_levels(), _Structure(supports=[(96.0, 2)], resistances=[(120.0, 2)]),
        2.0, min_rr=MIN_RR,
    )
    assert tight.rr != generous.rr
    assert tight.rr == pytest.approx(4.0 / 3.0)
    assert generous.rr == pytest.approx(20.0 / 4.0)


def test_the_rr_gate_is_no_longer_a_tautology():
    """A setup whose next obstacle does not pay should fail the gate."""
    levels = _atr_levels()
    refined = refine_levels_with_structure(
        levels, _Structure(supports=[(96.0, 2)], resistances=[(100.5, 2)]),
        2.0, min_rr=MIN_RR,
    )
    assert refined.rr < MIN_RR


# -- shorts are mirrored ---------------------------------------------------------------
def test_a_short_uses_resistance_for_the_stop_and_support_for_the_target():
    levels = _atr_levels(direction=SHORT, entry=100.0)
    refined = refine_levels_with_structure(
        levels,
        _Structure(supports=[(90.0, 2)], resistances=[(105.0, 2)]),
        2.0,
        min_rr=MIN_RR,
    )
    # Above the entry, because a short is defended from overhead.
    assert refined.stop == pytest.approx(105.0)
    # Below the entry, because that is where the profit is.
    assert refined.primary_target == pytest.approx(90.0)
    assert refined.rr == pytest.approx(10.0 / 5.0)


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
