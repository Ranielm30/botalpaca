"""Structure is a veto, never something a high score can pay for.

A long whose target sits above a well-tested resistance is not a 2:1 setup.
It is a bet that the resistance gives way; if it does not, the R that was
approved never existed. Refusing that here is what stops the score from
manufacturing conviction out of indicators.
"""

from __future__ import annotations

import pytest

from botalpaca.config import RiskSettings
from botalpaca.confluence.engine import ConfluenceEngine
from botalpaca.domain import Quality, SignalDirection, TradingEnvironment
from botalpaca.domain.models import StructureLevel, StructureState
from tests.conftest import make_opportunity

PAPER = TradingEnvironment.PAPER


def _engine(**kw):
    return ConfluenceEngine(risk_settings=RiskSettings(**kw))


class _Indicators:
    def __init__(self, atr):
        self.atr_14 = atr
        self.rsi_14 = 55.0
        self.close = 100.0


class _Snapshot:
    """Just enough for the gate: indicators, data quality and a chart."""

    def __init__(self, *, resistances=(), supports=(), atr=5.0):
        self.indicators = _Indicators(atr)
        self.data_quality = 0.9
        self.structure = StructureState(
            supports=[StructureLevel(price=p, kind="support", touches=t) for p, t in supports],
            resistances=[
                StructureLevel(price=p, kind="resistance", touches=t) for p, t in resistances
            ],
        )


def _long(*, entry=100.0, stop=95.0, target=110.0, score=92.0):
    return make_opportunity(
        symbol="AAPL",
        environment=PAPER,
        direction=SignalDirection.LONG,
        entry=entry,
        stop=stop,
        target=target,
        rr=(target - entry) / (entry - stop),
        score=score,
    )


def _short(*, entry=100.0, stop=105.0, target=90.0, score=92.0):
    return make_opportunity(
        symbol="AAPL",
        environment=PAPER,
        direction=SignalDirection.SHORT,
        entry=entry,
        stop=stop,
        target=target,
        rr=(entry - target) / (stop - entry),
        score=score,
    )


# -- the veto itself ---------------------------------------------------------


def test_a_long_whose_target_sits_above_a_resistance_is_refused():
    opp = _long()
    engine = _engine()
    engine._apply_tradability(opp, _Snapshot(resistances=[(105.0, 3)]), environment=PAPER)
    assert opp.tradable is False
    assert "resistencia" in opp.non_tradable_reason


def test_a_high_score_does_not_buy_its_way_past_a_resistance():
    """The whole point: 92/100 is not enough."""
    opp = _long(score=99.0)
    engine = _engine()
    engine._apply_tradability(opp, _Snapshot(resistances=[(105.0, 4)]), environment=PAPER)
    assert opp.tradable is False
    assert opp.quality is Quality.NO_OPERABLE


def test_a_short_whose_target_sits_below_a_support_is_refused():
    opp = _short()
    engine = _engine()
    engine._apply_tradability(opp, _Snapshot(supports=[(95.0, 3)]), environment=PAPER)
    assert opp.tradable is False
    assert "soporte" in opp.non_tradable_reason


# -- the veto must not fire when it should not --------------------------------


def test_a_long_whose_target_is_below_the_resistance_passes():
    """Target at 110, resistance at 118: nothing has to be broken."""
    opp = _long(entry=100.0, stop=95.0, target=110.0)
    engine = _engine()
    engine._apply_tradability(opp, _Snapshot(resistances=[(118.0, 3)]), environment=PAPER)
    assert opp.tradable is True
    assert opp.non_tradable_reason is None


def test_a_lone_resistance_is_noise_not_an_obstacle():
    """One touch is not a wall. Refusing those would empty the scanner."""
    opp = _long()
    engine = _engine()
    engine._apply_tradability(opp, _Snapshot(resistances=[(105.0, 1)]), environment=PAPER)
    assert opp.tradable is True


def test_a_chart_with_no_levels_at_all_is_not_blocked():
    opp = _long()
    engine = _engine()
    engine._apply_tradability(opp, _Snapshot(), environment=PAPER)
    assert opp.tradable is True


def test_a_resistance_behind_the_entry_does_not_block():
    """Only what stands between the entry and the target matters."""
    opp = _long(entry=100.0, stop=95.0, target=110.0)
    engine = _engine()
    engine._apply_tradability(opp, _Snapshot(resistances=[(97.0, 3)]), environment=PAPER)
    assert opp.tradable is True


def test_a_support_above_a_long_does_not_block_it():
    """Supports are where longs come from, not what stops them."""
    opp = _long()
    engine = _engine()
    engine._apply_tradability(
        opp, _Snapshot(resistances=[], supports=[(98.0, 3), (95.0, 3)]), environment=PAPER
    )
    assert opp.tradable is True


# -- configurability ---------------------------------------------------------


def test_the_touch_requirement_is_configurable():
    """A stricter reading of "wall" lets one-touch levels block."""
    opp = _long()
    engine = _engine(min_blocking_level_touches=1)
    engine._apply_tradability(opp, _Snapshot(resistances=[(105.0, 1)]), environment=PAPER)
    assert opp.tradable is False


def test_the_veto_can_be_switched_off():
    opp = _long()
    engine = _engine(min_blocking_level_touches=0)
    engine._apply_tradability(opp, _Snapshot(resistances=[(105.0, 5)]), environment=PAPER)
    assert opp.tradable is True


# -- the message has to be readable -------------------------------------------


def test_the_reason_names_the_level_and_both_sides():
    opp = _long(entry=100.0, stop=95.0, target=110.0)
    engine = _engine()
    engine._apply_tradability(opp, _Snapshot(resistances=[(105.0, 3)]), environment=PAPER)
    reason = opp.non_tradable_reason
    assert "105.00" in reason
    assert "3 toques" in reason
    assert "110.00" in reason


def test_a_degenerate_target_never_trips_the_veto():
    """Guard against the comparison itself, not the strategy."""
    opp = _long(entry=100.0, stop=95.0, target=100.0)
    engine = _engine()
    engine._apply_tradability(
        opp, _Snapshot(resistances=[(105.0, 3)], atr=5.0), environment=PAPER
    )
    # Whatever the outcome, it must not be the obstacle talking.
    assert "exige romperla" not in (opp.non_tradable_reason or "")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
