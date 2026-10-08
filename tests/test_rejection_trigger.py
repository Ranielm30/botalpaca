"""A bounce off a level needs a rejection, not just proximity.

The support/resistance strategy used to fire whenever price sat near a
well-tested level, and then reported "price rejection" as one of its
confluences. There was no rejection anywhere in the code -- being next to a
level and being pushed back off one are different events, and only the
second one is a trade.
"""

from __future__ import annotations

import pytest

from botalpaca.domain.enums import MarketRegime, SignalDirection
from botalpaca.indicators.core import rejection_wicks
from botalpaca.strategies.reversion import SupportResistanceBounceStrategy

from .conftest import make_snapshot

SUPPORT = 96.0
RESISTANCE = 104.0
ATR = 2.5


def _wick(open_: float, high: float, low: float, close: float) -> tuple[float, float]:
    return rejection_wicks([open_], [high], [low], [close])


# -- the measurement -------------------------------------------------------
def test_a_hammer_is_a_bullish_rejection():
    # Range 101-95 = 6, tail below 100 down to 95 = 5, so five sixths of the
    # bar was pushed down and bought back.
    bull, bear = _wick(100.0, 101.0, 95.0, 100.8)
    assert bull == pytest.approx(5 / 6)
    assert bear == 0.0


def test_a_shooting_star_is_a_bearish_rejection():
    bull, bear = _wick(100.0, 105.0, 99.0, 99.2)
    assert bear == pytest.approx(5 / 6)
    assert bull == 0.0


def test_a_bar_that_closes_on_its_low_rejected_nothing():
    # A long lower tail is not a rejection if the bar ended at the low:
    # nobody bought it back.
    bull, _ = _wick(101.0, 101.5, 99.0, 99.0)
    assert bull == 0.0


def test_a_bar_without_a_range_reports_nothing():
    assert _wick(100.0, 100.0, 100.0, 100.0) == (0.0, 0.0)


def test_only_the_last_bar_is_measured():
    earlier = _wick(100.0, 101.0, 95.0, 100.8)[0]
    bull, _ = rejection_wicks([1.0, 2.0, 100.0], [1.0, 3.0, 101.0], [1.0, 0.5, 95.0], [1.0, 2.0, 100.8])
    assert bull == earlier


# -- the trigger on the strategy ------------------------------------------
def _ranging(**ind_over):
    """A snapshot sitting on a level in a market with no trend to ride."""
    snap = make_snapshot(close=SUPPORT + 1.0, atr_14=ATR, atr_pct=ATR / 97.0 * 100.0, **ind_over)
    # The premise of a level bounce is that a trend is not already running.
    snap.volatility.regime = MarketRegime.RANGING
    return snap


def test_proximity_alone_is_not_a_bounce():
    """The exact case that used to fire: parked on a support, nothing else."""
    snap = _ranging(bull_rejection_atr=0.0, bear_rejection_atr=0.0)
    assert SupportResistanceBounceStrategy().detect(snap) is None


def test_a_rejection_at_support_produces_the_long():
    snap = _ranging(bull_rejection_atr=1.4, bear_rejection_atr=0.0)
    signal = SupportResistanceBounceStrategy().detect(snap)
    assert signal is not None
    assert signal.direction is SignalDirection.LONG


def test_a_rejection_at_resistance_produces_the_short():
    snap = make_snapshot(
        close=RESISTANCE - 1.0, atr_14=ATR, atr_pct=ATR / 103.0 * 100.0,
        bull_rejection_atr=0.0, bear_rejection_atr=1.4,
    )
    snap.volatility.regime = MarketRegime.RANGING
    signal = SupportResistanceBounceStrategy().detect(snap)
    assert signal is not None
    assert signal.direction is SignalDirection.SHORT


def test_a_bullish_rejection_does_not_trigger_the_short_side():
    """A wick on the wrong side of the level is not confirmation."""
    snap = make_snapshot(
        close=RESISTANCE - 1.0, atr_14=ATR, atr_pct=ATR / 103.0 * 100.0,
        bull_rejection_atr=1.4, bear_rejection_atr=0.0,
    )
    snap.volatility.regime = MarketRegime.RANGING
    assert SupportResistanceBounceStrategy().detect(snap) is None


def test_the_signal_says_how_deep_the_wick_was():
    """If the confirmation matters, the reason has to show it."""
    snap = _ranging(bull_rejection_atr=1.4, bear_rejection_atr=0.0)
    signal = SupportResistanceBounceStrategy().detect(snap)
    assert any("1.40 ATR" in reason for reason in signal.reasons)


def test_a_wick_below_the_threshold_is_only_noise():
    snap = _ranging(bull_rejection_atr=0.2, bear_rejection_atr=0.0)
    assert SupportResistanceBounceStrategy().detect(snap) is None


def test_the_trigger_threshold_is_a_knob():
    """Same tape, two settings, two answers -- and the default is not sacred."""
    quiet = _ranging(bull_rejection_atr=0.3, bear_rejection_atr=0.0)
    assert SupportResistanceBounceStrategy().detect(quiet) is None

    tolerant = SupportResistanceBounceStrategy()
    tolerant.min_rejection_atr = 0.2
    assert tolerant.detect(quiet) is not None


def test_a_strong_trend_still_vetoes_a_confirmed_bounce():
    """Adding a trigger must not reopen what the regime check used to close."""
    snap = _ranging(bull_rejection_atr=1.4, bear_rejection_atr=0.0)
    snap.volatility.regime = MarketRegime.TRENDING_UP
    assert SupportResistanceBounceStrategy().detect(snap) is None
