"""Short selling is a PAPER feature until the operator says otherwise.

I had it backwards. I told the user the locate API does not exist in paper, so
shorts were impossible there. The documentation says the opposite: Alpaca's own
feature matrix lists short selling as available in **both** paper and live. What
actually differs is live's cost and paperwork:

    | Feature         | Paper  | Live |
    | Short Selling   | yes    | yes  |
    | Borrow Fees     | no     | yes  |

and for hard-to-borrow names live additionally requires an approved locate
through ``/v1/locates`` (round lots of 100, single-use, non-refundable), which
paper does not.

The consequence for an unattended bot is the same either way: a short validated
in paper can be refused at submission in live, or accepted and then margin
called, because live applies borrow fees and maintenance the paper run never
saw. So the environment decides, not a constant.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from botalpaca.config import RiskSettings
from botalpaca.confluence.engine import ConfluenceEngine
from botalpaca.domain import TradingEnvironment
from tests.conftest import make_opportunity

PAPER = TradingEnvironment.PAPER
REAL = TradingEnvironment.REAL


def _engine(**kw) -> ConfluenceEngine:
    return ConfluenceEngine(risk_settings=RiskSettings(**kw))


# -- the gate ------------------------------------------------------------------------
def test_shorts_are_allowed_in_paper():
    """Paper establishes locates automatically and charges no borrow fees."""
    assert _engine()._shorts_allowed(PAPER) is True


def test_shorts_are_refused_in_real_by_default():
    """Live charges borrow fees and needs an approved locate on HTB names."""
    assert _engine()._shorts_allowed(REAL) is False


def test_an_unknown_environment_never_gets_shorts():
    """Guessing 'probably fine' is how a paper setup becomes a margin call."""
    assert _engine()._shorts_allowed(None) is False


def test_the_operator_can_opt_in_to_live_shorts():
    """Deliberate is allowed; accidental is not."""
    engine = _engine(allow_shorts_in_real=True)
    assert engine._shorts_allowed(REAL) is True


def test_shorts_can_be_disabled_in_paper_too():
    engine = _engine(allow_shorts=False)
    assert engine._shorts_allowed(PAPER) is False
    assert engine._shorts_allowed(REAL) is False


# -- the documented Alpaca rules this rests on -------------------------------------
def test_the_defaults_match_the_alpaca_feature_matrix():
    """Paper trades real shorting; only borrow fees are marked unavailable."""
    settings = RiskSettings()
    assert settings.allow_shorts is True
    assert settings.allow_shorts_in_real is False


# -- the tradability gate ------------------------------------------------------------
class _Snapshot:
    atr = 2.0
    data_quality = 1.0
    # The overbought-long guard reads the RSI off the snapshot, so a double
    # without indicators would fail before the short gate is ever consulted.
    indicators = SimpleNamespace(rsi_14=55.0)


def _short_opportunity(**kw):
    """A short built by the shared factory so the model stays valid."""
    from botalpaca.domain import SignalDirection

    defaults = dict(
        symbol="ORCL",
        environment=PAPER,
        direction=SignalDirection.SHORT,
        entry=100.0,
        stop=110.0,
        target=92.0,
        # The gate now demands 2:1. These cases are about the short gate, so
        # the fixture has to sit above the ratio and let it pass.
        rr=2.5,
        score=80.0,
    )
    defaults.update(kw)
    return make_opportunity(**defaults)


@pytest.mark.parametrize("environment", [PAPER, REAL])
def test_a_short_is_judged_against_its_own_environment(environment):
    engine = _engine()
    opp = _short_opportunity(environment=environment)
    engine._apply_tradability(opp, _Snapshot(), environment=environment)

    if environment is PAPER:
        assert opp.tradable is True, opp.non_tradable_reason
    else:
        assert opp.tradable is False
        assert "cortos deshabilitados" in (opp.non_tradable_reason or "")


def test_a_long_is_unaffected_by_the_short_gate():
    from botalpaca.domain import SignalDirection

    engine = _engine()
    opp = _short_opportunity(direction=SignalDirection.LONG, stop=95.0, target=110.0)
    engine._apply_tradability(opp, _Snapshot(), environment=REAL)
    assert opp.tradable is True, opp.non_tradable_reason
