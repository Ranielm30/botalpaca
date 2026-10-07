"""Entry geometry has to survive the gap between analysis and fill.

LLY was approved at 1.5 R:R on $1105.06 and filled at $1176.00: a real 0.08,
with the target 0.86% away and the stop 10.6% out. Every level was still on the
correct side of the price, so the sign check waved it through. These tests pin
the four guards that catch it instead.
"""

from __future__ import annotations

import pytest

from botalpaca.domain.enums import OrderClass, OrderType, SignalDirection, TradingEnvironment
from botalpaca.domain.errors import ValidationError
from botalpaca.domain.models import TradePlan

PAPER = TradingEnvironment.PAPER

# The exact geometry the bot approved and then filled 6.4% away.
LLY_ANALYSED_ENTRY = 1105.06
LLY_STOP = 1051.02
LLY_TARGET = 1186.12
LLY_FILL = 1176.00


class _Market:
    def __init__(self, price: float) -> None:
        self.price = price

    async def get_last_price(self, symbol: str) -> float:
        return self.price


class _App:
    """Just enough for the unbound accept-time guard."""

    def __init__(self, price: float, risk=None) -> None:
        self.active = type("Active", (), {})()
        self.active.market = _Market(price)
        settings = type("S", (), {})()
        settings.risk = risk if risk is not None else _risk()
        self.settings = settings


def _risk(**kw):
    from botalpaca.config import RiskSettings

    return RiskSettings(**kw)


def _plan(*, entry=100.0, stop=95.0, target=110.0, atr=2.0, long=True) -> TradePlan:
    return TradePlan(
        symbol="AAPL",
        environment=PAPER,
        direction=SignalDirection.LONG if long else SignalDirection.SHORT,
        order_type=OrderType.MARKET,
        qty=10.0,
        order_class=OrderClass.BRACKET,
        entry=entry,
        stop_loss=stop,
        take_profit=target,
        atr=atr,
        rr=2.0,
    )


async def _check(app, plan):
    from botalpaca.app import Application

    return await Application._check_levels_are_still_valid(app, plan)


# -- the LLY geometry ----------------------------------------------------------------
async def test_the_lly_fill_is_refused_for_its_degraded_ratio():
    """The trade that actually happened, reproduced exactly."""
    app = _App(LLY_FILL)
    plan = _plan(entry=LLY_ANALYSED_ENTRY, stop=LLY_STOP, target=LLY_TARGET, atr=27.0)
    with pytest.raises(ValidationError) as excinfo:
        await _check(app, plan)
    message = str(excinfo.value)
    # It drifted 6.4%, so the drift guard fires first and says so.
    assert "REJECTED_PRICE_DRIFT" in message
    assert "1,105.06" in message
    assert "1,176.00" in message


async def test_the_lly_fill_is_refused_on_the_ratio_alone_when_it_did_not_drift():
    """Same geometry, drift cap lifted: only the ratio guard is left standing."""
    # The target-proximity guard is switched off so this test exercises the
    # ratio on its own; a separate test covers proximity.
    app = _App(
        LLY_FILL,
        _risk(
            max_entry_drift_pct=99.0,
            min_target_distance_atr=0.0,
            min_target_distance_pct=0.0,
        ),
    )
    plan = _plan(entry=LLY_FILL, stop=LLY_STOP, target=LLY_TARGET, atr=27.0)
    with pytest.raises(ValidationError) as excinfo:
        await _check(app, plan)
    assert "REJECTED_DEGRADED_RR" in str(excinfo.value)


# -- guard 1: price drift -------------------------------------------------------------
async def test_a_moved_price_is_refused():
    app = _App(100.6)
    with pytest.raises(ValidationError, match="REJECTED_PRICE_DRIFT"):
        await _check(app, _plan(entry=100.0))


async def test_the_drift_cap_is_configurable():
    app = _App(100.6, _risk(max_entry_drift_pct=1.0))
    # 0.6% is inside a 1% cap.
    assert await _check(app, _plan(entry=100.0)) is None


# -- guard 2: the sign check ---------------------------------------------------------
async def test_a_target_the_price_overtook_is_refused():
    # The price moved 0.9%, so the drift cap has to stand aside for the
    # side check, which is what this case is actually about.
    app = _App(116.0, _risk(max_entry_drift_pct=99.0))
    with pytest.raises(ValidationError, match="REJECTED_LEVEL_SIDE"):
        await _check(app, _plan(entry=110.0, stop=105.0, target=115.0))


async def test_a_stop_underwater_is_refused():
    app = _App(104.0, _risk(max_entry_drift_pct=99.0))
    with pytest.raises(ValidationError, match="REJECTED_LEVEL_SIDE"):
        await _check(app, _plan(entry=105.0, stop=105.5, target=115.0))


# -- guard 3: the target must be worth reaching -------------------------------------
async def test_a_target_closer_than_one_atr_is_refused():
    app = _App(100.0)
    # Target 1.0 away with ATR 2.0 means half an ATR of room.
    with pytest.raises(ValidationError, match="REJECTED_TARGET_TOO_CLOSE"):
        await _check(app, _plan(entry=100.0, stop=95.0, target=101.0, atr=2.0))


async def test_a_target_closer_than_one_percent_is_refused():
    # 1.4% of room clears the 1 ATR floor but not the 1% one. The ratio
    # guard is switched off because at this distance the R:R is also poor,
    # and this case is about proximity alone.
    app = _App(100.0, _risk(min_rr_at_entry=0.0))
    with pytest.raises(ValidationError, match="REJECTED_TARGET_TOO_CLOSE"):
        await _check(app, _plan(entry=100.0, stop=95.0, target=100.7, atr=0.5))


async def test_a_target_with_room_is_accepted():
    app = _App(100.0)
    plan = _plan(entry=100.0, stop=95.0, target=110.0, atr=2.0)
    assert await _check(app, plan) is None


# -- guard 4: overbought longs -------------------------------------------------------
def _engine():
    from botalpaca.confluence.engine import ConfluenceEngine

    return ConfluenceEngine()


class _Snapshot:
    def __init__(self, rsi):
        self.indicators = type("I", (), {"rsi_14": rsi})()
        self.data_quality = 0.9


def _opportunity(**kw):
    from tests.conftest import make_opportunity

    defaults = dict(
        symbol="AAPL",
        environment=PAPER,
        direction=SignalDirection.LONG,
        entry=100.0,
        stop=95.0,
        target=110.0,
        rr=2.0,
        score=70.0,
    )
    defaults.update(kw)
    return make_opportunity(**defaults)


def test_a_long_at_extreme_overbought_is_not_tradable():
    """LLY was taken twice this way, at RSI 85 and 87."""
    engine = _engine()
    opp = _opportunity()
    engine._apply_tradability(opp, _Snapshot(85.0), environment=PAPER)
    assert opp.tradable is False
    assert "sobrecompra" in opp.non_tradable_reason
    assert "85" in opp.non_tradable_reason


def test_the_cap_is_configurable():
    engine = type("E", (), {})()
    engine.risk = _risk(max_rsi_for_long=90.0)
    from botalpaca.confluence.engine import ConfluenceEngine

    engine = ConfluenceEngine(risk_settings=_risk(max_rsi_for_long=90.0))
    opp = _opportunity()
    engine._apply_tradability(opp, _Snapshot(85.0), environment=PAPER)
    assert opp.tradable is True


def test_a_normal_long_is_unaffected():
    engine = _engine()
    opp = _opportunity()
    engine._apply_tradability(opp, _Snapshot(58.0), environment=PAPER)
    assert opp.tradable is True


def test_a_short_at_the_same_rsi_is_unaffected():
    """Overbought is the argument against a long; for a short it is the setup."""
    engine = _engine()
    opp = _opportunity(direction=SignalDirection.SHORT, entry=100.0, stop=105.0, target=90.0)
    engine._apply_tradability(opp, _Snapshot(85.0), environment=PAPER)
    assert opp.tradable is True


def test_a_missing_rsi_does_not_block():
    engine = _engine()
    opp = _opportunity()
    engine._apply_tradability(opp, _Snapshot(None), environment=PAPER)
    assert opp.tradable is True
