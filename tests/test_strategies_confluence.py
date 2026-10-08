"""Strategy engine + confluence scoring + entry-quality rendering."""

from __future__ import annotations

import pytest

from botalpaca.analysis.snapshot import analyze_symbol
from botalpaca.config.settings import AnalyticsSettings, RiskSettings, StrategySettings
from botalpaca.confluence.engine import (
    MIN_EXECUTABLE_SCORE,
    MIN_RR,
    WEIGHTS,
    ConfluenceEngine,
    classify_quality,
)
from botalpaca.confluence.entry_quality import format_entry_quality
from botalpaca.domain.enums import Quality, SetupType, SignalDirection, TradingEnvironment
from botalpaca.domain.models import StructureLevel
from botalpaca.strategies.base import levels_from_atr, make_fingerprint
from botalpaca.strategies.registry import StrategyEngine, build_strategies

from .conftest import make_bars, make_snapshot


def _engine() -> ConfluenceEngine:
    return ConfluenceEngine(
        risk_settings=RiskSettings(),
        strategy_settings=StrategySettings(),
        analytics_settings=AnalyticsSettings(),
    )


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


def test_levels_from_atr_long():
    levels = levels_from_atr(
        SignalDirection.LONG, 100.0, 2.0, stop_multiplier=2.0, target_multiplier=3.0
    )
    assert levels.stop == pytest.approx(96.0)
    assert levels.primary_target == pytest.approx(106.0)
    # 6.0 of reward over 4.0 of risk
    assert levels.rr == pytest.approx(1.5)
    assert levels.risk_per_share == pytest.approx(4.0)


def test_levels_from_atr_short():
    levels = levels_from_atr(
        SignalDirection.SHORT, 100.0, 2.0, stop_multiplier=2.0, target_multiplier=3.0
    )
    assert levels.stop == pytest.approx(104.0)
    assert levels.primary_target == pytest.approx(94.0)
    assert levels.rr == pytest.approx(1.5)


def test_levels_from_atr_rejects_zero_atr():
    with pytest.raises(ValueError):
        levels_from_atr(SignalDirection.LONG, 100.0, 0.0, stop_multiplier=2.0, target_multiplier=3.0)


def test_fingerprint_is_price_independent():
    a = make_fingerprint("AAPL", SetupType.PULLBACK, SignalDirection.LONG, "1D")
    b = make_fingerprint("AAPL", SetupType.PULLBACK, SignalDirection.LONG, "1D")
    c = make_fingerprint("AAPL", SetupType.PULLBACK, SignalDirection.SHORT, "1D")
    assert a == b
    assert a != c
    assert a != make_fingerprint("MSFT", SetupType.PULLBACK, SignalDirection.LONG, "1D")


def test_classify_quality_hard_vetoes():
    assert classify_quality(99.0, rr=1.0, data_quality=1.0) is Quality.NO_OPERABLE
    assert classify_quality(99.0, rr=3.0, data_quality=0.1) is Quality.NO_OPERABLE
    assert classify_quality(90.0, rr=2.0, data_quality=1.0) is Quality.ALTA
    assert classify_quality(58.0, rr=2.0, data_quality=1.0) is Quality.MEDIA
    assert classify_quality(40.0, rr=2.0, data_quality=1.0) is Quality.BAJA
    assert classify_quality(10.0, rr=2.0, data_quality=1.0) is Quality.NO_OPERABLE


def test_strategies_never_emit_incoherent_levels():
    up = analyze_symbol("AAPL", make_bars(300, drift=0.004, noise=0.6, seed=41))
    up.htf_direction = SignalDirection.LONG
    up.htf_timeframe = "1W"
    engine = StrategyEngine()
    signals = engine.run(up)
    for sig in signals:
        if sig.direction is SignalDirection.LONG:
            assert sig.stop < sig.entry < sig.targets[0]
        elif sig.direction is SignalDirection.SHORT:
            assert sig.stop > sig.entry > sig.targets[0]


def test_strategy_engine_merges_agreeing_strategies():
    up = analyze_symbol("AAPL", make_bars(320, drift=0.005, noise=0.5, seed=42))
    up.htf_direction = SignalDirection.LONG
    up.htf_timeframe = "1W"
    engine = StrategyEngine()
    signals = engine.run(up)
    assert signals
    for sig in signals:
        assert sig.rr > 0
        assert sig.reasons


def test_multi_timeframe_strategy_abstains_when_htf_unknown():
    bars = make_bars(300, drift=0.005, noise=0.5, seed=43)
    snap = analyze_symbol("AAPL", bars)
    snap.htf_direction = None
    engine = StrategyEngine()
    signals = [s for s in engine.run(snap) if s.strategy is SetupType.MULTI_TIMEFRAME]
    assert not signals


def test_confluence_vetoes_low_rr():
    """A setup whose next obstacle does not pay for the risk must be vetoed.

    The signal below is deliberately unrecoverable: the nearest resistance sits
    a few cents above the entry, so no placement of the stop rescues the ratio.
    The shared snapshot's own levels would have improved this trade, which is
    why it is not used here.
    """
    snap = make_snapshot()
    snap.structure.resistances = [StructureLevel(price=100.4, kind="resistance", touches=3)]
    signal = _signal(entry=100.0, stop=99.0, target=100.5)
    opp = _engine().score(snap, signal, environment=TradingEnvironment.PAPER)
    assert opp.rr < MIN_RR
    assert opp.quality is Quality.NO_OPERABLE
    assert opp.tradable is False
    assert opp.non_tradable_reason


def test_confluence_refines_levels_onto_real_structure():
    """The stop goes to a tested level, the target to the next obstacle.

    Both used to be ATR multiples, which made the ratio a constant 1.50 no
    matter what the chart looked like.
    """
    snap = make_snapshot()
    signal = _signal(entry=100.0, stop=96.0, target=112.0)
    opp = _engine().score(snap, signal, environment=TradingEnvironment.PAPER)
    # The nearest support is 96 and the noise buffer is half an ATR, so the
    # stop sits 1.25 below it. The only resistance on the chart is 4 points
    # away, which does not pay for 5.25 of risk at 2:1, so it is used anyway.
    assert opp.stop == pytest.approx(94.75)
    assert opp.target == pytest.approx(104.0)
    assert opp.rr == pytest.approx((opp.target - 100.0) / (100.0 - opp.stop), abs=0.005)
    assert opp.rr != 1.5


def test_confluence_tradable_requires_score_gate():
    """Tradable is decided by the score and the gates, and the R:R is measured.

    This test used to assert ``opp.rr >= MIN_RR``, which held only because the
    levels were ATR multiples and the ratio was a constant 1.50. The levels now
    come off real structure, so the ratio is a measurement and the assertion
    moved to what the test is actually about: the score gate.
    """
    snap = make_snapshot()
    signal = _signal(entry=100.0, stop=97.0, target=106.0)
    opp = _engine().score(snap, signal, environment=TradingEnvironment.PAPER)
    # The snapshot's support is 96 and its resistance 104; with half an ATR of
    # buffer the risk becomes 5.25 against 4 of reward, which no longer clears.
    assert opp.stop == pytest.approx(94.75)
    assert opp.target == pytest.approx(104.0)
    assert opp.rr == pytest.approx((opp.target - 100.0) / (100.0 - opp.stop), abs=0.005)
    # Below MIN_RR on real levels, so the setup is genuinely vetoed. Before the
    # refinement this same setup reported 1.50 and sailed through.
    assert opp.rr < MIN_RR
    assert opp.quality is Quality.NO_OPERABLE
    assert opp.tradable is False
    assert opp.tradable is (opp.score >= MIN_EXECUTABLE_SCORE and opp.quality is not Quality.NO_OPERABLE)


def test_confluence_is_deterministic():
    snap = make_snapshot()
    signal = _signal()
    a = _engine().score(snap, signal, environment=TradingEnvironment.PAPER)
    b = _engine().score(snap, signal, environment=TradingEnvironment.PAPER)
    assert a.score == b.score
    assert a.quality is b.quality


def test_confluence_recorded_environment():
    opp = _engine().score(
        make_snapshot(), _signal(), environment=TradingEnvironment.REAL
    )
    assert opp.environment is TradingEnvironment.REAL


def test_confluence_breakdown_sums_components():
    opp = _engine().score(make_snapshot(), _signal(), environment=TradingEnvironment.PAPER)
    d = opp.breakdown.as_dict()
    assert set(d) == set(WEIGHTS) | {"strategy_history", "symbol_history", "setup_history"}
    assert all(v >= 0 for v in d.values())


def test_confluence_poorer_data_lowers_score():
    signal = _signal()
    good = _engine().score(make_snapshot(data_quality=1.0), signal, environment=TradingEnvironment.PAPER)
    bad = _engine().score(make_snapshot(data_quality=0.3), signal, environment=TradingEnvironment.PAPER)
    assert bad.score <= good.score


def test_entry_quality_render_contains_required_blocks():
    snap = make_snapshot()
    opp = _engine().score(snap, _signal(), environment=TradingEnvironment.PAPER)
    text = format_entry_quality(opp, snap)
    for marker in ("POR", "CONFIRMA", "SCORE", "INVALID", "RIESGO"):
        assert marker in text.upper()


def test_entry_quality_warns_on_small_sample():
    snap = make_snapshot()
    opp = _engine().score(
        snap,
        _signal(),
        environment=TradingEnvironment.PAPER,
        historical={"total_sample": 3, "win_rate": 0.8, "avg_r": 1.1},
    )
    assert "muestra" in format_entry_quality(opp, snap).lower()


def test_build_strategies_covers_registry():
    from botalpaca.strategies.registry import STRATEGY_REGISTRY

    strategies = build_strategies()
    assert len(strategies) == len(STRATEGY_REGISTRY)


def _signal(*, entry: float = 100.0, stop: float = 97.0, target: float = 106.0):
    from botalpaca.domain.models import StrategySignal

    return StrategySignal(
        strategy=SetupType.PULLBACK,
        direction=SignalDirection.LONG,
        score=75.0,
        entry=entry,
        stop=stop,
        timeframe="1D",
        targets=[target],
        rr=(target - entry) / (entry - stop),
        reasons=["pullback to EMA20"],
        confluences=["trend", "structure"],
    )
