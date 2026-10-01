"""Trade journal + statistical learning engine."""

from __future__ import annotations

import datetime as dt

import pytest

from botalpaca.db.repositories import DailyPnlRepository, SignalRepository, TradeRepository
from botalpaca.domain.enums import MarketRegime, SetupType, SignalDirection, TradingEnvironment
from botalpaca.journal.service import TradeJournal
from botalpaca.learning.engine import StatisticalEngine

from .conftest import make_opportunity

PAPER = TradingEnvironment.PAPER
REAL = TradingEnvironment.REAL


async def _open(journal: TradeJournal, *, environment=PAPER, symbol="AAPL", r: float = 1.0, score=80.0, **over):
    kwargs = dict(
        environment=environment,
        symbol=symbol,
        direction=SignalDirection.LONG,
        strategy=SetupType.PULLBACK,
        setup=SetupType.PULLBACK,
        timeframe="1D",
        qty=10.0,
        entry_price=100.0,
        stop_price=98.0,
        target_price=106.0,
        score=score,
        rr=2.0,
        risk_amount=20.0,
        atr=2.5,
        regime=MarketRegime.TRENDING_UP.value,
        sector="TECHNOLOGY",
    )
    kwargs.update(over)
    return await journal.open_trade(**kwargs)


# -- journal ------------------------------------------------------------------------
async def test_open_and_close_trade_computes_r(database):
    journal = TradeJournal(database)
    trade = await _open(journal)
    assert trade.status.value == "OPEN" if hasattr(trade.status, "value") else trade.status == "OPEN"
    closed = await journal.close_trade(
        environment=PAPER, symbol="AAPL", exit_price=102.0, pnl=20.0, exit_reason="target"
    )
    assert closed is not None
    assert closed.r_multiple == pytest.approx(1.0)
    assert closed.pnl_pct == pytest.approx(2.0)
    assert closed.duration_seconds is not None


async def test_close_trade_without_open_position_returns_none(database):
    journal = TradeJournal(database)
    assert await journal.close_trade(
        environment=PAPER, symbol="TSLA", exit_price=1.0, pnl=0.0, exit_reason="manual"
    ) is None


async def test_close_trade_never_crosses_environments(database):
    journal = TradeJournal(database)
    await _open(journal, environment=PAPER)
    assert await journal.close_trade(
        environment=REAL, symbol="AAPL", exit_price=102.0, pnl=20.0, exit_reason="target"
    ) is None


async def test_close_trade_accumulates_daily_pnl(database):
    journal = TradeJournal(database)
    await _open(journal)
    await journal.close_trade(
        environment=PAPER, symbol="AAPL", exit_price=102.0, pnl=25.0, exit_reason="target"
    )
    async with database.session() as session:
        total = await DailyPnlRepository(session).realized_sum(
            PAPER, dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
        )
    assert total == pytest.approx(25.0)


async def test_explain_answers_why(database):
    journal = TradeJournal(database)
    opp = make_opportunity()
    await journal.record_signal(opp, accepted=True, indicators={"ema_20": 100.5})
    await _open(journal)
    await journal.close_trade(
        environment=PAPER, symbol="AAPL", exit_price=103.0, pnl=30.0, exit_reason="target"
    )
    answer = await journal.explain(PAPER, "AAPL")
    assert answer is not None
    assert answer["symbol"] == "AAPL"
    assert answer["trade"] is not None
    assert answer["signal"] is not None


async def test_explain_returns_none_when_nothing_is_known(database):
    journal = TradeJournal(database)
    assert await journal.explain(PAPER, "TSLA") is None


async def test_signal_recording_and_acceptance(database):
    journal = TradeJournal(database)
    opp = make_opportunity()
    sig = await journal.record_signal(opp, accepted=False, indicators={"rsi": 55})
    assert sig.accepted is False
    await journal.mark_signal_accepted(PAPER, opp.fingerprint)
    latest = await journal.latest_signal(PAPER, opp.fingerprint)
    assert latest is not None
    assert latest.accepted is True


async def test_record_signal_if_new_deduplicates(database):
    journal = TradeJournal(database)
    opp = make_opportunity()
    first = await journal.record_signal_if_new(opp, accepted=True, within_minutes=180)
    second = await journal.record_signal_if_new(opp, accepted=True, within_minutes=180)
    assert first is not None
    assert second is None
    async with database.session() as session:
        assert len(await SignalRepository(session).recent(PAPER, hours=24)) == 1


async def test_mark_signal_accepted_with_falsy_fingerprint_is_noop(database):
    journal = TradeJournal(database)
    assert await journal.mark_signal_accepted(PAPER, None) is None


async def test_trade_events_recorded(database):
    journal = TradeJournal(database)
    await _open(journal)
    await journal.close_trade(
        environment=PAPER, symbol="AAPL", exit_price=102.0, pnl=20.0, exit_reason="target"
    )
    async with database.session() as session:
        events = await TradeRepository(session).events(1, PAPER)
    assert len(events) >= 2


# -- learning -----------------------------------------------------------------------
async def test_overall_stats_empty_environment(database):
    engine = StatisticalEngine(database)
    summary = (await engine.overall(PAPER))
    assert summary.sample_size == 0
    assert summary.is_significant is False
    assert summary.caveat


async def _seed(journal: TradeJournal, n: int, *, environment=PAPER, symbol="AAPL", r_outcomes=None):
    for i in range(n):
        await _open(journal, environment=environment, symbol=symbol, score=70 + (i % 20))
        r = (r_outcomes[i % len(r_outcomes)]) if r_outcomes else 1.0
        exit_price = 100.0 + 2.0 * r
        pnl = 20.0 * r
        await journal.close_trade(
            environment=environment, symbol=symbol, exit_price=exit_price, pnl=pnl, exit_reason="target"
        )


async def test_overall_stats_win_rate_and_expectancy(database):
    journal = TradeJournal(database)
    await _seed(journal, 6, r_outcomes=[1.0, -1.0, 2.0, 1.0, -1.0, 1.0])
    summary = await StatisticalEngine(database).overall(PAPER)
    assert summary.sample_size == 6
    assert 0.0 < summary.win_rate < 1.0
    assert summary.expectancy_r != 0.0
    assert summary.average_r is not None


async def test_sharpe_only_reported_with_enough_sample(database):
    journal = TradeJournal(database)
    await _seed(journal, 4, r_outcomes=[1.0, 1.0, -1.0, 1.0])
    small = await StatisticalEngine(database).overall(PAPER)
    assert small.sharpe is None
    await _seed(journal, 30, r_outcomes=[1.0, 1.0, -1.0, 2.0])
    big = await StatisticalEngine(database).overall(PAPER)
    assert big.sharpe is not None


async def test_group_stats_are_isolated_per_environment(database):
    journal = TradeJournal(database)
    await _seed(journal, 4, environment=PAPER, r_outcomes=[1.0, 1.0, -1.0, 1.0])
    await _seed(journal, 2, environment=REAL, symbol="MSFT", r_outcomes=[-1.0, -1.0])
    engine = StatisticalEngine(database)
    assert (await engine.overall(PAPER)).sample_size == 4
    assert (await engine.overall(REAL)).sample_size == 2
    assert "MSFT" not in await engine.by_symbol(PAPER)
    assert "MSFT" in await engine.by_symbol(REAL)


async def test_by_strategy_and_setup(database):
    journal = TradeJournal(database)
    await _seed(journal, 4, r_outcomes=[1.0, 1.0, -1.0, 1.0])
    engine = StatisticalEngine(database)
    assert SetupType.PULLBACK.value in await engine.by_strategy(PAPER)
    assert SetupType.PULLBACK.value in await engine.by_setup(PAPER)
    assert "1D" in await engine.by_timeframe(PAPER)
    assert MarketRegime.TRENDING_UP.value in await engine.by_regime(PAPER)
    assert "TECHNOLOGY" in await engine.by_sector(PAPER)


async def test_score_bands_and_calendar_buckets(database):
    journal = TradeJournal(database)
    await _seed(journal, 4, r_outcomes=[1.0, 1.0, -1.0, 1.0])
    engine = StatisticalEngine(database)
    assert await engine.by_score_band(PAPER)
    assert await engine.by_weekday(PAPER)
    assert await engine.by_hour(PAPER)


async def test_similar_setups_excludes_current_trade(database):
    journal = TradeJournal(database)
    await _seed(journal, 5, r_outcomes=[1.0, 1.0, 1.0, -1.0, 1.0])
    engine = StatisticalEngine(database)
    async with database.session() as session:
        trades = await TradeRepository(session).get_closed(PAPER)
    exclude = trades[0].id
    similar = await engine.similar_setups(
        PAPER,
        strategy=SetupType.PULLBACK.value,
        setup=SetupType.PULLBACK.value,
        regime=MarketRegime.TRENDING_UP.value,
        exclude_trade_id=exclude,
    )
    assert similar.sample_size == 4


async def test_similar_setups_returns_none_without_history(database):
    similar = await StatisticalEngine(database).similar_setups(PAPER)
    assert similar is None


async def test_recommendations_are_non_binding_text(database):
    journal = TradeJournal(database)
    await _seed(journal, 3, r_outcomes=[-1.0, -1.0, -1.0])
    recs = await StatisticalEngine(database).recommendations(PAPER)
    assert isinstance(recs, list)
    assert all(isinstance(r, str) for r in recs)


async def test_symbol_detail(database):
    journal = TradeJournal(database)
    await _seed(journal, 4, r_outcomes=[1.0, 1.0, -1.0, 1.0])
    engine = StatisticalEngine(database)
    detail = await engine.symbol_detail(PAPER, "AAPL")
    assert detail is not None
    assert detail.total_trades == 4
    assert detail.wins == 3
    assert detail.win_rate == pytest.approx(0.75)
    assert (await engine.symbol_detail(PAPER, "TSLA")) is None
