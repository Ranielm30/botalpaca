"""Risk engine: position sizing and the full rule set."""

from __future__ import annotations

import datetime as dt

import pytest

from botalpaca.config.settings import RiskSettings
from botalpaca.db.repositories import DailyPnlRepository
from botalpaca.domain.enums import TradingEnvironment
from botalpaca.risk.engine import PortfolioRiskContext, RiskEngine

from .conftest import FakePortfolioService, FakeTradingClient, make_opportunity


def _engine(db, **overrides) -> RiskEngine:
    return RiskEngine(db, FakePortfolioService(FakeTradingClient()), settings=RiskSettings(**overrides))


def _context(**over) -> PortfolioRiskContext:
    base = dict(
        environment=TradingEnvironment.PAPER,
        equity=100_000.0,
        buying_power=200_000.0,
        cash=100_000.0,
    )
    base.update(over)
    return PortfolioRiskContext(**base)


# -- sizing -------------------------------------------------------------------------
def test_position_size_uses_risk_pct(database):
    qty, risk, dist = _engine(database).calculate_position_size(
        equity=100_000.0, entry=100.0, stop=98.0, risk_pct=1.0
    )
    assert dist == pytest.approx(2.0)
    assert qty == 500  # 1000 USD risk / 2 USD per share
    assert risk == pytest.approx(1000.0)


def test_position_size_is_whole_shares(database):
    engine = _engine(database)
    qty, _, _ = engine.calculate_position_size(equity=10_000.0, entry=100.0, stop=99.5, risk_pct=1.0)
    assert qty == int(qty)


def test_position_size_zero_for_degenerate_input(database):
    engine = _engine(database)
    assert engine.calculate_position_size(equity=1000.0, entry=100.0, stop=100.0)[0] == 0.0
    assert engine.calculate_position_size(equity=1000.0, entry=0.0, stop=1.0)[0] == 0.0
    assert engine.calculate_position_size(equity=1000.0, entry=100.0, stop=120.0)[0] == 0.0


def test_position_size_respects_max_qty_cap(database):
    qty, _, _ = _engine(database).calculate_position_size(
        equity=1_000_000.0, entry=100.0, stop=98.0, risk_pct=1.0, max_qty_cap=50.0
    )
    assert qty == 50


async def test_assess_approves_sane_opportunity(database):
    engine = _engine(database)
    opp = make_opportunity()
    result = await engine.assess(
        opp, _context(), average_dollar_volume=5_000_000.0, spread_pct=0.02
    )
    assert result.approved is True
    assert not result.blocks
    assert result.suggested_qty > 0
    assert 0 <= result.score <= 100
    assert result.risk_per_trade_pct == pytest.approx(1.0)


async def test_assess_blocks_environment_mismatch(database):
    engine = _engine(database)
    opp = make_opportunity(environment=TradingEnvironment.REAL)
    result = await engine.assess(opp, _context(environment=TradingEnvironment.PAPER))
    assert result.approved is False
    assert any("entorno" in b.lower() or "environment" in b.lower() for b in result.blocks)


async def test_assess_blocks_untradable(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(tradable=False), _context())
    assert not result.approved


async def test_assess_blocks_tiny_equity(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(), _context(equity=50.0))
    assert not result.approved


async def test_assess_blocks_blocked_account(database):
    from botalpaca.domain.models import AccountSnapshot

    engine = _engine(database)
    account = AccountSnapshot(
        environment=TradingEnvironment.PAPER,
        account_id="a",
        status="ACTIVE",
        trading_blocked=True,
    )
    result = await engine.assess(make_opportunity(), _context(account=account))
    assert not result.approved


async def test_assess_blocks_min_rr(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(rr=1.0, target=102.0), _context())
    assert not result.approved
    assert any("r:r" in b.lower() or "riesgo/beneficio" in b.lower() for b in result.blocks)


async def test_assess_blocks_zero_width_stop(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(stop=100.0, target=106.0), _context())
    assert not result.approved


async def test_assess_blocks_stop_too_tight(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(stop=99.95, target=110.0, rr=200.0), _context())
    assert not result.approved


async def test_assess_blocks_stop_too_wide(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(stop=50.0, target=250.0, rr=3.0), _context())
    assert not result.approved


async def test_assess_blocks_max_positions(database):
    engine = _engine(database, max_open_positions=1)
    from botalpaca.risk.engine import OpenExposure

    ctx = _context(open_positions=[OpenExposure(symbol="MSFT", sector="TECHNOLOGY", notional=10_000.0)])
    result = await engine.assess(make_opportunity(), ctx)
    assert not result.approved
    assert any("posiciones" in b.lower() for b in result.blocks)


async def test_same_symbol_position_is_exempt_from_max_positions(database):
    engine = _engine(database, max_open_positions=1)
    from botalpaca.risk.engine import OpenExposure

    ctx = _context(open_positions=[OpenExposure(symbol="AAPL", sector="TECHNOLOGY", notional=10_000.0)])
    result = await engine.assess(make_opportunity(symbol="AAPL"), ctx)
    assert result.approved is True


async def test_assess_blocks_sector_concentration(database):
    engine = _engine(database, max_sector_exposure_pct=10.0)
    from botalpaca.risk.engine import OpenExposure

    ctx = _context(open_positions=[OpenExposure(symbol="MSFT", sector="TECHNOLOGY", notional=50_000.0)])
    result = await engine.assess(make_opportunity(), ctx)
    assert not result.approved


async def test_assess_blocks_illiquid_symbol(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(), _context(), average_dollar_volume=1_000.0)
    assert not result.approved
    assert any("liquidez" in b.lower() for b in result.blocks)


async def test_assess_blocks_wide_spread(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(), _context(), spread_pct=5.0)
    assert not result.approved


async def test_assess_blocks_insufficient_buying_power(database):
    engine = _engine(database)
    result = await engine.assess(make_opportunity(), _context(buying_power=1.0, equity=1000.0))
    assert not result.approved


async def test_assess_trims_when_notional_above_cap(database):
    engine = _engine(database, max_order_notional=5_000.0, min_order_notional=50.0)
    result = await engine.assess(make_opportunity(), _context())
    assert result.suggested_qty * 100.0 <= 5_000.0 + 1e-6
    assert result.reasons


# -- the per-trade cap is a share of the capital still free ----------------------
# Measured against the whole account, the first trade takes the same allowance
# the tenth would get and everything after it is squeezed down to the crumbs.
# Measured against what is undeployed, each order takes its cut of the remainder.


async def test_the_first_order_measured_the_whole_account(database):
    engine = _engine(
        database,
        max_concentration_pct=25.0,
        max_sector_exposure_pct=100.0,
        max_total_exposure_pct=100.0,
        max_correlated_exposure_pct=100.0,
    )
    result = await engine.assess(make_opportunity(), _context(equity=100_000.0))
    # Nothing is invested yet, so free capital is the whole 100k.
    assert result.suggested_qty * 100.0 == pytest.approx(25_000.0)


async def test_each_order_takes_a_quarter_of_what_is_left_free(database):
    from botalpaca.risk.engine import OpenExposure

    engine = _engine(
        database,
        max_concentration_pct=25.0,
        max_sector_exposure_pct=100.0,
        max_total_exposure_pct=100.0,
        max_correlated_exposure_pct=100.0,
    )
    ctx = _context(
        equity=100_000.0,
        open_positions=[OpenExposure(symbol="AAPL", sector="TECHNOLOGY", notional=75_000.0)],
    )
    result = await engine.assess(make_opportunity(), ctx)
    # 25% of the remaining 25k, not 25% of the 100k account.
    assert result.suggested_qty * 100.0 == pytest.approx(6_200.0)  # 62 whole shares
    assert result.approved


async def test_the_cap_shrinks_as_the_book_fills(database):
    from botalpaca.risk.engine import OpenExposure

    engine = _engine(
        database,
        max_concentration_pct=25.0,
        max_sector_exposure_pct=100.0,
        max_total_exposure_pct=100.0,
        max_correlated_exposure_pct=100.0,
    )
    sizes = []
    for invested in (0.0, 25_000.0, 50_000.0):
        ctx = _context(
            equity=100_000.0,
            open_positions=[OpenExposure(symbol="X", sector="TECHNOLOGY", notional=invested)],
        )
        result = await engine.assess(make_opportunity(), ctx)
        sizes.append(result.suggested_qty * 100.0)
    assert sizes == pytest.approx([25_000.0, 18_700.0, 12_500.0])


async def test_nothing_left_free_blocks_the_next_order(database):
    from botalpaca.risk.engine import OpenExposure

    engine = _engine(
        database,
        max_concentration_pct=25.0,
        max_sector_exposure_pct=100.0,
        max_total_exposure_pct=100.0,
        max_correlated_exposure_pct=100.0,
    )
    ctx = _context(
        equity=100_000.0,
        open_positions=[OpenExposure(symbol="X", sector="TECHNOLOGY", notional=100_000.0)],
    )
    result = await engine.assess(make_opportunity(), ctx)
    assert not result.approved
    assert any("libre" in b for b in result.blocks)


async def test_the_free_cap_does_not_inflate_a_small_risk_budget(database):
    from botalpaca.risk.engine import OpenExposure

    engine = _engine(
        database,
        max_concentration_pct=25.0,
        max_sector_exposure_pct=100.0,
        max_total_exposure_pct=100.0,
        max_correlated_exposure_pct=100.0,
    )
    ctx = _context(
        equity=100_000.0,
        open_positions=[OpenExposure(symbol="X", sector="TECHNOLOGY", notional=75_000.0)],
    )
    # A stop 20 wide costs 20 a share, so the 1% budget only reaches 50 shares.
    result = await engine.assess(make_opportunity(stop=80.0, target=120.0, rr=2.0), ctx)
    assert result.suggested_qty == pytest.approx(50.0)


async def test_daily_loss_limit_blocks(database):
    engine = _engine(database, max_daily_loss_pct=1.0)
    repo = DailyPnlRepository
    async with database.session() as session:
        row = await repo(session).upsert_day(TradingEnvironment.PAPER, dt.date.today())
        row.realized_pnl = -5_000.0
    result = await engine.assess(make_opportunity(), _context())
    assert not result.approved
    assert any("diaria" in b.lower() or "diario" in b.lower() for b in result.blocks)


async def test_monthly_loss_limit_blocks(database):
    engine = _engine(database, max_monthly_loss_pct=1.0)
    async with database.session() as session:
        repo = DailyPnlRepository(session)
        today = dt.date.today()
        for day in (today, today - dt.timedelta(days=1), today - dt.timedelta(days=2)):
            row = await repo.upsert_day(TradingEnvironment.PAPER, day)
            row.realized_pnl = -2_000.0
    result = await engine.assess(make_opportunity(), _context())
    assert not result.approved


async def test_real_environment_history_does_not_affect_paper(database):
    """Isolation: REAL losses must never block PAPER."""
    engine = _engine(database, max_daily_loss_pct=1.0)
    async with database.session() as session:
        repo = DailyPnlRepository(session)
        row = await repo.upsert_day(TradingEnvironment.REAL, dt.date.today())
        row.realized_pnl = -50_000.0
    result = await engine.assess(make_opportunity(), _context(environment=TradingEnvironment.PAPER))
    assert result.approved is True


async def test_open_risk_reduces_per_trade_budget(database):
    engine = _engine(database, max_drawdown_pct=5.0, max_risk_per_trade_pct=1.0)
    ctx = _context(open_risk=40_000.0)  # 40% of 100k
    result = await engine.assess(make_opportunity(), ctx)
    assert result.risk_per_trade_pct < 1.0


async def test_build_context_reads_account_and_positions(database):
    client = FakeTradingClient()
    client.positions = []
    portfolio = FakePortfolioService(client, equity=50_000.0)
    engine = RiskEngine(database, portfolio)
    ctx = await engine.build_context(TradingEnvironment.PAPER)
    assert ctx.environment is TradingEnvironment.PAPER
    assert ctx.equity == pytest.approx(50_000.0)
