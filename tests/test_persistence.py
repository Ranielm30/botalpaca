"""SQLite persistence: repositories, isolation, and restart recovery."""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from botalpaca.db.repositories import (
    AppStateRepository,
    DailyPnlRepository,
    OrderAuditRepository,
    ProtectionRepository,
    SettingRepository,
    SignalRepository,
    TradeRepository,
)
from botalpaca.db.session import Database
from botalpaca.domain.enums import MarketRegime, SetupType, TradingEnvironment

PAPER = TradingEnvironment.PAPER
REAL = TradingEnvironment.REAL


# -- database lifecycle -------------------------------------------------------------
async def test_database_healthcheck(settings):
    db = Database(settings.database_url, create_all=True)
    await db.start()
    try:
        assert await db.healthcheck() is True
    finally:
        await db.stop()


async def test_database_engine_requires_start(settings):
    db = Database(settings.database_url, create_all=False)
    with pytest.raises(RuntimeError):
        _ = db.engine


async def test_database_creates_parent_directory(tmp_path):
    target = tmp_path / "nested" / "dir" / "botalpaca.db"
    db = Database(f"sqlite+aiosqlite:///{target.as_posix()}", create_all=True)
    await db.start()
    try:
        assert target.parent.exists()
    finally:
        await db.stop()


# -- app state ---------------------------------------------------------------------
async def test_app_state_roundtrip(database):
    async with database.session() as session:
        repo = AppStateRepository(session)
        await repo.set("kill_switch", True)
        assert await repo.get("kill_switch") is True
        await repo.set("kill_switch", False)
        assert await repo.get("kill_switch") is False
        assert await repo.get("missing", "fallback") == "fallback"
        assert "kill_switch" in await repo.all_state()
        await repo.delete("kill_switch")
        assert "kill_switch" not in await repo.all_state()


async def test_app_state_survives_reconnection(settings):
    db = Database(settings.database_url, create_all=True)
    await db.start()
    async with db.session() as session:
        await AppStateRepository(session).set("active_environment", REAL)
    await db.stop()

    db2 = Database(settings.database_url, create_all=False)
    await db2.start()
    try:
        async with db2.session() as session:
            stored = await AppStateRepository(session).get("active_environment")
            assert TradingEnvironment(stored) == REAL
    finally:
        await db2.stop()


# -- settings ----------------------------------------------------------------------
async def test_settings_roundtrip(database):
    async with database.session() as session:
        repo = SettingRepository(session)
        await repo.set("max_risk", 1.5, updated_by=111)
        assert await repo.get("max_risk") == 1.5
        assert await repo.get("missing", default=7) == 7
        assert "max_risk" in await repo.all_settings()
        await repo.delete("max_risk")
        assert "max_risk" not in await repo.all_settings()


# -- trades ------------------------------------------------------------------------
def _trade(*, environment=PAPER, symbol="AAPL", status="OPEN"):
    from botalpaca.db.models import TradeModel

    return TradeModel(
        environment=environment.value,
        symbol=symbol,
        direction="LONG",
        strategy=SetupType.PULLBACK.value,
        setup=SetupType.PULLBACK.value,
        timeframe="1D",
        sector="TECHNOLOGY",
        regime=MarketRegime.TRENDING_UP.value,
        status=status,
        qty=10.0,
        entry_price=100.0,
        stop_price=98.0,
        target_price=106.0,
        score=80.0,
        rr=2.0,
        risk_amount=20.0,
        atr=2.5,
        has_stop=True,
    )


async def test_trade_repository_requires_environment(database):
    async with database.session() as session:
        repo = TradeRepository(session)
        await repo.create(_trade())
        assert await repo.count_open(PAPER) == 1
        assert await repo.count_open(REAL) == 0
        assert await repo.get_open_for_symbol("AAPL", PAPER) is not None
        assert await repo.get_open_for_symbol("AAPL", REAL) is None


async def test_trade_totals_are_isolated(database):
    async with database.session() as session:
        repo = TradeRepository(session)
        await repo.create(_trade(environment=PAPER, symbol="AAPL"))
        await repo.create(_trade(environment=REAL, symbol="MSFT"))
        assert await repo.total_open_exposure(PAPER) == pytest.approx(1000.0)
        assert await repo.total_open_exposure(REAL) == pytest.approx(1000.0)
        assert len(await repo.get_open_positions(PAPER)) == 1
        assert (await repo.get_open_positions(PAPER))[0].symbol == "AAPL"


async def test_trade_exposure_grouping(database):
    async with database.session() as session:
        repo = TradeRepository(session)
        await repo.create(_trade(symbol="AAPL"))
        await repo.create(_trade(symbol="MSFT"))
        assert (await repo.exposure_by_sector(PAPER))["TECHNOLOGY"] == pytest.approx(2000.0)
        assert (await repo.exposure_by_symbol(PAPER))["AAPL"] == pytest.approx(1000.0)


async def test_trade_events_append_only(database):
    async with database.session() as session:
        repo = TradeRepository(session)
        trade = await repo.create(_trade())
        await repo.add_event(trade.id, PAPER, "OPENED", {"note": "entry"})
        await repo.add_event(trade.id, PAPER, "NOTE", {"note": "checked"})
        events = await repo.events(trade.id, PAPER)
    assert [e.event_type for e in events] == ["OPENED", "NOTE"]


async def test_trade_lookup_by_client_order_id(database):
    async with database.session() as session:
        repo = TradeRepository(session)
        t = _trade()
        t.client_order_id = "cid-123"
        trade = await repo.create(t)
        assert (await repo.get_by_client_order_id("cid-123", PAPER)).id == trade.id
        assert await repo.get_by_client_order_id("cid-123", REAL) is None


# -- signals -----------------------------------------------------------------------
def _signal(environment=PAPER, *, fingerprint="fp-1"):
    from botalpaca.db.models import SignalModel

    return SignalModel(
        environment=environment.value,
        symbol="AAPL",
        timeframe="1D",
        htf_timeframe="1W",
        direction="LONG",
        strategy=SetupType.PULLBACK.value,
        setup=SetupType.PULLBACK.value,
        quality="ALTA",
        score=88.0,
        rr=2.0,
        entry=100.0,
        stop=98.0,
        target=106.0,
        fingerprint=fingerprint,
    )


async def test_signal_fingerprint_dedup(database):
    async with database.session() as session:
        repo = SignalRepository(session)
        model = await repo.create(_signal())
        assert model.id is not None
        assert await repo.exists_fingerprint(PAPER, "fp-1") is True
        assert await repo.get_latest_by_fingerprint(PAPER, "fp-1") is not None
        assert await repo.get_latest_by_fingerprint(REAL, "fp-1") is None


async def test_signal_set_accepted_and_rejected(database):
    async with database.session() as session:
        repo = SignalRepository(session)
        row = await repo.create(_signal(fingerprint="fp-2"))
        await repo.set_accepted(row.id, PAPER)
        assert (await repo.get(row.id, PAPER)).accepted is True

        other = await repo.create(_signal(fingerprint="fp-3"))
        await repo.set_rejected(other.id, PAPER, "score bajo")
        rejected = await repo.get(other.id, PAPER)
        assert rejected.accepted is False
        assert rejected.rejected_reason == "score bajo"

        await repo.mark_notified(row.id, PAPER)
        assert (await repo.get(row.id, PAPER)).notified is True


async def test_signal_recent_filters_by_symbol(database):
    async with database.session() as session:
        repo = SignalRepository(session)
        await repo.create(_signal(fingerprint="fp-a"))
        other = _signal(fingerprint="fp-b")
        other.symbol = "MSFT"
        await repo.create(other)
        assert len(await repo.recent(PAPER, hours=24)) == 2
        assert len(await repo.recent(PAPER, hours=24, symbol="MSFT")) == 1


# -- daily pnl ---------------------------------------------------------------------
async def test_daily_pnl_upsert_is_idempotent(database):
    async with database.session() as session:
        repo = DailyPnlRepository(session)
        first = await repo.upsert_day(PAPER, dt.date(2024, 6, 3))
        second = await repo.upsert_day(PAPER, dt.date(2024, 6, 3))
        assert first.id == second.id
        first.realized_pnl = 100.0
        await session.flush()
        assert (await repo.get(PAPER, dt.date(2024, 6, 3))).realized_pnl == pytest.approx(100.0)


async def test_daily_pnl_isolated_by_environment(database):
    async with database.session() as session:
        repo = DailyPnlRepository(session)
        paper = await repo.upsert_day(PAPER, dt.date(2024, 6, 3))
        real = await repo.upsert_day(REAL, dt.date(2024, 6, 3))
        assert paper.id != real.id


# -- protection --------------------------------------------------------------------
async def test_protection_upsert_and_isolation(database):
    async with database.session() as session:
        repo = ProtectionRepository(session)
        await repo.upsert(PAPER, "AAPL", stop_price=97.0, stop_order_id="s1", qty=10)
        await repo.upsert(PAPER, "AAPL", stop_price=98.0)
        row = await repo.get(PAPER, "AAPL")
        assert row.stop_price == pytest.approx(98.0)
        assert row.stop_order_id == "s1"
        assert await repo.get(REAL, "AAPL") is None
        assert len(await repo.all_for(PAPER)) == 1
        await repo.delete(PAPER, "AAPL")
        assert await repo.get(PAPER, "AAPL") is None


async def test_protection_upsert_ignores_unknown_fields(database):
    async with database.session() as session:
        repo = ProtectionRepository(session)
        await repo.upsert(PAPER, "AAPL", stop_price=97.0, totally_unknown="x")
        row = await repo.get(PAPER, "AAPL")
        assert row is not None
        assert not hasattr(row, "totally_unknown")


# -- order audit -------------------------------------------------------------------
async def test_order_audit_idempotency_lookup(database):
    from botalpaca.db.models import OrderAuditModel

    async with database.session() as session:
        repo = OrderAuditRepository(session)
        await repo.log(
            OrderAuditModel(
                environment=PAPER.value,
                action="SUBMIT",
                symbol="AAPL",
                status="SUBMITTED",
                client_order_id="cid-1",
                idempotency_key="key-1",
            )
        )
        assert await repo.find_by_idempotency_key(PAPER, "key-1") is not None
        assert await repo.find_by_idempotency_key(REAL, "key-1") is None
        assert await repo.client_order_exists("cid-1") is True


def alembic_head() -> str:
    """Read the head revision from the migration scripts themselves.

    Hardcoding a revision id makes the test fail every time a migration is
    added; the contract under test is "start() brings the database to head".
    """
    from alembic.script import ScriptDirectory

    from botalpaca.db.migrate import alembic_config

    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    assert head is not None
    return head


async def test_migrations_run_on_start(settings):
    from botalpaca.db.migrate import current_revision

    db = Database(settings.database_url, create_all=False, migrate=True)
    await db.start()
    try:
        revision = await asyncio.to_thread(current_revision, settings.database_url)
        assert revision == alembic_head()
        assert await db.healthcheck() is True
    finally:
        await db.stop()
