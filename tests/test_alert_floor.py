"""A tradable opportunity must actually reach Telegram.

The scan reported ten tradable setups and nothing was ever delivered, for
hours. The scanner's execution floor had been lowered to 60 while the market
monitor's own alert floor sat at 75, so every opportunity the scanner called
tradable was discarded one line later, in silence, with nothing in the logs.

Two things are pinned here: the defaults must agree, and a deliberate mismatch
must be stated at boot instead of being discovered by waiting.
"""

from __future__ import annotations

import pytest

from botalpaca.config import MonitoringSettings, get_settings
from botalpaca.confluence.engine import MIN_EXECUTABLE_SCORE
from tests.test_scanner_monitoring_app import (
    PAPER,
    FakeMarketService,
    _StubScanner,
    make_opportunity,
)


def test_the_default_alert_floor_is_not_above_the_execution_floor():
    """The two floors are separate knobs, so their defaults must agree."""
    settings = MonitoringSettings()
    assert settings.opportunity_alert_min_score <= MIN_EXECUTABLE_SCORE, (
        "an alert floor above the execution floor silently discards every "
        "opportunity the scanner just called tradable"
    )


def test_the_deployed_default_is_not_above_the_execution_floor():
    """Guard the settings object the process actually boots with."""
    settings = get_settings()
    assert settings.monitoring.opportunity_alert_min_score <= MIN_EXECUTABLE_SCORE


async def test_a_tradable_opportunity_at_the_floor_is_delivered(database):
    """Exactly the case that vanished: tradable, and above the default floor."""
    sent: list[str] = []

    async def notify(text, keyboard=None):
        sent.append(text)

    monitor = __import__(
        "botalpaca.monitoring.service", fromlist=["MarketMonitor"]
    ).MarketMonitor(
        _StubScanner([make_opportunity(score=MIN_EXECUTABLE_SCORE)]),
        FakeMarketService(),
        database,
        environment=PAPER,
        notify=notify,
        settings=MonitoringSettings(),
    )
    await monitor.run_once()

    assert sent, "a tradable opportunity must not be discarded by the alert floor"


async def test_the_operator_may_still_ask_for_a_stricter_floor(database):
    """Raising it on purpose is allowed -- it just has to be a deliberate choice."""
    sent: list[str] = []

    async def notify(text, keyboard=None):
        sent.append(text)

    monitor = __import__(
        "botalpaca.monitoring.service", fromlist=["MarketMonitor"]
    ).MarketMonitor(
        _StubScanner([make_opportunity(score=80)]),
        FakeMarketService(),
        database,
        environment=PAPER,
        notify=notify,
        settings=MonitoringSettings(opportunity_alert_min_score=95),
    )
    await monitor.run_once()
    assert not sent


def _record_boot(monkeypatch, settings: MonitoringSettings) -> list[str]:
    """Capture the warning the monitor emits on construction.

    structlog does not route through the stdlib logging caplog fixture, so the
    module logger is swapped for a recorder instead.
    """
    from botalpaca.monitoring import service as monitoring

    events: list[str] = []

    class _Log:
        def __getattr__(self, _name):
            def _record(event, **fields):
                events.append(event)

            return _record

    monkeypatch.setattr(monitoring, "log", _Log())
    monitoring.MarketMonitor(
        _StubScanner([]),
        FakeMarketService(),
        None,
        environment=PAPER,
        settings=settings,
    )
    return events


def test_a_mismatched_floor_is_announced_at_boot(monkeypatch):
    """It must be said out loud, not discovered hours later by not receiving."""
    events = _record_boot(monkeypatch, MonitoringSettings(opportunity_alert_min_score=95))
    assert "market_monitor.alert_floor_above_execution_floor" in events


def test_a_matching_floor_says_nothing(monkeypatch):
    events = _record_boot(monkeypatch, MonitoringSettings())
    assert "market_monitor.alert_floor_above_execution_floor" not in events


@pytest.mark.parametrize("score", [60, 65, 70, 74, 75, 80])
async def test_nothing_in_the_deliverable_band_is_dropped(database, score):
    """Every score the scanner can call tradable has to arrive."""
    sent: list[str] = []

    async def notify(text, keyboard=None):
        sent.append(text)

    monitor = __import__(
        "botalpaca.monitoring.service", fromlist=["MarketMonitor"]
    ).MarketMonitor(
        _StubScanner([make_opportunity(score=score)]),
        FakeMarketService(),
        database,
        environment=PAPER,
        notify=notify,
        settings=MonitoringSettings(),
    )
    await monitor.run_once()
    assert sent, f"score {score} was tradable and never arrived"
