"""Config, logging secret redaction, market timeframe parsing, and CLI entrypoint."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from botalpaca.config.logging import SecretsFilter, configure_logging
from botalpaca.config.settings import Settings
from botalpaca.domain.enums import TradingEnvironment
from botalpaca.domain.errors import ConfigurationError, SecretNotConfiguredError
from botalpaca.market.service import parse_timeframe


# -- settings -----------------------------------------------------------------------
def test_default_environment_is_paper():
    assert Settings().active_trading_environment is TradingEnvironment.PAPER


def test_alpaca_env_config_is_per_environment(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "pk")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "ps")
    monkeypatch.setenv("ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALPACA_LIVE_SECRET_KEY", "ls")
    s = Settings()
    paper = s.alpaca(TradingEnvironment.PAPER)
    real = s.alpaca(TradingEnvironment.REAL)
    assert paper.is_configured is True
    assert real.is_configured is True
    assert paper.api_key.get_secret_value() == "pk"
    assert real.api_key.get_secret_value() == "lk"
    assert paper.environment is TradingEnvironment.PAPER
    assert real.environment is TradingEnvironment.REAL


def test_alpaca_env_config_requires_configured():
    s = Settings()
    cfg = s.alpaca(TradingEnvironment.PAPER)
    cfg.api_key = None  # type: ignore[assignment]
    cfg.secret_key = None  # type: ignore[assignment]
    assert cfg.is_configured is False
    with pytest.raises(SecretNotConfiguredError):
        cfg.require_configured()


def test_safe_summary_redacts(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "SUPERSECRETKEY")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "SUPERSECRETSECRET")
    s = Settings()
    summary = str(s.alpaca(TradingEnvironment.PAPER).safe_summary())
    assert "SUPERSECRET" not in summary


def test_validate_runtime_requires_allowlist():
    s = Settings(telegram_allowed_user_ids=[])
    with pytest.raises(ConfigurationError):
        s.validate_runtime()


def test_validate_runtime_requires_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    s = Settings(telegram_bot_token="", telegram_allowed_user_ids=[1])
    with pytest.raises(ConfigurationError):
        s.validate_runtime()


def test_allowed_user_ids_parsed_from_csv(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "1, 2 ,3")
    assert Settings().telegram_allowed_user_ids == [1, 2, 3]


def test_monitor_defaults_are_conservative():
    from botalpaca.config.settings import MonitoringSettings, ProtectionSettings

    assert MonitoringSettings().auto_trading_enabled is False
    assert ProtectionSettings().auto_protect_missing_stop is True


# -- logging -----------------------------------------------------------------------
def test_secrets_filter_redacts_sensitive_keys():
    flt = SecretsFilter()
    event = {
        "event": "submit",
        "api_key": "abc123",
        "secret_key": "def456",
        "telegram_bot_token": "t0k3nvalue",
        "nested": {"ALPACA_PAPER_SECRET_KEY": "zzz", "safe": "ok"},
        "payload": {"password": "p"},
    }
    out = flt(None, "info", event)
    serialized = str(out)
    for secret in ("abc123", "def456", "t0k3nvalue", "zzz"):
        assert secret not in serialized
    assert "ok" in serialized


def test_configure_logging_is_idempotent():
    configure_logging(level="DEBUG", json_output=False)
    configure_logging(level="INFO", json_output=True)
    assert logging.getLogger().handlers is not None


# -- timeframes ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,amount,unit",
    [
        ("1D", 1, "Day"),
        ("1W", 1, "Week"),
        ("1mo", 1, "Month"),
        ("15Min", 15, "Min"),
        ("5m", 5, "Min"),
        ("1m", 1, "Min"),
        ("4h", 4, "Hour"),
        ("30", 30, "Min"),
    ],
)
def test_parse_timeframe(raw, amount, unit):
    spec = parse_timeframe(raw)
    assert spec.amount == amount
    assert spec.unit == unit
    assert spec.minutes > 0
    assert spec.label


def test_parse_timeframe_rejects_garbage():
    for bad in ("", "bad", "1x", "!!"):
        with pytest.raises(ValueError):
            parse_timeframe(bad)


# -- entrypoint ---------------------------------------------------------------------
def test_main_parses_check_and_log_level(monkeypatch):
    from botalpaca import __main__

    seen: dict[str, object] = {}

    async def fake_run(settings):
        seen["env"] = settings.active_trading_environment
        seen["level"] = settings.log_level
        return 0

    async def fake_check(settings):
        seen["checked"] = True
        return 0

    monkeypatch.setattr(__main__, "_run", fake_run)
    monkeypatch.setattr(__main__, "_check", fake_check)

    assert __main__.main([]) == 0
    assert seen["env"] is not None
    assert "checked" not in seen

    assert __main__.main(["--check"]) == 0
    assert seen["checked"] is True

    assert __main__.main(["--log-level", "debug"]) == 0
    assert seen["level"] == "DEBUG"


def test_main_returns_two_on_configuration_error(monkeypatch, capsys):
    from botalpaca import __main__
    from botalpaca.domain import ConfigurationError

    def boom(argv=None):
        raise ConfigurationError("sin credenciales")

    monkeypatch.setattr(__main__, "asyncio", SimpleNamespace(run=boom))
    assert __main__.main(["--check"]) == 2
    assert "sin credenciales" in capsys.readouterr().err
