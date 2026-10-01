"""Centralized configuration.

All secrets are read from environment variables only. Nothing in this module
ever logs a secret value, and :meth:`AlpacaEnvironmentConfig.safe_summary`
is what gets used in logs and /status output.
"""

from __future__ import annotations

import functools
import json
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from botalpaca.domain import ConfigurationError, SecretNotConfiguredError, TradingEnvironment


class AlpacaEnvironmentConfig(BaseSettings):
    """Credentials and endpoints for exactly one Alpaca account.

    PAPER and LIVE instances are constructed independently and never share
    mutable state.
    """

    model_config = SettingsConfigDict(
        env_prefix="ALPACA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: TradingEnvironment = TradingEnvironment.PAPER
    api_key: SecretStr = Field(default=SecretStr(""))
    secret_key: SecretStr = Field(default=SecretStr(""))
    base_url: str = ""
    data_url: str = ""

    @field_validator("api_key", "secret_key", mode="before")
    @classmethod
    def _blank_secret(cls, value: object) -> object:
        """An explicitly empty/absent secret becomes an empty SecretStr, never None."""
        if value is None:
            return SecretStr("")
        return value

    @property
    def is_configured(self) -> bool:
        return bool(self._secret_value(self.api_key)) and bool(
            self._secret_value(self.secret_key)
        )

    @staticmethod
    def _secret_value(value: SecretStr | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return str(value)

    def require_configured(self) -> None:
        if not self.is_configured:
            raise SecretNotConfiguredError(
                f"Alpaca {self.environment.value} credentials are not configured. "
                f"Set the ALPACA_{self.environment.value}_API_KEY / _SECRET_KEY variables."
            )

    def safe_summary(self) -> dict[str, object]:
        """Redacted representation safe for logs and Telegram responses."""
        return {
            "environment": self.environment.value,
            "api_key_set": bool(self._secret_value(self.api_key)),
            "secret_key_set": bool(self._secret_value(self.secret_key)),
            "base_url": self.base_url,
            "data_url": self.data_url,
        }


class RiskSettings(BaseSettings):
    """Limits enforced by the Risk Engine. No magic numbers in code."""

    model_config = SettingsConfigDict(env_prefix="RISK_", env_file=".env", extra="ignore")

    max_risk_per_trade_pct: float = Field(default=1.0, gt=0, le=100)
    max_total_exposure_pct: float = Field(default=100.0, gt=0, le=500)
    max_sector_exposure_pct: float = Field(default=40.0, gt=0, le=100)
    max_correlated_exposure_pct: float = Field(default=60.0, gt=0, le=100)
    max_open_positions: int = Field(default=10, gt=0, le=100)
    max_daily_loss_pct: float = Field(default=3.0, gt=0, le=100)
    max_weekly_loss_pct: float = Field(default=6.0, gt=0, le=100)
    max_monthly_loss_pct: float = Field(default=10.0, gt=0, le=100)
    max_drawdown_pct: float = Field(default=15.0, gt=0, le=100)
    max_concentration_pct: float = Field(default=25.0, gt=0, le=100)
    min_stop_distance_pct: float = Field(default=0.3, ge=0, le=50)
    max_stop_distance_pct: float = Field(default=25.0, gt=0, le=100)
    min_order_notional: float = Field(default=50.0, ge=0)
    max_order_notional: float = Field(default=50_000.0, gt=0)
    max_slippage_estimate_bps: float = Field(default=30.0, ge=0)
    correlation_threshold: float = Field(default=0.7, gt=0, lt=1)
    min_rr: float = Field(default=1.5, ge=0)
    min_liquidity_avg_dollar_volume: float = Field(default=2_000_000.0, ge=0)
    max_spread_pct: float = Field(default=0.5, ge=0)


class StrategySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STRATEGY_", env_file=".env", extra="ignore"
    )

    atr_period: int = Field(default=14, gt=1)
    atr_stop_multiplier: float = Field(default=2.0, gt=0)
    atr_target_multiplier: float = Field(default=3.0, gt=0)
    min_bars_required: int = Field(default=60, gt=0)
    default_timeframe: str = "1D"
    higher_timeframe: str = "1W"
    lower_timeframe: str = "1H"
    max_signals_per_scan: int = Field(default=10, gt=0)
    min_score_to_report: float = Field(default=40.0, ge=0, le=100)
    min_score_to_execute: float = Field(default=70.0, ge=0, le=100)
    enable_weak_strategies: bool = True


class ProtectionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROTECTION_", env_file=".env", extra="ignore"
    )

    default_trailing_percent: float = Field(default=2.0, gt=0)
    break_even_trigger_r: float = Field(default=1.0, gt=0)
    break_even_buffer_pct: float = Field(default=0.15, ge=0)
    atr_trailing_multiplier: float = Field(default=2.5, gt=0)
    progressive_stops: bool = True
    time_stop_minutes: int = Field(default=0, ge=0)  # 0 disables
    default_time_stop_minutes: int = Field(default=720, ge=0)
    momentum_exit_enabled: bool = True
    auto_protect_missing_stop: bool = True
    fallback_stop_atr_multiplier: float = Field(default=2.0, gt=0)
    # Used only when no ATR is available (e.g. right after a restart, before any
    # bar history is loaded). Deliberately wide: a naked position is worse than
    # a wide stop.
    fallback_stop_pct: float = Field(default=3.0, gt=0, le=25)


class MonitoringSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MONITORING_", env_file=".env", extra="ignore"
    )

    auto_trading_enabled: bool = False  # signals never auto-execute by default
    position_alert_cooldown_seconds: int = Field(default=900, ge=0)
    signal_alert_cooldown_seconds: int = Field(default=1800, ge=0)
    opportunity_alert_min_score: float = Field(default=75.0, ge=0, le=100)
    max_alerts_per_hour: int = Field(default=20, gt=0)
    score_drop_alert_threshold: float = Field(default=20.0, gt=0)
    momentum_score_threshold: float = Field(default=50.0, ge=0, le=100)
    # When false, a score drop never produces an alert at all (not even as an
    # informational one), so monitoring can be silenced without touching exits.
    momentum_alerts_enabled: bool = True


class AnalyticsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANALYTICS_", env_file=".env", extra="ignore"
    )

    min_sample_for_significance: int = Field(default=20, gt=0)
    min_sample_for_sharpe: int = Field(default=30, gt=0)
    min_sample_for_confidence: int = Field(default=10, gt=0)
    confidence_level: float = Field(default=95.0, gt=50, le=99.99)
    historical_weight_max: float = Field(default=0.15, ge=0, le=0.5)


class ScannerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCANNER_", env_file=".env", extra="ignore")

    universe: list[str] = Field(
        default_factory=lambda: [
            "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
            "XLI", "XLB", "XLU", "XLRE", "XLC", "SMH", "SOXX", "ARKK", "AMD", "NVDA",
            "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "AVGO", "CRM",
            "ORCL", "ADBE", "INTC", "QCOM", "MU", "PLTR", "COIN", "UBER", "ABNB",
            "JPM", "BAC", "GS", "WFC", "V", "MA", "UNH", "LLY", "PFE", "MRK", "TMO",
            "XOM", "CVX", "COP", "SLB", "CAT", "DE", "GE", "BA", "LMT", "RTX", "WMT",
            "COST", "HD", "NKE", "SBUX", "MCD", "DIS", "KO", "PEP", "PG", "CRM", "DOCU",
        ]
    )
    benchmark: str = "SPY"
    max_concurrent_analyses: int = Field(default=8, gt=0)
    signal_dedup_window_minutes: int = Field(default=180, ge=0)
    max_fingerprint_history: int = Field(default=500, ge=0)


class Settings(BaseSettings):
    """Application settings root."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_name: str = "botalpaca"
    environment_name: str = "production"

    telegram_bot_token: SecretStr = Field(default=SecretStr(""))
    # ``NoDecode`` stops pydantic-settings from JSON-decoding the raw env value, so
    # the plain CSV form documented in .env.example works as well as a JSON list.
    telegram_allowed_user_ids: Annotated[list[int], NoDecode] = Field(default_factory=list)
    telegram_rate_limit_per_minute: int = Field(default=20, gt=0)

    active_trading_environment: TradingEnvironment = TradingEnvironment.PAPER

    database_url: str = "sqlite+aiosqlite:///./data/botalpaca.db"
    database_echo: bool = False
    database_migrate: bool = True
    database_auto_create: bool = True

    timezone: str = "America/New_York"
    log_level: str = "INFO"
    log_json: bool = True

    scan_interval_seconds: int = Field(default=300, gt=0)
    monitor_interval_seconds: int = Field(default=60, gt=0)
    reconcile_interval_seconds: int = Field(default=300, gt=0)
    session_recovery_enabled: bool = True

    api_timeout_seconds: float = Field(default=20.0, gt=0)
    api_max_retries: int = Field(default=3, ge=0)
    api_retry_backoff_seconds: float = Field(default=1.0, ge=0)

    @field_validator("telegram_allowed_user_ids", mode="before")
    @classmethod
    def _parse_user_ids(cls, v: object) -> object:
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return []
            if raw.startswith("["):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "TELEGRAM_ALLOWED_USER_IDS must be a CSV of integers or a JSON list"
                    ) from exc
            parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError as exc:
                raise ValueError(
                    f"TELEGRAM_ALLOWED_USER_IDS contains a non-integer value: {raw!r}"
                ) from exc
        return v

    @field_validator("telegram_bot_token", mode="before")
    @classmethod
    def _strip_token(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @model_validator(mode="after")
    def _validate_security(self) -> Settings:
        # PAPER is the default and must stay the default unless explicitly set.
        if self.active_trading_environment is not TradingEnvironment.REAL and (
            "ACTIVE_TRADING_ENVIRONMENT" not in _raw_env()
        ):
            object.__setattr__(self, "active_trading_environment", TradingEnvironment.PAPER)
        return self

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token.get_secret_value())

    # Sub-settings groups. Services default to ``get_settings().<group>`` when no
    # explicit object is injected, so these accessors must exist on the root.
    @property
    def risk(self) -> RiskSettings:
        return RiskSettings()

    @property
    def strategy(self) -> StrategySettings:
        return StrategySettings()

    @property
    def protection(self) -> ProtectionSettings:
        return ProtectionSettings()

    @property
    def monitoring(self) -> MonitoringSettings:
        return MonitoringSettings()

    @property
    def analytics(self) -> AnalyticsSettings:
        return AnalyticsSettings()

    @property
    def scanner(self) -> ScannerSettings:
        return ScannerSettings()

    def alpaca(self, environment: TradingEnvironment) -> AlpacaEnvironmentConfig:
        """Build the config object for one specific environment."""
        if environment is TradingEnvironment.PAPER:
            return AlpacaEnvironmentConfig(
                environment=TradingEnvironment.PAPER,
                api_key=SecretStr(_env("ALPACA_PAPER_API_KEY", "")),
                secret_key=SecretStr(_env("ALPACA_PAPER_SECRET_KEY", "")),
                base_url=_env("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets"),
                data_url=_env("ALPACA_PAPER_DATA_URL", "https://data.alpaca.markets"),
            )
        return AlpacaEnvironmentConfig(
            environment=TradingEnvironment.REAL,
            api_key=SecretStr(_env("ALPACA_LIVE_API_KEY", "")),
            secret_key=SecretStr(_env("ALPACA_LIVE_SECRET_KEY", "")),
            base_url=_env("ALPACA_LIVE_BASE_URL", "https://api.alpaca.markets"),
            data_url=_env("ALPACA_LIVE_DATA_URL", "https://data.alpaca.markets"),
        )

    def validate_runtime(self) -> None:
        """Fail fast on misconfiguration that would otherwise surface mid-trade."""
        if not self.telegram_enabled:
            raise ConfigurationError("TELEGRAM_BOT_TOKEN is not set.")
        if not self.telegram_allowed_user_ids:
            raise ConfigurationError(
                "TELEGRAM_ALLOWED_USER_IDS is empty. The bot refuses to start with an "
                "empty allowlist."
            )


def _env(name: str, default: str = "") -> str:
    import os

    return os.environ.get(name, default)


def _raw_env() -> dict[str, str]:
    import os

    return dict(os.environ)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Use :func:`reset_settings` in tests."""
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()


LogFormat = Literal["json", "console"]

__all__ = [
    "AlpacaEnvironmentConfig",
    "AnalyticsSettings",
    "MonitoringSettings",
    "ProtectionSettings",
    "RiskSettings",
    "ScannerSettings",
    "Settings",
    "StrategySettings",
    "get_settings",
    "reset_settings",
]
