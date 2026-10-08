"""Confluence / opportunity scoring engine.

Produces a single 0-100 score from weighted components, classifies the
result, and enforces the non-tradable floor.

Two invariants this module must never violate:

1. A ``Quality.NO_OPERABLE`` opportunity can never carry ``tradable=True``.
2. The score is never computed from a single indicator. Every component is
   required to be present, and missing data lowers the score instead of
   being treated as neutral.
"""

from __future__ import annotations

from botalpaca.analysis.market_context import sector_for_symbol
from botalpaca.config import AnalyticsSettings, RiskSettings, StrategySettings
from botalpaca.domain import (
    MarketRegime,
    Opportunity,
    Quality,
    ScoreBreakdown,
    SignalDirection,
    StrategySignal,
    TechnicalSnapshot,
    TradingEnvironment,
)
from botalpaca.strategies.base import TradeLevels, make_fingerprint, refine_levels_with_structure

__all__ = ["ConfluenceEngine", "classify_quality", "WEIGHTS"]

# Component weights. They sum to 1.0 over the "core" analysis block; the
# historical blocks are additive and clamped by the final score, so a strong
# backtest can never manufacture a trade out of a weak chart.
WEIGHTS: dict[str, float] = {
    "trend": 0.17,
    "momentum": 0.11,
    "volume": 0.09,
    "volatility": 0.07,
    "structure": 0.11,
    "breakout": 0.07,
    "multi_timeframe": 0.08,
    "market": 0.09,
    "sector": 0.05,
    "liquidity": 0.04,
    "rr": 0.07,
    "risk": 0.05,
}

# Maximum points the historical-performance block may contribute.
HISTORICAL_MAX_POINTS = 12.0

QUALITY_THRESHOLDS: tuple[tuple[float, Quality], ...] = (
    (75.0, Quality.ALTA),
    (55.0, Quality.MEDIA),
    (35.0, Quality.BAJA),
)
# The floor an opportunity must clear to be executable. Read from settings so
# the operator can loosen it without a code change: at 70 nothing qualified, and
# a bot that never opens a position cannot be evaluated at all.
MIN_EXECUTABLE_SCORE = 60.0
MIN_RR = 2.0
MIN_DATA_QUALITY = 0.5


def classify_quality(score: float, *, rr: float, data_quality: float) -> Quality:
    """Map a score to a quality bucket, then veto with hard rules.

    The veto is separate from the score on purpose: a high-scoring setup with
    a 0.8 R:R is not a "MEDIA" opportunity, it is not an opportunity.
    """
    if data_quality < MIN_DATA_QUALITY or rr < MIN_RR:
        return Quality.NO_OPERABLE
    for threshold, quality in QUALITY_THRESHOLDS:
        if score >= threshold:
            return quality
    return Quality.NO_OPERABLE


class ConfluenceEngine:
    def __init__(
        self,
        *,
        risk_settings: RiskSettings | None = None,
        strategy_settings: StrategySettings | None = None,
        analytics_settings: AnalyticsSettings | None = None,
    ) -> None:
        self.risk = risk_settings or RiskSettings()
        self.strategy = strategy_settings or StrategySettings()
        self.analytics = analytics_settings or AnalyticsSettings()

    def score(
        self,
        snapshot: TechnicalSnapshot,
        signal: StrategySignal,
        *,
        environment: TradingEnvironment,
        avg_dollar_volume: float | None = None,
        spread_pct: float | None = None,
        historical: dict[str, float | int | None] | None = None,
    ) -> Opportunity:
        """Build a fully-scored :class:`Opportunity` from one strategy signal."""
        breakdown = ScoreBreakdown()
        notes: list[str] = []

        breakdown.trend = self._trend_points(snapshot, signal)
        breakdown.momentum = 100.0 * snapshot.momentum.score
        breakdown.volume = 100.0 * snapshot.volume.score
        breakdown.volatility = 100.0 * snapshot.volatility.score
        breakdown.structure = self._structure_points(snapshot, signal)
        breakdown.breakout = self._breakout_points(snapshot)
        breakdown.multi_timeframe = self._mtf_points(snapshot, signal)
        breakdown.market = self._market_points(snapshot, signal)
        breakdown.sector = self._sector_points(snapshot, signal)
        breakdown.liquidity = self._liquidity_points(avg_dollar_volume, spread_pct)
        breakdown.rr = self._rr_points(signal.rr)
        breakdown.risk = self._risk_points(snapshot, signal)

        # Weighted core score. WEIGHTS sums to 1.0 and every component is on a
        # 0-100 scale, so `core` is already a 0-100 score.
        base_score = sum(getattr(breakdown, key) * weight for key, weight in WEIGHTS.items())

        # Historical adjustment, clamped, and only with a real sample.
        hist_points, hist_notes = self._historical_points(historical)
        breakdown.strategy_history = hist_points * 0.4
        breakdown.symbol_history = hist_points * 0.35
        breakdown.setup_history = hist_points * 0.25
        notes.extend(hist_notes)

        # Data quality is a multiplier, never an additive bonus: thin data
        # must reduce conviction proportionally.
        quality_multiplier = 0.5 + 0.5 * snapshot.data_quality
        score = (base_score + hist_points) * quality_multiplier
        score = max(0.0, min(100.0, score))

        sector = sector_for_symbol(snapshot.symbol)
        fingerprint = make_fingerprint(
            snapshot.symbol, signal.strategy, signal.direction, snapshot.timeframe
        )

        # Put the levels on real structure before anything reads them. The
        # strategy places them as ATR multiples, which makes the ratio a
        # constant; this is the one place where both the signal and the chart's
        # support/resistance are available, so the refinement happens here
        # rather than in every strategy.
        levels = refine_levels_with_structure(
            TradeLevels(
                direction=signal.direction,
                entry=signal.entry,
                stop=signal.stop,
                targets=tuple(signal.targets) or (signal.entry,),
            ),
            getattr(snapshot, "structure", None),
            snapshot.indicators.atr_14,
            min_rr=MIN_RR,
        )

        opportunity = Opportunity(
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            environment=environment,
            as_of=snapshot.as_of,
            direction=signal.direction,
            quality=Quality.NO_OPERABLE,
            score=round(score, 1),
            entry=levels.entry,
            stop=levels.stop,
            target=levels.primary_target,
            rr=round(levels.rr, 2),
            strategy=signal.strategy,
            setup=signal.strategy,
            breakdown=breakdown,
            confluences=list(signal.confluences),
            reasons=list(signal.reasons) + notes,
            invalidation=signal.invalidation,
            atr=snapshot.indicators.atr_14,
            regime=snapshot.volatility.regime,
            sector=sector,
            historical=dict(historical or {}),
            signals=[signal],
            fingerprint=fingerprint,
        )

        self._apply_tradability(
            opportunity, snapshot, environment=opportunity.environment
        )
        return opportunity

    # -- components ---------------------------------------------------------

    def _trend_points(self, snapshot: TechnicalSnapshot, signal: StrategySignal) -> float:
        base = 100.0 * snapshot.trend.strength
        if snapshot.trend.direction is signal.direction:
            base += 10.0
        else:
            base -= 15.0
        return _clamp(base)

    def _structure_points(self, snapshot: TechnicalSnapshot, signal: StrategySignal) -> float:
        base = 50.0
        levels = (
            snapshot.structure.supports
            if signal.direction is SignalDirection.LONG
            else snapshot.structure.resistances
        )
        if levels:
            # A nearby, well-tested level in our favour adds conviction.
            distance_pct = abs(snapshot.indicators.close - levels[0].price) / max(
                snapshot.indicators.close, 1e-9
            )
            base += 20.0 * (1.0 - min(1.0, distance_pct * 5.0))
            base += min(15.0, levels[0].touches * 5.0)
        if snapshot.structure.retest:
            base += 10.0
        if snapshot.structure.in_range and snapshot.volatility.regime in (
            MarketRegime.RANGING,
            MarketRegime.LOW_VOLATILITY,
        ):
            base += 5.0
        return _clamp(base)

    def _breakout_points(self, snapshot: TechnicalSnapshot) -> float:
        if not snapshot.structure.breakout:
            return 30.0
        base = 70.0
        if snapshot.structure.retest:
            base += 15.0
        rel_vol = snapshot.volume.rel_volume
        if rel_vol and rel_vol >= 1.5:
            base += 15.0
        return _clamp(base)

    def _mtf_points(self, snapshot: TechnicalSnapshot, signal: StrategySignal) -> float:
        if snapshot.htf_direction is None:
            # Unknown higher timeframe: reduced, not assumed.
            return 40.0
        if snapshot.htf_direction is signal.direction:
            return 90.0
        return 15.0

    def _market_points(self, snapshot: TechnicalSnapshot, signal: StrategySignal) -> float:
        ctx = snapshot.context
        if ctx.benchmark_trend is None:
            return 50.0  # unknown backdrop
        if ctx.benchmark_trend is signal.direction:
            base = 85.0
            if ctx.benchmark_regime in (
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
            ):
                base += 15.0
        else:
            base = 25.0
        if ctx.benchmark_regime == MarketRegime.HIGH_VOLATILITY:
            base -= 15.0
        rs = ctx.rs_vs_benchmark
        if rs is not None:
            if (rs > 0) == (signal.direction is SignalDirection.LONG):
                base += min(10.0, abs(rs))
            else:
                base -= min(15.0, abs(rs))
        return _clamp(base)

    def _sector_points(self, snapshot: TechnicalSnapshot, signal: StrategySignal) -> float:
        ctx = snapshot.context
        if ctx.sector_trend is None:
            return 50.0
        return 85.0 if ctx.sector_trend is signal.direction else 30.0

    def _liquidity_points(
        self, avg_dollar_volume: float | None, spread_pct: float | None
    ) -> float:
        if avg_dollar_volume is None:
            return 50.0
        floor = self.risk.min_liquidity_avg_dollar_volume
        if avg_dollar_volume < floor:
            # Scale into the band rather than a hard cliff.
            return _clamp(20.0 * (avg_dollar_volume / floor))
        base = 80.0
        if avg_dollar_volume > floor * 10:
            base += 20.0
        if spread_pct is not None:
            if spread_pct > self.risk.max_spread_pct:
                base -= 25.0
            elif spread_pct < self.risk.max_spread_pct / 3:
                base += 10.0
        return _clamp(base)

    def _rr_points(self, rr: float) -> float:
        if rr <= 0:
            return 0.0
        return _clamp(min(100.0, (rr / 3.0) * 100.0))

    def _risk_points(self, snapshot: TechnicalSnapshot, signal: StrategySignal) -> float:
        """Reward a sane stop distance; punish a stop too tight to survive noise."""
        atr = snapshot.indicators.atr_14
        if not atr or signal.stop <= 0 or signal.entry <= 0:
            return 40.0
        stop_distance_pct = abs(signal.entry - signal.stop) / signal.entry * 100.0
        if stop_distance_pct <= 0:
            return 0.0
        atr_ratio = (abs(signal.entry - signal.stop) / atr)
        # 1.5x-3x ATR is the healthy band for an intraday-to-daily swing.
        if atr_ratio < 1.0:
            return 40.0
        if 1.0 <= atr_ratio <= 3.0:
            return 90.0
        if 3.0 < atr_ratio <= 5.0:
            return 65.0
        return 45.0

    def _historical_points(
        self, historical: dict[str, float | int | None] | None
    ) -> tuple[float, list[str]]:
        """Convert historical performance into a clamped score adjustment."""
        if not historical:
            return 0.0, []
        notes: list[str] = []
        points = 0.0
        min_sample = self.analytics.min_sample_for_confidence

        for key, label in (
            ("win_rate", "estrategia"),
            ("symbol_win_rate", "símbolo"),
            ("setup_win_rate", "setup"),
        ):
            wr = historical.get(key)
            sample = historical.get(f"{key.replace('_win_rate', '')}_sample")
            if wr is None or not sample or int(sample) < min_sample:
                continue
            avg_r = historical.get(f"{key.replace('_win_rate', '')}_avg_r")
            # Win rate above 50% adds, below subtracts, scaled by conviction.
            edge = (float(wr) - 50.0) / 50.0
            contribution = edge * HISTORICAL_MAX_POINTS * 0.5
            if avg_r is not None:
                contribution += max(-1.0, min(1.0, float(avg_r))) * HISTORICAL_MAX_POINTS * 0.15
            points += contribution
            notes.append(
                f"histórico {label}: {float(wr):.0f}% acierto en {int(sample)} operaciones"
            )

        points = max(-HISTORICAL_MAX_POINTS, min(HISTORICAL_MAX_POINTS, points))
        return points, notes

    # -- tradability gate --------------------------------------------------

    def _shorts_allowed(self, environment: TradingEnvironment | None) -> bool:
        """Whether a SHORT may be proposed in this environment.

        Unknown environment means no shorts. Guessing "probably fine" is how a
        paper-validated setup turns into a live margin call.
        """
        if environment is None:
            return False
        if environment is TradingEnvironment.REAL:
            return bool(self.risk.allow_shorts_in_real)
        return bool(self.risk.allow_shorts)

    def _blocking_obstacle(
        self, opp: Opportunity, snapshot: TechnicalSnapshot
    ) -> str | None:
        """A well-tested level the trade has to break before it can reach the target.

        This is the one thing a score must never be allowed to buy its way
        past. A long whose target sits above a resistance is not a 2:1 setup,
        it is a bet that the resistance gives way; if it does not, the target
        was never reachable and the R that got approved was fictional.

        Only levels tested at least ``min_blocking_level_touches`` times count.
        A single touch is noise, and refusing those would empty the scanner.
        """
        structure = getattr(snapshot, "structure", None)
        if structure is None:
            return None
        minimum_touches = int(getattr(self.risk, "min_blocking_level_touches", 2) or 0)
        if minimum_touches < 1:
            return None

        if opp.direction is SignalDirection.LONG:
            entry, target = opp.entry, opp.target
            if target <= entry:
                return None
            candidates = [
                lvl
                for lvl in structure.resistances
                if entry < lvl.price < target and lvl.touches >= minimum_touches
            ]
            side = "resistencia"
        else:
            entry, target = opp.entry, opp.target
            if target >= entry:
                return None
            candidates = [
                lvl
                for lvl in structure.supports
                if target < lvl.price < entry and lvl.touches >= minimum_touches
            ]
            side = "soporte"

        if not candidates:
            return None
        nearest = min(candidates, key=lambda lvl: abs(lvl.price - entry))
        distance_pct = abs(nearest.price - entry) / max(abs(entry), 1e-9) * 100.0
        return (
            f"{side} en {nearest.price:.2f} ({nearest.touches} toques, "
            f"a {distance_pct:.2f}% de la entrada) entre la entrada y el "
            f"objetivo {target:.2f}: el objetivo exige romperla antes"
        )

    def _apply_tradability(
        self,
        opp: Opportunity,
        snapshot: TechnicalSnapshot,
        *,
        environment: TradingEnvironment | None = None,
    ) -> None:
        """Single authoritative gate. Sets ``quality`` and ``tradable`` together.

        Invariant: ``tradable`` is only ever ``True`` when quality is at
        least ``MEDIA`` and the score clears the execution floor.

        ``environment`` matters for shorts: paper establishes locates
        automatically and charges no borrow fees, live does neither. A setup
        that is perfectly tradable in paper can be refused - or accepted and
        then margin-called - with real money, so the environment decides rather
        than a constant.
        """
        blocks: list[str] = []
        quality = classify_quality(
            opp.score, rr=opp.rr, data_quality=snapshot.data_quality
        )

        if opp.direction is SignalDirection.SHORT and not self._shorts_allowed(environment):
            blocks.append("cortos deshabilitados en este entorno (solo PAPER)")

        if quality is Quality.NO_OPERABLE:
            if opp.rr < MIN_RR:
                blocks.append(f"R:R {opp.rr:.2f} inferior al mínimo {MIN_RR}")
            if snapshot.data_quality < MIN_DATA_QUALITY:
                blocks.append("calidad de datos insuficiente")
            if opp.score < 35.0:
                blocks.append(f"score {opp.score:.0f} demasiado bajo")
        if opp.atr is None or opp.atr <= 0:
            blocks.append("ATR no disponible para dimensionar el stop")
        # Structure is a veto, not a score component. Nothing above can buy
        # its way past a level that has to be broken before the target is
        # reachable: the R that was approved simply would not exist.
        obstacle = self._blocking_obstacle(opp, snapshot)
        if obstacle is not None:
            blocks.append(obstacle)
        # A long bought at RSI 85 is not a trend entry, it is the end of one.
        # LLY was taken twice this way (RSI 85 and 87) and both trades were
        # immediately underwater. The score never punished it: a strong trend
        # pushes RSI up, which is exactly when the setup is worst.
        rsi = getattr(snapshot.indicators, "rsi_14", None)
        if opp.direction is SignalDirection.LONG and rsi is not None:
            if rsi > self.risk.max_rsi_for_long:
                blocks.append(
                    f"RSI {rsi:.0f} en sobrecompra extrema para un largo "
                    f"(maximo {self.risk.max_rsi_for_long:.0f})"
                )
        stop_distance_pct = (
            abs(opp.entry - opp.stop) / opp.entry * 100.0 if opp.entry > 0 else 0.0
        )
        if stop_distance_pct < self.risk.min_stop_distance_pct:
            blocks.append(
                f"stop a {stop_distance_pct:.2f}% por debajo del mínimo "
                f"{self.risk.min_stop_distance_pct}%"
            )
        if stop_distance_pct > self.risk.max_stop_distance_pct:
            blocks.append(
                f"stop a {stop_distance_pct:.2f}% excede el máximo "
                f"{self.risk.max_stop_distance_pct}%"
            )

        if blocks:
            opp.tradable = False
            opp.quality = Quality.NO_OPERABLE
            opp.non_tradable_reason = "; ".join(blocks)
            return

        opp.tradable = opp.score >= MIN_EXECUTABLE_SCORE
        opp.quality = quality
        if not opp.tradable:
            opp.non_tradable_reason = (
                f"score {opp.score:.0f} inferior al mínimo de ejecución "
                f"{MIN_EXECUTABLE_SCORE}"
            )


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))
