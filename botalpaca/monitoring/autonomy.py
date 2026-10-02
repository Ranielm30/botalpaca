"""Autonomous position management.

The operator asked for one job to be the bot's and one to be theirs:

* **Their job** is deciding whether to *enter*. Nothing here opens a position.
* **The bot's job** is protecting a position that is already open.

So every rule in this module only ever moves a stop up, arms a trailing stop, or
closes a position that has stopped making sense. None of it can increase
exposure, which is what makes running it unattended acceptable.

Each firing is reported through ``notify`` so the operator sees exactly what was
done, when, and why - a silent autonomous system is indistinguishable from a
broken one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

from botalpaca.config import ProtectionSettings
from botalpaca.config.logging import get_logger
from botalpaca.domain import PositionSnapshot, TradingEnvironment
from botalpaca.protection import PositionProtectionManager, r_multiple_of

log = get_logger(__name__)


class AutonomyAction(StrEnum):
    BREAK_EVEN = "break_even"
    PROGRESSIVE = "progressive"
    TRAILING = "trailing"
    TRAILING_WIDENED = "trailing_widened"
    TIME_STOP = "time_stop"
    MOMENTUM_EXIT = "momentum_exit"
    STOP_CREATED = "stop_created"


#: Actions the operator must always be told about, in one place.
ACTION_LABELS: dict[AutonomyAction, str] = {
    AutonomyAction.BREAK_EVEN: "Stop movido a break-even",
    AutonomyAction.PROGRESSIVE: "Stop subido (proteccion progresiva)",
    AutonomyAction.TRAILING: "Trailing stop activado",
    AutonomyAction.TRAILING_WIDENED: "Trailing stop ensanchado",
    AutonomyAction.TIME_STOP: "Cierre por time stop",
    AutonomyAction.MOMENTUM_EXIT: "Cierre por perdida de momentum",
    AutonomyAction.STOP_CREATED: "Stop de emergencia creado",
}


@dataclass(frozen=True)
class AutonomyReport:
    """One autonomous action that was actually applied."""

    action: AutonomyAction
    symbol: str
    environment: TradingEnvironment
    position: PositionSnapshot | None
    detail: str
    r_multiple: float | None = None
    new_stop: float | None = None
    trail_percent: float | None = None

    @property
    def label(self) -> str:
        return ACTION_LABELS.get(self.action, self.action.value)

    @property
    def is_profitable(self) -> bool | None:
        """None when unknown; used to pick the emoji in the notification."""
        if self.r_multiple is None:
            return None
        return self.r_multiple >= 0

    def render(self) -> str:
        """Human summary for Telegram."""
        badge = self.environment.badge
        arrow = "+" if (self.r_multiple or 0) >= 0 else ""
        lines = [f"{badge} <b>Accion autonoma · {self.label}</b>", f"Simbolo: {self.symbol}"]
        if self.r_multiple is not None:
            lines.append(f"R multiple: {arrow}{self.r_multiple:.2f}R")
        if self.new_stop is not None:
            lines.append(f"Nuevo stop: ${self.new_stop:,.2f}")
        if self.trail_percent is not None:
            lines.append(f"Trailing: {self.trail_percent:.2f}%")
        lines.append(f"Motivo: {self.detail}")
        return "\n".join(lines)


class AutonomousProtector:
    """Applies the configured protection rules to open positions, unattended."""

    def __init__(
        self,
        protection: PositionProtectionManager,
        *,
        environment: TradingEnvironment,
        settings: ProtectionSettings | None = None,
        notify: object | None = None,
    ) -> None:
        self._protection = protection
        self.environment = environment
        self.settings = settings or ProtectionSettings()
        self._notify = notify
        self._last_report: dict[str, dt.datetime] = {}

    # -- helpers -------------------------------------------------------------

    def _should_report(self, symbol: str) -> bool:
        cooldown = self.settings.autonomy_notify_cooldown_minutes
        if cooldown <= 0:
            return True
        now = dt.datetime.now(dt.UTC)
        last = self._last_report.get(symbol)
        if last is not None and (now - last) < dt.timedelta(minutes=cooldown):
            return False
        self._last_report[symbol] = now
        return True

    async def _send(self, report: AutonomyReport) -> None:
        log.info(
            "autonomy.applied",
            action=report.action.value,
            symbol=report.symbol,
            environment=self.environment.value,
            detail=report.detail,
        )
        if self._notify is None or not self._should_report(report.symbol):
            return
        try:
            await self._notify(report.render())
        except Exception:  # noqa: BLE001 - a failed notice must not stop protection
            log.warning("autonomy.notify_failed", symbol=report.symbol, exc_info=True)

    @staticmethod
    def _r_multiple(position: PositionSnapshot, stop_price: float | None) -> float | None:
        return r_multiple_of(position, stop_price)

    # -- rules ---------------------------------------------------------------

    async def evaluate(self, position: PositionSnapshot) -> list[AutonomyReport]:
        """Run every enabled rule for one position.

        Rules are ordered from cheapest and safest to most involved, and at most
        one state-changing rule fires per pass so the report stays readable.
        """
        reports: list[AutonomyReport] = []
        if position.qty == 0:
            return reports

        state = await self._protection.state_for_position(self.environment, position)
        # R is always measured against the risk the trade was opened with. Using
        # the live stop made a protected trade measure its own profit as risk,
        # and once a trailing stop replaced the fixed stop there was no stop
        # price left at all, so the engine fell back to comparing a percent move
        # against an R threshold and went permanently quiet.
        r_now = self._r_multiple(position, getattr(state, "initial_stop_price", None))

        # 1. A naked position is the worst state; fix that before anything else.
        if not state.has_stop and not state.has_trailing:
            reports.extend(await self._ensure_stop(position, state))
            return reports

        # 2. Time stop: the trade has had its window and has not delivered.
        if self.settings.auto_time_stop and await self._protection.is_time_stop_due(
            self.environment, position.symbol
        ):
            closed = await self._close(position, AutonomyAction.TIME_STOP, "tiempo maximo agotado")
            if closed is not None:
                reports.append(closed)
            return reports

        # 3. Once the trailing stop owns the position there is nothing left to
        # ratchet: it already follows the price, and a fixed stop on top of it
        # would fight it and cancel the very protection that is running.
        if state.has_trailing:
            widened = await self._widen_trailing_if_tight(position, state)
            if widened is not None:
                reports.append(widened)
            return reports

        # 4. Break-even, then progressive ratcheting while it keeps winning.
        if self.settings.auto_break_even and r_now is not None:
            if r_now >= self.settings.break_even_trigger_r and not (
                state.break_even_active and state.stop_price is not None
                and self._already_at_break_even(position, state.stop_price)
            ):
                moved = await self._break_even(position, state, r_now)
                if moved is not None:
                    reports.append(moved)
                    return reports

            # Below the trailing trigger the stop is ratcheted by hand. At or above it the
            # trade belongs to the trailing stop, so progressive must yield --
            # otherwise it fires on every pass, returns early, and the trailing
            # stop is never armed at all.
            if (
                self.settings.auto_progressive
                and r_now is not None
                and r_now <= self.settings.trailing_trigger_r
                and r_now > self.settings.break_even_trigger_r + 0.5
            ):
                stepped = await self._progressive(position, state, r_now)
                if stepped is not None:
                    reports.append(stepped)
                    return reports

        # 5. Arm the trailing stop once the trade is clearly in profit.
        if (
            self.settings.auto_trailing
            and not state.has_trailing
            and r_now is not None
            and r_now >= self.settings.trailing_trigger_r
        ):
            armed = await self._trailing(position, state, r_now)
            if armed is not None:
                reports.append(armed)

        return reports

    def _already_at_break_even(self, position: PositionSnapshot, stop: float | None) -> bool:
        if stop is None:
            return False
        buffer = self.settings.break_even_buffer_pct
        if position.qty >= 0:
            return stop >= position.avg_entry_price * (1 + buffer / 100.0)
        return stop <= position.avg_entry_price * (1 - buffer / 100.0)

    async def _ensure_stop(
        self, position: PositionSnapshot, state: object
    ) -> list[AutonomyReport]:
        try:
            updated = await self._protection.ensure_stop(
                environment=self.environment,
                position=position,
                reason="posicion desprotegida detectada por el monitor",
            )
        except Exception as exc:  # noqa: BLE001
            log.error("autonomy.ensure_stop_failed", symbol=position.symbol, error=str(exc))
            return []
        report = AutonomyReport(
            action=AutonomyAction.STOP_CREATED,
            symbol=position.symbol,
            environment=self.environment,
            position=position,
            detail="la posicion estaba abierta sin stop",
            new_stop=updated.stop_price,
        )
        await self._send(report)
        return [report]

    async def _break_even(
        self, position: PositionSnapshot, state: object, r_now: float
    ) -> AutonomyReport | None:
        try:
            updated = await self._protection.move_to_break_even(
                environment=self.environment, position=position
            )
        except Exception as exc:  # noqa: BLE001
            log.error("autonomy.break_even_failed", symbol=position.symbol, error=str(exc))
            return None
        if updated.stop_price is None or updated.stop_price == state.stop_price:
            return None
        report = AutonomyReport(
            action=AutonomyAction.BREAK_EVEN,
            symbol=position.symbol,
            environment=self.environment,
            position=position,
            detail=f"la operacion supero {self.settings.break_even_trigger_r:.1f}R",
            r_multiple=r_now,
            new_stop=updated.stop_price,
        )
        await self._send(report)
        return report

    async def _progressive(
        self, position: PositionSnapshot, state: object, r_now: float
    ) -> AutonomyReport | None:
        try:
            updated = await self._protection.progressive_step(
                environment=self.environment, position=position
            )
        except Exception as exc:  # noqa: BLE001
            log.error("autonomy.progressive_failed", symbol=position.symbol, error=str(exc))
            return None
        if updated is None or updated.stop_price is None:
            return None
        if updated.stop_price == state.stop_price:
            return None
        report = AutonomyReport(
            action=AutonomyAction.PROGRESSIVE,
            symbol=position.symbol,
            environment=self.environment,
            position=position,
            detail="la operacion sigue ganando, el stop sube para asegurar",
            r_multiple=r_now,
            new_stop=updated.stop_price,
        )
        await self._send(report)
        return report

    async def _trailing(
        self, position: PositionSnapshot, state: object, r_now: float
    ) -> AutonomyReport | None:
        # The trail must never be looser than what break-even already secured,
        # or arming it would quietly give back the protection that just landed.
        floor = self._break_even_floor(position)
        try:
            updated = await self._protection.enable_trailing_stop(
                environment=self.environment, position=position, floor_stop=floor
            )
        except Exception as exc:  # noqa: BLE001
            log.error("autonomy.trailing_failed", symbol=position.symbol, error=str(exc))
            return None
        percent = updated.trail_percent
        if not updated.has_trailing or percent is None:
            return None
        detail = f"la operacion supero {self.settings.trailing_trigger_r:.1f}R"
        if floor is not None:
            detail += "; el trailing mantiene la ganancia asegurada en break-even"
        report = AutonomyReport(
            action=AutonomyAction.TRAILING,
            symbol=position.symbol,
            environment=self.environment,
            position=position,
            detail=detail,
            r_multiple=r_now,
            trail_percent=percent,
            new_stop=floor,
        )
        await self._send(report)
        return report

    def _break_even_floor(self, position: PositionSnapshot) -> float | None:
        """The lowest price the trailing stop may sit at, or None if unconstrained."""
        entry = position.avg_entry_price
        if entry <= 0:
            return None
        buffer = self.settings.break_even_buffer_pct
        return (
            round(entry * (1 + buffer / 100.0), 2)
            if position.qty >= 0
            else round(entry * (1 - buffer / 100.0), 2)
        )

    async def _widen_trailing_if_tight(
        self, position: PositionSnapshot, state: object
    ) -> AutonomyReport | None:
        """A trailing stop squeezed against the price gets out on noise.

        Alpaca's trailing stop follows the price up but never recalculates its
        width, so a gap can leave it a few cents under the market. That is not
        protection, it is a coin flip, and it will take the trade out on noise
        instead of letting a real move develop.
        """
        percent = getattr(state, "trail_percent", None)
        price = getattr(state, "trail_price", None)
        current = position.current_price
        if percent is None or current <= 0:
            return None
        if position.qty >= 0:
            distance_pct = (current - price) / current * 100.0 if price else None
        else:
            distance_pct = (price - current) / current * 100.0 if price else None
        if distance_pct is None:
            return None
        # Half the trail width is the floor: below that the stop is inside the
        # noise band of a normal session.
        if distance_pct >= percent / 2.0:
            return None
        # Ask for genuinely more room. Requesting the width the trail already has
        # would resolve to the same percentage and quietly do nothing.
        wider = round(percent * 2.0, 2)
        try:
            updated = await self._protection.enable_trailing_stop(
                environment=self.environment, position=position, widen_to=wider
            )
        except Exception as exc:  # noqa: BLE001
            log.error("autonomy.trailing_widen_failed", symbol=position.symbol, error=str(exc))
            return None
        if not updated.has_trailing or updated.trail_percent == percent:
            return None
        report = AutonomyReport(
            action=AutonomyAction.TRAILING_WIDENED,
            symbol=position.symbol,
            environment=self.environment,
            position=position,
            detail=(
                f"el trailing se habia acercado a {distance_pct:.2f}% del precio; "
                f"se ensancha a {updated.trail_percent:.2f}% para no saltar con el ruido"
            ),
            r_multiple=self._r_multiple(position, getattr(state, "initial_stop_price", None)),
            trail_percent=updated.trail_percent,
        )
        await self._send(report)
        return report

    async def _close(
        self, position: PositionSnapshot, action: AutonomyAction, detail: str
    ) -> AutonomyReport | None:
        closer = getattr(self._protection, "_engine", None)
        if closer is None:
            return None
        try:
            await closer.close_position(
                position.symbol, environment=self.environment, confirmed=True
            )
        except Exception as exc:  # noqa: BLE001
            log.error("autonomy.close_failed", symbol=position.symbol, error=str(exc))
            return None
        report = AutonomyReport(
            action=action,
            symbol=position.symbol,
            environment=self.environment,
            position=position,
            detail=detail,
        )
        await self._send(report)
        return report


__all__ = [
    "ACTION_LABELS",
    "AutonomousProtector",
    "AutonomyAction",
    "AutonomyReport",
]
