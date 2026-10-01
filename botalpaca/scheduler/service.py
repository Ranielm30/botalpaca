"""Layer 19 — Scheduler and background workers.

The scheduler owns every periodic job: market scanning, position monitoring,
reconciliation and daily statistics. Each job runs the loop of the monitor bound
to the *currently active* environment, so a ``/modo`` switch changes what the
workers act on without a process restart.
"""

from __future__ import annotations

import asyncio
import datetime as dt

from botalpaca.config.logging import get_logger
from botalpaca.config.settings import Settings
from botalpaca.db import AppStateRepository
from botalpaca.monitoring import LAST_RECONCILE_KEY

log = get_logger(__name__)


class Scheduler:
    """Runs the periodic jobs as supervised asyncio tasks."""

    def __init__(self, app: object) -> None:
        self.app = app
        self.settings: Settings = app.settings
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Bind the workers to the active environment's monitors."""
        if self._running:
            return
        self._running = True
        self._stopping.clear()
        context = self.app.active
        self._tasks = [
            asyncio.create_task(
                context.market_monitor.loop(self.settings.scan_interval_seconds),
                name="market-monitor",
            ),
            asyncio.create_task(
                context.position_monitor.loop(self.settings.monitor_interval_seconds),
                name="position-monitor",
            ),
            asyncio.create_task(self._reconcile_loop(), name="reconciliation"),
        ]
        log.info(
            "scheduler.started",
            environment=self.app.active_environment.value,
            tasks=[t.get_name() for t in self._tasks],
            scan_interval=self.settings.scan_interval_seconds,
            monitor_interval=self.settings.monitor_interval_seconds,
            reconcile_interval=self.settings.reconcile_interval_seconds,
        )

    async def stop(self) -> None:
        self._stopping.set()
        for context in getattr(self.app, "contexts", {}).values():
            context.market_monitor.stop()
            context.position_monitor.stop()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._running = False
        log.info("scheduler.stopped")

    async def restart(self) -> None:
        """Re-bind the workers after an environment switch."""
        await self.stop()
        await self.start()

    async def run_pending(self) -> None:
        """One-shot drain, used on boot to recover state before serving."""
        await self.app.reconcile()
        await self._mark_reconciled()

    # ------------------------------------------------------------------- jobs

    async def _reconcile_loop(self) -> None:
        interval = self.settings.reconcile_interval_seconds
        while not self._stopping.is_set():
            await self._sleep(interval)
            if self._stopping.is_set():
                return
            try:
                notes = await self.app.reconcile()
                await self._mark_reconciled()
                if notes:
                    log.info("scheduler.reconciled", notes=notes)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a bad cycle must not kill the worker
                log.exception("scheduler.reconcile_failed")

    async def _mark_reconciled(self) -> None:
        async with self.app.database.session() as session:
            await AppStateRepository(session).set(
                LAST_RECONCILE_KEY, dt.datetime.now(dt.UTC).isoformat()
            )

    async def _sleep(self, seconds: int) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except TimeoutError:
            return


__all__ = ["Scheduler"]
