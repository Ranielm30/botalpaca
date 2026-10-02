"""Layer 14 — Monitoring engine.

Two jobs run in the background on a schedule:

* :class:`MarketMonitor` looks for new opportunities and pushes them to Telegram.
* :class:`PositionMonitor` watches open positions, applies the unattended
  protection rules through :class:`AutonomousProtector`, and reports what it did.
"""

from botalpaca.monitoring.autonomy import (
    ACTION_LABELS,
    AutonomousProtector,
    AutonomyAction,
    AutonomyReport,
)
from botalpaca.monitoring.service import (
    AUTOMATIC_EXIT_REASONS,
    LAST_RECONCILE_KEY,
    LAST_SCAN_KEY,
    SCORE_PREFIX,
    MarketMonitor,
    PositionAlert,
    PositionMonitor,
)

__all__ = [
    "ACTION_LABELS",
    "AUTOMATIC_EXIT_REASONS",
    "LAST_RECONCILE_KEY",
    "LAST_SCAN_KEY",
    "SCORE_PREFIX",
    "AutonomousProtector",
    "AutonomyAction",
    "AutonomyReport",
    "MarketMonitor",
    "PositionAlert",
    "PositionMonitor",
]
