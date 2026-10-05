"""An ACTIVE account must never be read as inactive.

The operator pressed ACEPTAR on every opportunity and got:

    Operacion no permitida
    XLV: Estado de cuenta no activo: AccountStatus.ACTIVE

The message rejects the very status it prints. ``str(AccountStatus.ACTIVE)`` is
``"AccountStatus.ACTIVE"`` on Python 3.11+, so comparing ``str(status).upper()``
against ``{"ACTIVE", ""}`` fails for every account on earth -- including healthy
ones. Every entry was blocked, permanently, with no log line and no test.

These tests use the real Alpaca enum rather than a stand-in string, which is
what let this survive: a fake carrying the plain text "ACTIVE" passes the old
check and proves nothing.
"""

from __future__ import annotations

import pytest
from alpaca.trading.enums import AccountStatus

from botalpaca.risk.engine import _status_is_active


def test_the_real_alpaca_active_enum_is_recognised():
    assert _status_is_active(AccountStatus.ACTIVE) is True


def test_the_old_comparison_rejected_the_healthy_account():
    """The exact line that shipped, kept so it cannot come back."""
    status = AccountStatus.ACTIVE
    assert str(status).upper() not in {"ACTIVE", ""}
    assert str(status).upper() == "ACCOUNTSTATUS.ACTIVE"


@pytest.mark.parametrize(
    "status",
    [AccountStatus.ACTIVE, "ACTIVE", "active", " Active ", "AccountStatus.ACTIVE"],
)
def test_anything_meaning_active_is_accepted(status):
    assert _status_is_active(status) is True


@pytest.mark.parametrize(
    "status",
    [AccountStatus.INACTIVE, AccountStatus.REJECTED, AccountStatus.DISABLED,
     AccountStatus.ACTION_REQUIRED, "inactive", "BLOCKED"],
)
def test_anything_else_is_rejected(status):
    assert _status_is_active(status) is False


def test_a_missing_status_is_not_a_blocked_account():
    """An absent status must not invent a reason to refuse."""
    assert _status_is_active(None) is True


def test_the_rejection_message_shows_the_value_not_the_repr():
    """So the operator never again reads "no activo: ACTIVE"."""
    status = AccountStatus.ACTIVE
    value = getattr(status, "value", status)
    assert str(value) == "ACTIVE"
    assert str(status) != "ACTIVE"
