"""freeze the risk the trade was opened with

Revisions reviewed:
  * 6cf6a5e9c120 added the trade_events foreign key.
  * f4e0ba9f112b made signals.htf_timeframe nullable.
  * b81f2c60d4a7 added trades.filled_at.

Revision ID: c3d9e18a7f42
Revises: b81f2c60d4a7
Create Date: 2026-10-02

The R multiple a trade is judged by must stay the R it was opened with. It used
to be recomputed from whatever stop happened to be live, so a trade whose stop had
already been ratcheted to break-even measured itself against its own profit and
looked like it had risked far more than it had. Worse, once a trailing stop
replaced the fixed stop there was no stop price left at all, and the engine fell
back to comparing a percent move against an R threshold.

This column records the entry risk once, at registration, and never rewrites it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c3d9e18a7f42"
down_revision = "b81f2c60d4a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("position_protection") as batch:
        batch.add_column(sa.Column("initial_stop_price", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("position_protection") as batch:
        batch.drop_column("initial_stop_price")
