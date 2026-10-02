"""trades.filled_at

Orders submitted while the market is closed sit as ``accepted`` in Alpaca
without filling, so the bot has to remember that a trade row exists but no
position does yet.

A nullable ``filled_at`` distinguishes the two without touching ``status``:
``filled_at IS NULL`` means the entry is still pending, anything else means it
filled and is a real trade. Repurposing ``status`` instead would have removed
those rows from every open-position and exposure query, which is exactly where
they need to be visible from.

Revision ID: b81f2c60d4a7
Revises: f4e0ba9f112b
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b81f2c60d4a7"
down_revision: str | None = "f4e0ba9f112b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("trades", schema=None) as batch_op:
        batch_op.add_column(sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("trades", schema=None) as batch_op:
        batch_op.drop_column("filled_at")
