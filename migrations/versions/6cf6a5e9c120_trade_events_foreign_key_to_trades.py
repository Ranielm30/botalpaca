"""trade_events foreign key to trades

Adds the missing ``trade_events.trade_id -> trades.id`` foreign key (with
``ON DELETE CASCADE``) that the ORM relationship requires. SQLite cannot
``ALTER TABLE`` to add a constraint, so this uses Alembic's batch mode, which
recreates the table. The constraint is named so ``downgrade`` works too.

Revision ID: 6cf6a5e9c120
Revises: 96855598aa51
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "6cf6a5e9c120"
down_revision: str | None = "96855598aa51"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT_NAME = "fk_trade_events_trade_id"


def upgrade() -> None:
    with op.batch_alter_table("trade_events", schema=None) as batch_op:
        batch_op.create_foreign_key(
            CONSTRAINT_NAME, "trades", ["trade_id"], ["id"], ondelete="CASCADE"
        )


def downgrade() -> None:
    with op.batch_alter_table("trade_events", schema=None) as batch_op:
        batch_op.drop_constraint(CONSTRAINT_NAME, type_="foreignkey")
