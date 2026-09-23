"""rename users.oplist_user_id to identity_id, drop sessions.oplist_token

The server owns its accounts now, so the two columns that existed only to carry an
external service's identity are gone. ``identity_id`` keeps the
``"<provider>:<subject>"`` convention; existing rows are re-pointed at
``local:<name>`` the next time their owner logs in (or an admin recreates them).

Revision ID: 4f7c1d2ab9e3
Revises: 045be3078645
Create Date: 2026-09-22 20:10:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4f7c1d2ab9e3"
down_revision: Union[str, Sequence[str], None] = "045be3078645"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("sessions", schema=None) as batch_op:
        batch_op.drop_column("oplist_token")

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_oplist_user_id"))
        batch_op.alter_column(
            "oplist_user_id",
            new_column_name="identity_id",
            existing_type=sa.String(length=64),
            existing_nullable=False,
        )

    # the index is created in its own batch step: SQLite recreates the table, and
    # the column only exists under its new name once that step is done
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_users_identity_id"), ["identity_id"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_identity_id"))
        batch_op.alter_column(
            "identity_id",
            new_column_name="oplist_user_id",
            existing_type=sa.String(length=64),
            existing_nullable=False,
        )

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_users_oplist_user_id"), ["oplist_user_id"], unique=True)

    with op.batch_alter_table("sessions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("oplist_token", sa.String(length=512), nullable=False, server_default="")
        )
