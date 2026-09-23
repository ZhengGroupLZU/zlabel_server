"""projects.key: the stable project id behind every anno_id

Existing projects get a generated key; scanning adopts (or writes) the key in the
dataset's ``.zlabel/project.json`` and ``v2.cli migrate-anno-ids`` renames the
files that were written with the old ``md5("<project name>/<rel>")`` ids.

Revision ID: 7c2b9a41d0f5
Revises: 4f7c1d2ab9e3
Create Date: 2026-09-22 21:40:00.000000

"""

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c2b9a41d0f5"
down_revision: Union[str, Sequence[str], None] = "4f7c1d2ab9e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # nullable first: the column has to exist before every row can get a key
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.add_column(sa.Column("key", sa.String(length=64), nullable=True))

    bind = op.get_bind()
    for (project_id,) in bind.execute(sa.text("SELECT id FROM projects")).fetchall():
        bind.execute(
            sa.text("UPDATE projects SET key = :key WHERE id = :id"),
            {"key": uuid.uuid4().hex[:12], "id": project_id},
        )

    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.alter_column("key", existing_type=sa.String(length=64), nullable=False)

    # the index is created in its own batch step: SQLite recreates the table
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_projects_key"), ["key"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_projects_key"))
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_column("key")
