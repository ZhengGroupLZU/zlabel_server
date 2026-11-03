"""label instance-type mapping

Adds the two columns that tie a label (part) to the instance types it may appear
in, and backfills the seed-germination spec table (``docs/export_spec.md`` §1.1)
onto the labels that already exist under a spec name. Project-specific labels are
left ``NULL`` = unconfigured, which the client treats as "no constraint".

The table is frozen here on purpose: a migration must not change meaning when the
runtime constant (``app/services/label_types.py``) is edited later.

Revision ID: 8c41d2f7a6b9
Revises: 753db795f6b6
Create Date: 2026-09-23 10:12:44.183920

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c41d2f7a6b9"
down_revision: Union[str, Sequence[str], None] = "753db795f6b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: (label name, default instance type, allowed instance types) — spec §1.1
SPEC_LABEL_TYPES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Seed", "normal_seed", ("normal_seed", "moldy_seed", "dead_seed")),
    ("Root", "normal_seedling", ("normal_seedling", "abnormal_seedling", "moldy_seed")),
    ("Shoot", "normal_seedling", ("normal_seedling", "abnormal_seedling")),
    ("Seed Coat", "normal_seed", ("normal_seed", "normal_seedling", "abnormal_seedling")),
    ("Dish", "dish", ("dish",)),
    ("Timestamp", "text", ("text",)),
    ("Number", "text", ("text",)),
)


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("default_instance_type", sa.String(length=64), nullable=False, server_default="")
        )
        # NULL (not "[]"): unconfigured, so the client keeps its spec-table fallback
        batch_op.add_column(sa.Column("allowed_instance_types", sa.Text(), nullable=True))

    labels = sa.table(
        "labels",
        sa.column("id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("default_instance_type", sa.String),
        sa.column("allowed_instance_types", sa.Text),
    )
    conn = op.get_bind()
    for name, default, allowed in SPEC_LABEL_TYPES:
        conn.execute(
            labels.update()
            .where(sa.func.lower(labels.c.name) == name.lower())
            .values(
                default_instance_type=default,
                allowed_instance_types=json.dumps(list(allowed), ensure_ascii=False),
            )
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.drop_column("allowed_instance_types")
        batch_op.drop_column("default_instance_type")
