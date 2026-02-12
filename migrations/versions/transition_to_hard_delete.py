"""remove soft delete columns

Revision ID: hard_delete_transition
Revises:
Create Date: 2026-02-12

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "hard_delete_transition"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Tables to drop columns from
    tables = [
        "users",
        "user_agent_instances",
        "llm_config_overrides",
        "user_llm_preferences",
        "allowed_user_overrides",
        "batch_jobs",
        "batch_items",
        "batch_events",
    ]

    for table in tables:
        # Drop columns if they exist
        # Note: drop_column handles constraints if the DB driver supports it
        # For SQLite, this might require recreate (Alembic handles batch mode)
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_column("is_deleted")
            batch_op.drop_column("deleted_at")


def downgrade():
    # Re-add columns
    tables = [
        "users",
        "user_agent_instances",
        "llm_config_overrides",
        "user_llm_preferences",
        "allowed_user_overrides",
        "batch_jobs",
        "batch_items",
        "batch_events",
    ]

    for table in tables:
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "is_deleted",
                    sa.Boolean(),
                    server_default=sa.text("false"),
                    nullable=False,
                )
            )
            batch_op.add_column(
                sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True)
            )
            batch_op.create_index(f"ix_{table}_is_deleted", ["is_deleted"])
