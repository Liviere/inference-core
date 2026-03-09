"""add skills column to user_agent_instances

Revision ID: e70bd305a912
Revises: b3dac404c08b
Create Date: 2026-03-09 15:13:08.261450

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

import inference_core

# revision identifiers, used by Alembic.
revision: str = "e70bd305a912"
down_revision: Union[str, Sequence[str], None] = "b3dac404c08b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "user_agent_instances",
        sa.Column(
            "skills", inference_core.database.sql.base.SmartJSON(), nullable=True
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("user_agent_instances", "skills")
