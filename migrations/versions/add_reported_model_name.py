"""add reported_model_name to llm_request_logs

Revision ID: add_reported_model_name
Revises: widen_agent_instance_text_cols
Create Date: 2026-10-12

``llm_request_logs.model_name`` held whatever the provider's response named,
which is often not the name the model is configured under: a dated snapshot
(``gpt-5-nano-2025-08-07``), an alias, a served version. Pricing and provider
were then looked up by a name the configuration does not know.

From this revision on ``model_name`` is the configured name of the model the
call was made on and ``provider`` is that model's provider. What the provider
reported is kept next to them, in this column. Rows written before stay as they
are: their ``model_name`` is the reported one and this column is NULL.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_reported_model_name"
down_revision: Union[str, Sequence[str], None] = "widen_agent_instance_text_cols"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_request_logs",
        sa.Column("reported_model_name", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_request_logs", "reported_model_name")
