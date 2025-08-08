"""Add bot to channel

Revision ID: b1a2c3d4e5f6
Revises: 9f0c9cd09105
Create Date: 2025-07-14 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1a2c3d4e5f6"
down_revision: Union[str, None] = "9f0c9cd09105"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("channel", sa.Column("bot_name", sa.Text(), nullable=True))
    op.add_column("channel", sa.Column("bot_model", sa.Text(), nullable=True))
    op.add_column(
        "channel", sa.Column("bot_enabled", sa.Boolean(), nullable=False, server_default=sa.text('true'))
    )
    op.add_column("channel", sa.Column("bot_config", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("channel", "bot_config")
    op.drop_column("channel", "bot_enabled")
    op.drop_column("channel", "bot_model")
    op.drop_column("channel", "bot_name") 