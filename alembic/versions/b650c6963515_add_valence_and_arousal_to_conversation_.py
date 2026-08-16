"""add valence and arousal to conversation emotions

Revision ID: b650c6963515
Revises: 6d655aa8b6aa
Create Date: 2026-08-16 10:41:16.327133

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b650c6963515"
down_revision: Union[str, Sequence[str], None] = "6d655aa8b6aa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversation_emotions",
        sa.Column(
            "valence",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "conversation_emotions",
        sa.Column(
            "arousal",
            sa.Float(),
            nullable=True,
        ),
    )

    op.create_check_constraint(
        "check_conversation_emotion_valence",
        "conversation_emotions",
        "valence >= -1.0 AND valence <= 1.0 OR valence IS NULL",
    )

    op.create_check_constraint(
        "check_conversation_emotion_arousal",
        "conversation_emotions",
        "arousal >= 0.0 AND arousal <= 1.0 OR arousal IS NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "check_conversation_emotion_arousal",
        "conversation_emotions",
        type_="check",
    )

    op.drop_constraint(
        "check_conversation_emotion_valence",
        "conversation_emotions",
        type_="check",
    )

    op.drop_column(
        "conversation_emotions",
        "arousal",
    )

    op.drop_column(
        "conversation_emotions",
        "valence",
    )