"""reorder recommendations and their audit log

Revision ID: 0001
Revises:

Adds `recommendations` (system output + human decision, side by side) and `recommendation_events` (append-only log).
The Phase 2 placeholder table `reorder_recommendations` from init.sql is left untouched: nothing reads it, and
dropping a table is not needed to add the new ones.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "recommendations",
        sa.Column("rec_id", sa.String(32), primary_key=True),
        sa.Column("sku_id", sa.String(32), nullable=False),
        sa.Column("product_name", sa.String(200)),
        sa.Column("category", sa.String(40)),
        sa.Column("trigger", sa.String(40), nullable=False),
        sa.Column("event", sa.String(80)),
        sa.Column("urgency", sa.String(12), nullable=False),
        sa.Column("order_qty", sa.Integer, nullable=False),
        sa.Column("order_by_date", sa.Date),
        sa.Column("window_start", sa.Date),
        sa.Column("window_end", sa.Date),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("confidence_basis", sa.Text),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("replay", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("record", JSONType, nullable=False),
        sa.Column("status", sa.String(12), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_by", sa.String(80)),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.Column("decision_reason", sa.Text),
        sa.Column("final_qty", sa.Integer),
        sa.CheckConstraint("status IN ('pending','approved','rejected','modified')", name="ck_rec_status"),
    )
    op.create_index("ix_recommendations_sku_id", "recommendations", ["sku_id"])
    op.create_index("ix_recommendations_status", "recommendations", ["status"])
    op.create_table(
        "recommendation_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("rec_id", sa.String(32), sa.ForeignKey("recommendations.rec_id"), nullable=False),
        sa.Column("event_type", sa.String(16), nullable=False),
        sa.Column("actor", sa.String(80), nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("qty_before", sa.Integer),
        sa.Column("qty_after", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_recommendation_events_rec_id", "recommendation_events", ["rec_id"])


def downgrade() -> None:
    op.drop_index("ix_recommendation_events_rec_id", table_name="recommendation_events")
    op.drop_table("recommendation_events")
    op.drop_index("ix_recommendations_status", table_name="recommendations")
    op.drop_index("ix_recommendations_sku_id", table_name="recommendations")
    op.drop_table("recommendations")
