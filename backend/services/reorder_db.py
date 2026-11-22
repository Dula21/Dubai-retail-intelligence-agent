"""
services/reorder_db.py
-----------------------
PostgreSQL persistence for reorder recommendations and their audit trail (Phase 3). SQLAlchemy 2.0 Core + asyncpg;
the schema is owned by Alembic (backend/migrations), not created here.

Tables
  recommendations        one row per recommendation. `record` holds the system's output exactly as produced (trace,
                         confidence, flags, numbers) and is never rewritten after creation; the columns status,
                         decided_by, decided_at, decision_reason and final_qty hold the human decision beside it.
  recommendation_events  append-only audit log: created / approved / modified / rejected, who, why, qty before/after.

DECISION: Alembic migrations, not init.sql.
  init.sql runs only when the Postgres volume is first created, so it cannot evolve a database that already has data.
  Alternative: keep editing init.sql and tell people to wipe the volume. Trade-off: one more command
  (`alembic upgrade head`) and a migrations folder; in return a schema change is a reviewed, reversible file.

DECISION: no ORM. Core tables + explicit statements. Seven queries do not justify mapped classes, and every
statement is readable in an interview.

Degradation: DATABASE_URL unset, or Postgres unreachable -> every function raises DatabaseUnavailable and the caller
falls back to the in-memory store (and says so). Scale ceiling: a single Postgres on the free tier handles this
volume (hundreds of recommendations per scan); at noon scale the audit log would be partitioned by month and
streamed to a warehouse. Documented, not implemented.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

metadata = sa.MetaData()
JSONType = sa.JSON().with_variant(JSONB(), "postgresql")

recommendations = sa.Table(
    "recommendations", metadata,
    sa.Column("rec_id", sa.String(32), primary_key=True),
    sa.Column("sku_id", sa.String(32), nullable=False, index=True),
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
    sa.Column("status", sa.String(12), nullable=False, server_default="pending", index=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("decided_by", sa.String(80)),
    sa.Column("decided_at", sa.DateTime(timezone=True)),
    sa.Column("decision_reason", sa.Text),
    sa.Column("final_qty", sa.Integer),
)

events = sa.Table(
    "recommendation_events", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("rec_id", sa.String(32), sa.ForeignKey("recommendations.rec_id"), nullable=False, index=True),
    sa.Column("event_type", sa.String(16), nullable=False),
    sa.Column("actor", sa.String(80), nullable=False),
    sa.Column("reason", sa.Text),
    sa.Column("qty_before", sa.Integer),
    sa.Column("qty_after", sa.Integer),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

SYSTEM_ACTOR = "system:reorder-scan"


class DatabaseUnavailable(RuntimeError):
    """No DATABASE_URL, or the database could not be reached."""


@lru_cache(maxsize=1)
def _engine_for(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_size=3, max_overflow=2, pool_pre_ping=True) \
        if not url.startswith("sqlite") else create_async_engine(url)


def get_engine() -> AsyncEngine:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise DatabaseUnavailable("DATABASE_URL is not set")
    return _engine_for(url)


def reset_engine() -> None:
    """For tests."""
    _engine_for.cache_clear()


def _d(value) -> Optional[date]:
    return date.fromisoformat(value) if isinstance(value, str) and value else None


def _dt(value) -> Optional[datetime]:
    return datetime.fromisoformat(value) if isinstance(value, str) and value else None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_rec(row) -> dict:
    """The system's record, with the CURRENT decision state laid over it."""
    rec = dict(row["record"])
    rec.update(status=row["status"], decided_by=row["decided_by"],
               decided_at=row["decided_at"].isoformat(timespec="seconds") if row["decided_at"] else None,
               decision_reason=row["decision_reason"], final_qty=row["final_qty"])
    if row["status"] == "modified" and row["final_qty"] is not None and row["order_qty"]:
        # Derived on read, so the percentage is the same right after a decision and after a reload from the database.
        rec["modification_delta_pct"] = round(100.0 * (row["final_qty"] - row["order_qty"]) / row["order_qty"], 1)
    return rec


def _wrap(exc: Exception) -> DatabaseUnavailable:
    return DatabaseUnavailable(f"{type(exc).__name__}: {str(exc)[:200]}")


# ── writes ─────────────────────────────────────────────────────────────────────

async def save_new(recs: list[dict]) -> list[dict]:
    """
    Insert recommendations that do not exist yet (with a 'created' event); refresh the system record of one that is
    still pending; never touch one a human has decided. Returns the CURRENT stored version of every input, in order.
    """
    try:
        out = []
        async with get_engine().begin() as conn:
            for r in recs:
                found = (await conn.execute(sa.select(recommendations).where(
                    recommendations.c.rec_id == r["rec_id"]))).mappings().first()
                if found is None:
                    await conn.execute(sa.insert(recommendations).values(
                        rec_id=r["rec_id"], sku_id=r["sku_id"], product_name=r.get("product_name"),
                        category=r.get("category"), trigger=r["trigger"], event=r.get("event"), urgency=r["urgency"],
                        order_qty=r["order_qty"], order_by_date=_d(r.get("order_by_date")),
                        window_start=_d(r.get("window_start")), window_end=_d(r.get("window_end")),
                        confidence=r["confidence"], confidence_basis=r.get("confidence_basis"),
                        as_of=_d(r["as_of"]), replay=bool(r.get("replay")), record=r, status="pending",
                        created_at=_dt(r.get("created_at")) or _now()))
                    await conn.execute(sa.insert(events).values(
                        rec_id=r["rec_id"], event_type="created", actor=SYSTEM_ACTOR,
                        reason=r.get("headline"), qty_before=None, qty_after=r["order_qty"], created_at=_now()))
                    out.append(r)
                elif found["status"] == "pending":
                    await conn.execute(sa.update(recommendations).where(recommendations.c.rec_id == r["rec_id"])
                                       .values(urgency=r["urgency"], order_qty=r["order_qty"],
                                               order_by_date=_d(r.get("order_by_date")), confidence=r["confidence"],
                                               confidence_basis=r.get("confidence_basis"), record=r))
                    out.append(r)
                else:
                    out.append(_row_to_rec(found))                    # a human decided: keep their decision
        return out
    except DatabaseUnavailable:
        raise
    except Exception as exc:                                          # connection, missing table (no migration), ...
        raise _wrap(exc)


async def record_decision(updated: dict, event: dict) -> bool:
    """
    Apply a decision atomically: the UPDATE only matches a row that is still pending, so two people deciding at the
    same moment cannot both win. Returns False when the row was already decided (or does not exist).
    """
    try:
        async with get_engine().begin() as conn:
            res = await conn.execute(
                sa.update(recommendations)
                .where(recommendations.c.rec_id == updated["rec_id"], recommendations.c.status == "pending")
                .values(status=updated["status"], decided_by=updated["decided_by"],
                        decided_at=_dt(updated["decided_at"]), decision_reason=updated["decision_reason"],
                        final_qty=updated["final_qty"]))
            if res.rowcount != 1:
                return False
            await conn.execute(sa.insert(events).values(**{**event, "created_at": _dt(event["created_at"])}))
            return True
    except Exception as exc:
        raise _wrap(exc)


# ── reads ──────────────────────────────────────────────────────────────────────

async def fetch(rec_id: str) -> Optional[dict]:
    try:
        async with get_engine().connect() as conn:
            row = (await conn.execute(sa.select(recommendations).where(
                recommendations.c.rec_id == rec_id))).mappings().first()
        return _row_to_rec(row) if row else None
    except Exception as exc:
        raise _wrap(exc)


async def fetch_list(status: Optional[str] = None, sku_id: Optional[str] = None, limit: int = 500) -> list[dict]:
    try:
        q = sa.select(recommendations)
        if status:
            q = q.where(recommendations.c.status == status)
        if sku_id:
            q = q.where(recommendations.c.sku_id == sku_id)
        q = q.order_by(recommendations.c.order_by_date.asc().nulls_last(), recommendations.c.sku_id).limit(limit)
        async with get_engine().connect() as conn:
            rows = (await conn.execute(q)).mappings().all()
        order = {"late": 0, "now": 1, "upcoming": 2}
        return sorted((_row_to_rec(r) for r in rows), key=lambda r: (order.get(r["urgency"], 3),))
    except Exception as exc:
        raise _wrap(exc)


async def fetch_events(rec_id: str) -> list[dict]:
    try:
        async with get_engine().connect() as conn:
            rows = (await conn.execute(sa.select(events).where(events.c.rec_id == rec_id)
                                       .order_by(events.c.id))).mappings().all()
        return [{**dict(r), "created_at": r["created_at"].isoformat(timespec="seconds")} for r in rows]
    except Exception as exc:
        raise _wrap(exc)


async def ping() -> bool:
    try:
        async with get_engine().connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
        return True
    except Exception:
        return False
