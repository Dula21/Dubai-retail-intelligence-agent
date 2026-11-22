"""
agents/reorder_repo.py
-----------------------
One place that decides WHERE reorder recommendations live: PostgreSQL when DATABASE_URL is set, the bounded in-process
store (agents/reorder_scan.py) when it is not (local dev, unit tests).

Policy when Postgres is configured but unreachable (or the migration has not been run):
  * READS fall back to the in-memory store and say so (`source: "memory"`); nothing is lost, it is just not durable.
  * A scan still completes; its recommendations are held in memory and the response says `persisted: false`.
  * A HUMAN DECISION is refused (503), never applied to memory only. An approval or rejection that vanishes on restart
    is worse than a refusal: the buyer thinks the audit trail holds it, and it does not.
"""

from __future__ import annotations

import os
from typing import Optional

import structlog

from services import reorder_db as db
from services import reorder_override as ov

from . import reorder_scan as rs

logger = structlog.get_logger(__name__)

_mem_events: dict[str, list[dict]] = {}


def db_configured() -> bool:
    return bool(os.getenv("DATABASE_URL", "").strip())


def clear_memory() -> None:
    """For tests."""
    rs.clear_state()
    _mem_events.clear()


def _mem_log(event: dict) -> None:
    _mem_events.setdefault(event["rec_id"], []).append(event)


# ── writes ─────────────────────────────────────────────────────────────────────

async def persist_new(recs: list[dict]) -> tuple[list[dict], bool, Optional[str]]:
    """
    Store scan output. Returns (current version of each recommendation, persisted_to_database, warning).
    Always also fills the in-memory store, which is the read fallback.
    """
    for r in recs:
        is_new = r["rec_id"] not in rs._store
        rs.store_put(r)
        if is_new:
            _mem_log({"rec_id": r["rec_id"], "event_type": "created", "actor": db.SYSTEM_ACTOR,
                      "reason": r.get("headline"), "qty_before": None, "qty_after": r["order_qty"],
                      "created_at": r["created_at"]})
    if not db_configured():
        return rs.store_list_ids([r["rec_id"] for r in recs]), False, None
    try:
        current = await db.save_new(recs)
        for c in current:                                   # keep the cache in step with human decisions
            if c.get("status") != "pending":
                rs._store[c["rec_id"]] = c
        return current, True, None
    except db.DatabaseUnavailable as exc:
        msg = (f"database unavailable ({exc}); recommendations are held in memory only. "
               f"If this is a new database, run: alembic upgrade head")
        logger.warning("reorder_persist_failed", error=str(exc))
        return rs.store_list_ids([r["rec_id"] for r in recs]), False, msg


async def decide(rec_id: str, decision: str, actor: str, reason: Optional[str] = None, qty: Optional[int] = None) -> tuple[dict, bool]:
    """
    Apply a human decision. Raises LookupError (unknown id), ov.OverrideError (invalid / already decided) and
    db.DatabaseUnavailable (database configured but down: the decision is NOT applied).
    """
    if db_configured():
        rec = await db.fetch(rec_id)
        if rec is None:
            raise LookupError(rec_id)
        updated, event = ov.apply_decision(rec, decision, actor, reason, qty)
        if not await db.record_decision(updated, event):
            raise ov.OverrideError(f"recommendation {rec_id} was decided by someone else a moment ago; "
                                   f"a decision is final", code="conflict")
        rs._store[rec_id] = updated
        _mem_log(event)
        return updated, True
    rec = rs._store.get(rec_id)
    if rec is None:
        raise LookupError(rec_id)
    updated, event = ov.apply_decision(rec, decision, actor, reason, qty)
    rs._store[rec_id] = updated
    _mem_log(event)
    return updated, False


# ── reads ──────────────────────────────────────────────────────────────────────

async def list_recs(status: Optional[str], sku_id: Optional[str]) -> tuple[list[dict], str]:
    if db_configured():
        try:
            return await db.fetch_list(status, sku_id), "postgres"
        except db.DatabaseUnavailable as exc:
            logger.warning("reorder_list_fell_back_to_memory", error=str(exc))
    return rs.store_list(status=status, sku_id=sku_id), "memory"


async def get_one(rec_id: str) -> tuple[Optional[dict], list[dict], str]:
    """(recommendation, history, source)."""
    if db_configured():
        try:
            rec = await db.fetch(rec_id)
            return rec, (await db.fetch_events(rec_id) if rec else []), "postgres"
        except db.DatabaseUnavailable as exc:
            logger.warning("reorder_get_fell_back_to_memory", error=str(exc))
    return rs._store.get(rec_id), list(_mem_events.get(rec_id, [])), "memory"
