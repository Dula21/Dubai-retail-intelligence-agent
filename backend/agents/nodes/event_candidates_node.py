"""
agents/nodes/event_candidates_node.py
---------------------------------------
Entry node for EVENT-PLANNING questions ("what should I stock more of before Diwali").

Why this node exists
  Retrieval is the wrong first step for a planning question. There is no single product to look up,
  near-identical variants tie within ~0.003, and the catalogue holds no event-specific products yet,
  so a retrieval-margin gate returns "insufficient data" to a question the calendar can answer.
  Planning therefore starts from the calendar:

    event (named in the query, else the strongest upcoming one)
      -> catalogue SKUs the event's lift table says will surge (category + keyword rows, same rule the
         synthetic data generator uses)
      -> ranked: highest expected lift first, then shortest stock runway
      -> handed to the normal chain: sales -> inventory -> seasonality -> recommendation

Deterministic and explainable: no vector search, no LLM. Every selection is in the trace.
If the event is unknown, or no SKU is lifted, the node returns no candidates and says so; the
recommendation node then reports insufficient data rather than guessing.

Known limits: lifts come from the SYNTHETIC lift table (not observed history); the catalogue has no
festival-specific products yet (diyas, sweet boxes, Christmas decor), so only existing categories and
keywords can match.
"""

from __future__ import annotations

import structlog

from .. import catalogue
from ..state import AgentState
from .seasonality_node import (
    DEFAULT_LEAD_TIME_DAYS,
    MIN_LIFT,
    _load_events,
    _query_text,
    _today,
    lift_for,
    pick_event,
)

logger = structlog.get_logger(__name__)

MAX_CANDIDATES = 8


def _lead_days(row: dict) -> int | None:
    try:
        return int(float(row.get("supplier_lead_days")))
    except (TypeError, ValueError):
        return None


async def event_candidates_node(state: AgentState) -> dict:
    """Pick the event, then the catalogue SKUs it lifts. Returns retrieved_chunks in the usual shape."""
    today = _today()
    query = _query_text(state)
    trace = list(state["reasoning_trace"])

    _events, source = _load_events()
    if source != "calendar":
        trace.append("Event candidates: calendar file not found; cannot rank SKUs by event lift")
        return {"retrieved_chunks": [], "retrieval_confidence": 0.0, "retrieval_margin": 0.0,
                "reasoning_trace": trace}

    event, _others, named, notes = pick_event(query, [], today, DEFAULT_LEAD_TIME_DAYS)
    trace.extend(notes)
    if event is None:
        trace.append("Event candidates: no calendar event to plan for (none named, none lifting products "
                     "in the next 90 days); no candidates selected")
        return {"retrieved_chunks": [], "retrieval_confidence": 0.0, "retrieval_margin": 0.0,
                "reasoning_trace": trace}

    rows = catalogue.all_rows()
    if not rows:
        trace.append("Event candidates: product catalogue not found; no candidates selected")
        return {"retrieved_chunks": [], "retrieval_confidence": 0.0, "retrieval_margin": 0.0,
                "reasoning_trace": trace}

    lifted = []
    for row in rows:
        text = f"{row.get('name_en', '')} {row.get('tags_en', '')}"
        lift = lift_for(event, row.get("category", ""), text)
        if lift >= MIN_LIFT:
            runway = catalogue.days_of_stock_left(row["sku_id"])
            lifted.append((row, lift, runway))

    # Highest expected lift first; among equals, shortest runway first; unknown runway last. SKU as a stable tiebreak.
    lifted.sort(key=lambda t: (-t[1], t[2] if t[2] is not None else 1e9, t[0]["sku_id"]))
    chosen = lifted[:MAX_CANDIDATES]

    if not chosen:
        trace.append(f"Event candidates: {event.name} ({event.start} to {event.end}) lifts no catalogue "
                     f"SKU at {MIN_LIFT}x or more; no candidates selected")
        return {"retrieved_chunks": [], "retrieval_confidence": 0.0, "retrieval_margin": 0.0,
                "reasoning_trace": trace}

    chunks = []
    for row, lift, _runway in chosen:
        chunks.append({
            "sku_id": row["sku_id"],
            "product_name_en": row.get("name_en", ""),
            "product_name_ar": row.get("name_ar", ""),
            "category": row.get("category", ""),
            "similarity_score": 1.0,          # not a similarity: selection is deterministic (lift table)
            "content": " | ".join(str(row.get(k, "")) for k in ("name_en", "name_ar", "category", "tags_en")
                                  if row.get(k)),
            "event_name": event.name,
            "event_lift": round(lift, 2),
            "supplier_lead_days": _lead_days(row),
        })

    trace.append(
        f"Event candidates: {event.name} ({event.start} to {event.end}); {len(lifted)} catalogue SKUs "
        f"are lifted {MIN_LIFT}x or more by this event; showing the top {len(chunks)} by expected lift, then "
        f"shortest stock runway (deterministic, from the lift table; no vector search): "
        + ", ".join(f"{c['sku_id']} {c['event_lift']}x" for c in chunks)
    )
    logger.info("event_candidates_selected", event_name=event.name, lifted=len(lifted), returned=len(chunks),
                named=sorted(named))

    return {
        "retrieved_chunks": chunks,
        "retrieval_confidence": 1.0,       # deterministic selection, like the exact-SKU route
        "retrieval_margin": 1.0,
        "reasoning_trace": trace,
    }