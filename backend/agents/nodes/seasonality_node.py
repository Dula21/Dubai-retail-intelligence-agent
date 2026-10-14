"""
agents/nodes/seasonality_node.py
----------------------------------
Node 4: UAE retail seasonality, driven by the retail calendar.

Reads data/synthetic/uae_retail_calendar.csv and event_lifts.csv, the SAME two files the synthetic
data generator uses (all data is SYNTHETIC; see data/make_calendar.py for how event dates are
computed: Umm al-Qura for Hijri dates, the `holidays` library, fixed or rule-based dates, and one
event, Dubai Shopping Festival, marked estimate_verify because its dates are announced yearly).

What the node answers
  "what should I stock before Diwali?"  -> that event's next occurrence, its dates, the expected lift
                                           for the retrieved products, and an ORDER-BY date.
  no event named in the query            -> the next event within HORIZON_DAYS that actually lifts
                                           the retrieved products' categories, plus the runners-up.

Order-by date (the actionable number)
  order_by = event_start - lead_in_days - supplier_lead_days - ORDER_BUFFER_DAYS
  Demand starts building lead_in_days before the event (calendar column), and stock must be ON THE
  SHELF by then, so supplier lead time is subtracted from that point, not from the event start.

Alert level is keyed to the order-by date, not the event date, because lead time is what makes a
date actionable:
  "critical" -> order-by date already passed (expedite / accept a stockout risk)
  "act"      -> order-by within ACT_WITHIN_DAYS days, or the event is already running
  "watch"    -> event within the horizon, order-by still comfortably ahead
  "none"     -> nothing relevant in the horizon

Engineering decision: calendar file over a hard-coded table.
  Chosen: CSV calendar shared with the data generator and (next) Prophet's `holidays` frame, so one
  source of truth drives data, forecasting and this node. Trade-off: the file must be regenerated
  each year (make_calendar.py does that); the old hard-coded table silently went stale instead.
Fallback: if the calendar file is missing, the old static table is used and the trace says so.

Chunk enrichment: the RAG chunk has no supplier lead time or tags, so those are read from the catalogue
CSV by SKU (agents/catalogue.py). Missing everywhere -> the default is used and the trace says "assumed".

Known limits (documented, not hidden):
  * multipliers come from the synthetic calendar's lift table, not from observed history yet.
  * DSF dates are an estimate until Dubai Tourism announces them (date_basis = estimate_verify).
  * overlapping events are reported one by one; their lifts are not combined here.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import structlog

from services.event_aliases import named_event_names

from .. import catalogue
from ..state import AgentState, SeasonalitySignal

logger = structlog.get_logger(__name__)

CALENDAR_PATH = Path(os.getenv("CALENDAR_DATA_PATH", "data/synthetic/uae_retail_calendar.csv"))
LIFTS_PATH = Path(os.getenv("EVENT_LIFTS_DATA_PATH", "data/synthetic/event_lifts.csv"))

HORIZON_DAYS = 90               # look-ahead when the query names no event
NAMED_EVENT_HORIZON_DAYS = 400  # look-ahead when the query names an event ("before Ramadan")
ORDER_BUFFER_DAYS = 5           # safety margin on top of supplier lead time
DEFAULT_LEAD_TIME_DAYS = 14     # used only when no retrieved product carries a lead time (trace says so)
MIN_LIFT = 1.15                 # below this an event is not worth flagging for the retrieved products
STRONG_LIFT = 1.5               # events at or above this rank ahead of mild ones, whatever their start date
ACT_WITHIN_DAYS = 14
BACKGROUND_EVENT_DAYS = 45      # a running event longer than this (a season) does not hide sharper ones
OTHER_EVENTS_SHOWN = 4

# Legacy table: used ONLY if the calendar CSV is missing.
UAE_RETAIL_EVENTS = [
    {"name": "Back-to-School", "start": date(2026, 8, 20), "end": date(2026, 9, 10), "multiplier": 1.6,
     "categories": ["fashion", "electronics", "stationery"]},
    {"name": "National Day", "start": date(2026, 12, 1), "end": date(2026, 12, 4), "multiplier": 1.5,
     "categories": ["fashion", "home", "food", "beauty"]},
    {"name": "DSF", "start": date(2026, 12, 15), "end": date(2027, 1, 15), "multiplier": 2.1,
     "categories": ["fashion", "electronics", "home", "beauty"]},
    {"name": "Ramadan", "start": date(2027, 2, 8), "end": date(2027, 3, 8), "multiplier": 1.8,
     "categories": ["food", "beauty", "fashion", "home"]},
    {"name": "Eid Al Fitr", "start": date(2027, 3, 9), "end": date(2027, 3, 12), "multiplier": 1.9,
     "categories": ["fashion", "food", "beauty"]},
    {"name": "Eid Al Adha", "start": date(2027, 5, 16), "end": date(2027, 5, 19), "multiplier": 1.7,
     "categories": ["fashion", "food", "home"]},
]

QUERY_KEYS = ("query", "user_query", "question", "original_query")


@dataclass(frozen=True)
class Event:
    name: str
    start: date
    end: date
    lead_in_days: int
    date_basis: str
    lifts: tuple  # of (category, compiled regex or None, multiplier)


def _today() -> date:
    return date.today()


def _legacy_events() -> tuple:
    return tuple(sorted(
        (Event(e["name"], e["start"], e["end"], 0, "static_fallback",
               tuple((c, None, float(e["multiplier"])) for c in e["categories"]))
         for e in UAE_RETAIL_EVENTS),
        key=lambda e: e.start))


@lru_cache(maxsize=1)
def _load_events() -> tuple:
    """(events sorted by start, source) where source is 'calendar' or 'static_fallback'."""
    if not CALENDAR_PATH.exists():
        logger.warning("calendar_csv_not_found", path=str(CALENDAR_PATH))
        return _legacy_events(), "static_fallback"

    lifts: dict[str, list] = {}
    if LIFTS_PATH.exists():
        with open(LIFTS_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pattern = (r.get("name_regex") or "").strip()
                lifts.setdefault(r["event"], []).append((
                    r["category"].strip().lower(),
                    re.compile(pattern, re.IGNORECASE) if pattern else None,
                    float(r["multiplier"]),
                ))
    else:
        logger.warning("event_lifts_csv_not_found", path=str(LIFTS_PATH))

    events = []
    with open(CALENDAR_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            events.append(Event(
                name=r["event"],
                start=date.fromisoformat(r["start_date"][:10]),
                end=date.fromisoformat(r["end_date"][:10]),
                lead_in_days=int(float(r.get("lead_in_days") or 0)),
                date_basis=(r.get("date_basis") or "unknown"),
                lifts=tuple(lifts.get(r["event"], ())),
            ))
    events.sort(key=lambda e: e.start)
    logger.info("seasonality_calendar_loaded", events=len(events))
    return tuple(events), "calendar"


# ── helpers ───────────────────────────────────────────────────────────────────

_FIELD_ALIASES = {"name_en": ("name_en", "product_name_en")}


def _chunk_field(chunk: dict, key: str):
    """
    Field of a retrieved chunk. The RAG chunk carries only sku_id, product_name_en/ar, category and a
    content preview, so fields it lacks (supplier_lead_days, tags_en) are read from the catalogue CSV
    by SKU. Missing everywhere -> None (callers then say 'assumed', never invent a value).
    """
    for name in _FIELD_ALIASES.get(key, (key,)):
        value = chunk.get(name)
        if value in (None, ""):
            value = (chunk.get("metadata") or {}).get(name)
        if value not in (None, ""):
            return value
    row = catalogue.get_row(str(chunk.get("sku_id", "")))
    value = row.get(key)
    return value if value not in (None, "") else None


def _query_text(state: AgentState) -> str:
    for key in QUERY_KEYS:
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _lead_time(chunks: list) -> tuple[int, bool]:
    """Longest supplier lead time among retrieved products (conservative). (days, assumed?)"""
    values = []
    for chunk in chunks:
        raw = _chunk_field(chunk, "supplier_lead_days")
        try:
            values.append(int(float(raw)))
        except (TypeError, ValueError):
            continue
    return (max(values), False) if values else (DEFAULT_LEAD_TIME_DAYS, True)


def lift_for(event: Event, category: str, text: str) -> float:
    """
    Peak lift the event gives ONE product (category + name/tags text): max over matching rows of the
    lift table, the same rule the synthetic data generator uses (hero-flag boost not applied). 1.0 = none.
    """
    best = 1.0
    category = (category or "").lower()
    text = (text or "").lower()
    for lift_cat, pattern, mult in event.lifts:
        if lift_cat != category:
            continue
        if pattern is not None and not pattern.search(text):
            continue
        best = max(best, mult)
    return best


def _multiplier(event: Event, chunks: list) -> float:
    """
    Expected peak lift for the retrieved products (max over products). With no products to look at,
    the event's overall peak is returned. 1.0 = no lift.
    """
    if not event.lifts:
        return 1.0
    if not chunks:
        return round(max(m for _, _, m in event.lifts), 2)
    best = 1.0
    for chunk in chunks:
        category = str(_chunk_field(chunk, "category") or "")
        text = " ".join(str(_chunk_field(chunk, k) or "") for k in ("name_en", "tags_en"))
        best = max(best, lift_for(event, category, text))
    return round(best, 2)


def _describe(event: Event, today: date, chunks: list, lead_days: int) -> dict:
    order_by = event.start - timedelta(days=event.lead_in_days + lead_days + ORDER_BUFFER_DAYS)
    return {
        "name": event.name,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "days_until": (event.start - today).days,          # negative = already running
        "ongoing": event.start <= today <= event.end,
        "multiplier": _multiplier(event, chunks),
        "order_by": order_by.isoformat(),
        "days_to_order_by": (order_by - today).days,
        "date_basis": event.date_basis,
    }


def _alert(d: dict) -> str:
    if d["ongoing"]:
        return "act"
    if d["days_to_order_by"] <= 0:
        return "critical"
    if d["days_to_order_by"] <= ACT_WITHIN_DAYS:
        return "act"
    return "watch"


def _timing_phrase(d: dict) -> str:
    if d["ongoing"]:
        return "running now"
    return f"starts in {d['days_until']} days"


def _order_phrase(d: dict) -> str:
    n = d["days_to_order_by"]
    if d["ongoing"]:
        return f"order-by {d['order_by']} (event already running)"
    if n < 0:
        return f"order-by {d['order_by']} (passed {-n} days ago)"
    if n == 0:
        return f"order-by {d['order_by']} (today)"
    return f"order-by {d['order_by']} ({n} days left)"


# ── event selection (shared with the event-candidates node) ──────────────────

def pick_event(query: str, chunks: list, today: date, lead_days: int, force_name: str | None = None):
    """
    Choose the event to talk about. force_name (the event the candidates node already planned for)
    wins over the query text, so both nodes always talk about the same event.

    Returns (event, others, named, notes):
      event   the Event, or None when nothing relevant is in the horizon
      others  runner-up descriptions (dicts), strongest/soonest first
      named   calendar event names found in the query (set, may be empty)
      notes   trace lines explaining a fallback decision
    """
    events, _source = _load_events()
    named = {force_name} if force_name else (named_event_names(query) if query else set())
    notes: list[str] = []

    if named:
        horizon_end = today + timedelta(days=NAMED_EVENT_HORIZON_DAYS)
        matches = [e for e in events if e.name in named and e.end >= today and e.start <= horizon_end]
        if matches:
            return matches[0], [], named, notes
        notes.append(f"Seasonality: query names {sorted(named)} but no occurrence within "
                     f"{NAMED_EVENT_HORIZON_DAYS} days in the calendar; falling back to the next relevant event")

    horizon_end = today + timedelta(days=HORIZON_DAYS)
    scored = []
    for e in events:
        if e.end >= today and e.start <= horizon_end:
            d = _describe(e, today, chunks, lead_days)
            if d["multiplier"] >= MIN_LIFT:
                scored.append((e, d))
    # Ranking: a long-running season (e.g. Summer) must not hide a sharper event about to start, and a
    # mild event (1.2x) must not outrank a strong one (2.6x) just because it starts a few days earlier.
    scored.sort(key=lambda ed: (
        ed[1]["ongoing"] and (ed[0].end - ed[0].start).days > BACKGROUND_EVENT_DAYS,
        ed[1]["multiplier"] < STRONG_LIFT,
        ed[0].start,
    ))
    if not scored:
        return None, [], named, notes
    return scored[0][0], [d for _, d in scored[1:1 + OTHER_EVENTS_SHOWN]], named, notes


# ── node ──────────────────────────────────────────────────────────────────────

async def seasonality_check_node(state: AgentState) -> dict:
    """
    Node 4: pick the relevant UAE retail event and say when the retailer must order.

    Returns a partial state dict: seasonality_signal (the original four fields, plus event dates,
    order-by date, lead time, date basis and the runners-up) and reasoning_trace entries.
    """
    today = _today()
    chunks = list(state.get("retrieved_chunks", []))
    _events, source = _load_events()
    lead_days, lead_assumed = _lead_time(chunks)
    query = _query_text(state)
    trace: list[str] = []

    if source != "calendar":
        trace.append("Seasonality: calendar file not found; using the static fallback table (dates may be stale)")

    planned = next((c.get("event_name") for c in chunks if c.get("event_name")), None)
    event, others, named, notes = pick_event(query, chunks, today, lead_days, force_name=planned)
    trace.extend(notes)
    primary = _describe(event, today, chunks, lead_days) if event else None

    if primary is None:
        signal = SeasonalitySignal(
            upcoming_event=None, days_until_event=None, expected_demand_multiplier=1.0, alert_level="none",
            event_start=None, event_end=None, order_by_date=None, days_to_order_by=None,
            lead_time_days=lead_days, lead_time_assumed=lead_assumed, date_basis=None,
            named_in_query=bool(named), other_upcoming=[],
        )
        trace.append(f"Seasonality: no UAE retail event lifting these products within {HORIZON_DAYS} days; "
                     "baseline demand")
    else:
        alert = _alert(primary)
        signal = SeasonalitySignal(
            upcoming_event=primary["name"],
            days_until_event=primary["days_until"],
            expected_demand_multiplier=primary["multiplier"],
            alert_level=alert,
            event_start=primary["start"], event_end=primary["end"],
            order_by_date=primary["order_by"], days_to_order_by=primary["days_to_order_by"],
            lead_time_days=lead_days, lead_time_assumed=lead_assumed,
            date_basis=primary["date_basis"], named_in_query=bool(named),
            other_upcoming=others,
        )
        note = ", ".join(filter(None, [
            "lead time assumed (no supplier data on retrieved products)" if lead_assumed else "",
            "event dates are an estimate: verify" if primary["date_basis"] == "estimate_verify" else "",
            "named in query" if named else "",
        ]))
        trace.append(
            f"Seasonality: {primary['name']} {primary['start']} to {primary['end']} "
            f"({_timing_phrase(primary)}), expected demand {primary['multiplier']}x for these products, "
            f"supplier lead time {lead_days}d, {_order_phrase(primary)}, alert={alert}"
            + (f" [{note}]" if note else "")
        )
        if others:
            trace.append("Seasonality: also coming up: " + "; ".join(
                f"{o['name']} ({o['start']}, {o['multiplier']}x, order by {o['order_by']})" for o in others))

    logger.info("seasonality_node_complete", upcoming_event=signal["upcoming_event"],
                days_until=signal["days_until_event"], alert=signal["alert_level"], source=source)

    return {
        "seasonality_signal": signal,
        "reasoning_trace": state["reasoning_trace"] + trace,
    }