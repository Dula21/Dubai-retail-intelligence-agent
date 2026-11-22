"""
agents/reorder_scan.py
-----------------------
The autonomous reorder scan (Phase 3): no question is asked. It looks at every SKU, decides which ones need an order
(rules in services/reorder_trigger.py), sizes those with the quantity engine, and stores a recommendation with its
own human-readable reasoning trace and a calibrated confidence.

No LLM is involved: the explanation is assembled from the same numbers the engine used, so it cannot drift from them.
(The chat agent explains a quantity in prose on request; this path is the unattended one.)

Cost control. Screening 500 SKUs is milliseconds; the Prophet quantity engine is ~0.1-0.5 s per SKU. At most
`limit` triggered SKUs are sized per scan (default MAX_SIZED_PER_SCAN), most urgent first; the rest are reported as
`deferred` with their ids, never silently dropped. Next architectural step at scale: batch-precomputed forecasts.

Confidence. A recommendation's confidence is the MEASURED held-out coverage of the group it belongs to (see
data/quantity_calibration.json), or UNVALIDATED_CONFIDENCE_CAP when it is outside what the backtest validated:
  * windows beyond the validated horizon,
  * windows with no event (the backtest cases were all event windows),
  * stock figures older than the snapshot limit.

Storage. Recommendations live in a bounded in-process store (MAX_STORED). It is NOT durable: a restart empties it
and the next scan rebuilds it. The PostgreSQL audit log (Alembic) is the next Phase 3 step; the status field
("pending") and the stable rec_id are what the human-override step will build on.
"""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import structlog

from services import quantity_engine as qe
from services import reorder_trigger as rt

from . import catalogue
from .nodes import inventory_node, quantity_node as qn, sales_node, seasonality_node as sn
from .nodes.recommendation_node import UNVALIDATED_CONFIDENCE_CAP

logger = structlog.get_logger(__name__)

MAX_SIZED_PER_SCAN = int(os.getenv("REORDER_MAX_SIZED_PER_SCAN", "40"))
MAX_STORED = int(os.getenv("REORDER_MAX_STORED", "2000"))
DEFERRED_PREVIEW = 20           # the response names this many deferred SKUs; `deferred_count` has the total

_store: "OrderedDict[str, dict]" = OrderedDict()
_scan_lock: Optional[asyncio.Lock] = None
_last_scan: Optional[dict] = None


# ── store ──────────────────────────────────────────────────────────────────────

def store_put(rec: dict) -> None:
    """Insert, or refresh a same-id recommendation that has not been acted on (status pending)."""
    old = _store.get(rec["rec_id"])
    if old is not None and old.get("status") != "pending":
        return                                      # a human already acted on it; never overwrite that
    _store[rec["rec_id"]] = rec
    _store.move_to_end(rec["rec_id"])
    while len(_store) > MAX_STORED:
        _store.popitem(last=False)


def store_list(status: Optional[str] = None, sku_id: Optional[str] = None) -> list[dict]:
    items = [r for r in _store.values() if (status is None or r["status"] == status)
             and (sku_id is None or r["sku_id"] == sku_id)]
    order = {"late": 0, "now": 1, "upcoming": 2}
    return sorted(items, key=lambda r: (order.get(r["urgency"], 3), r["order_by_date"] or "9999", r["sku_id"]))


def clear_state() -> None:
    """For tests."""
    global _last_scan
    _store.clear()
    _last_scan = None


def last_scan() -> Optional[dict]:
    return _last_scan


# ── confidence and wording ─────────────────────────────────────────────────────

def _confidence(res: dict, has_event: bool) -> tuple[float, str]:
    """(confidence, basis). Calibrated coverage, or the unvalidated cap with the reason."""
    if "stock_snapshot_stale" in res["flags"]:
        return UNVALIDATED_CONFIDENCE_CAP, "stock figure is stale"
    if not has_event:
        return UNVALIDATED_CONFIDENCE_CAP, "windows without an event were not part of the backtest; coverage not measured"
    cov = res.get("coverage_estimate")
    if cov is None:
        return UNVALIDATED_CONFIDENCE_CAP, res.get("coverage_basis") or "coverage not measured"
    return round(float(cov), 2), res.get("coverage_basis", "")


def _headline(sku: str, name: str, qty: int, order_by: Optional[date], event: Optional[rt.EventTrigger],
              as_of: date, arrival: date) -> str:
    base = f"Reorder {qty} units of {sku} ({name})" + (f" before {event.name}" if event else "")
    if order_by is None:
        return base
    d = (order_by - as_of).days
    if d > 0:
        return f"{base} by {order_by.isoformat()} ({d} days left)"
    if d == 0:
        return f"{base} today"
    return f"{base}: the order-by date was {order_by.isoformat()} ({-d} days ago); an order placed today arrives {arrival.isoformat()}"


# ── scan ───────────────────────────────────────────────────────────────────────

async def run_scan(as_of: Optional[date] = None, limit: Optional[int] = None) -> dict:
    """
    Scan every SKU in the inventory snapshot as of `as_of` (default today). A past `as_of` is a REPLAY: the data
    ends at the last sales date, so scanning "as of" that day shows what the agent would have raised then.
    Raises RuntimeError if a scan is already running.
    """
    global _scan_lock, _last_scan
    if _scan_lock is None:
        _scan_lock = asyncio.Lock()
    if _scan_lock.locked():
        raise RuntimeError("a scan is already running")
    async with _scan_lock:
        result = await _scan(as_of, limit if limit is not None else MAX_SIZED_PER_SCAN)
        from . import reorder_repo                      # lazy: reorder_repo imports this module
        current, persisted, warning = await reorder_repo.persist_new(result.pop("_new", []))
        result["recommendations"], result["persisted"] = current, persisted
        if warning:
            result["scan_trace"].append(f"WARNING: {warning}")
        _last_scan = {k: v for k, v in result.items() if k != "recommendations"}
        return result


async def _scan(as_of: Optional[date], limit: int) -> dict:
    today = sn._today()
    as_of = as_of or today
    if as_of > today:
        raise ValueError("as_of cannot be in the future")
    trace: list[str] = []

    calendar = qn._calendar_df()
    calibration = qn.load_calibration()
    sales_df = sales_node._load_sales_df()
    inv_df = inventory_node._load_inventory_df()
    if calendar is None or sales_df.empty or inv_df.empty:
        return _empty(as_of, today, "calendar, sales history or inventory snapshot not loaded; nothing scanned")

    snapshot_date = sales_df["date"].max().date()
    age = (as_of - snapshot_date).days
    events, _ = sn._load_events()
    by_sku = {k: g for k, g in sales_df.groupby("sku_id")}
    replay = as_of != today
    trace.append(f"Scan as of {as_of}{' (replay)' if replay else ''}: stock snapshot dated {snapshot_date} "
                 f"(= last sales date), {age} days old; {len(inv_df)} SKUs in the snapshot")

    # ── 1. cheap screen over every SKU ────────────────────────────────────────
    not_assessed = {"stock_unknown": 0, "snapshot_stale": 0, "rate_unknown": 0}
    candidates = []
    for _, row in inv_df.iterrows():
        sku = str(row["sku_id"])
        cat = catalogue.get_row(sku)
        lead, lead_assumed = qn._lead_days({"sku_id": sku})
        raw = inventory_node._read_number(row, inventory_node.STOCK_COLUMNS)
        rate, rate_days = rt.recent_rate(by_sku[sku]) if sku in by_sku else (None, 0)

        runway = rt.assess_runway(raw, snapshot_date, as_of, rate, rate_days, lead)
        if runway.status in not_assessed:
            not_assessed[runway.status] += 1

        category = str(cat.get("category", ""))
        text = " ".join(str(cat.get(k, "")) for k in ("name_en", "tags_en"))
        event = rt.next_event_trigger(events, lambda e: sn.lift_for(e, category, text), as_of, lead)

        if not (runway.triggered or event):
            continue
        reasons = [name for name, on in (("runway", runway.triggered), ("event", event is not None)) if on]
        if event is not None:
            dtob = event.days_to_order_by
        else:
            dtob = int(runway.must_order_in) if runway.must_order_in is not None else 0
        candidates.append({
            "sku": sku, "name": str(cat.get("name_en", sku)), "category": category, "lead": lead,
            "lead_assumed": lead_assumed, "on_hand": raw, "rate": rate, "runway": runway, "event": event,
            "trigger": "+".join(reasons), "days_to_order_by": dtob,
        })

    # Most urgent first: empty shelves, then overdue before upcoming, then the shortest remaining runway.
    candidates.sort(key=lambda c: (c["runway"].remaining_runway != 0.0,        # already out of stock first
                                   c["days_to_order_by"],
                                   c["runway"].remaining_runway if c["runway"].remaining_runway is not None else 1e9,
                                   c["sku"]))
    trace.append(f"Screen: {len(candidates)} SKUs triggered "
                 f"({sum('runway' in c['trigger'] for c in candidates)} runway, "
                 f"{sum('event' in c['trigger'] for c in candidates)} event); not assessed by the runway rule: "
                 f"{not_assessed['snapshot_stale']} stale stock, {not_assessed['stock_unknown']} unknown stock, "
                 f"{not_assessed['rate_unknown']} no usable sales rate")

    # ── 2. size the most urgent ones with the quantity engine ─────────────────
    to_size, deferred = candidates[:limit], candidates[limit:]
    recs: list[dict] = []
    skipped = {"already_covered": 0, "no_quantity": 0, "arrives_too_late": 0}
    for c in to_size:
        ev, lead = c["event"], c["lead"]
        event_dict = {"start": ev.start, "end": ev.end, "lead_in_days": ev.lead_in_days} if ev else None
        window = qe.cover_window(as_of, lead, event_dict)
        if window is None:
            skipped["arrives_too_late"] += 1
            continue
        forecast = await qn._get_forecast(c["sku"], sales_df, calendar)
        res = qe.order_quantity(
            sku_id=c["sku"], forecast=forecast, window=window,
            safety=calibration["safety_obj"] if calibration else None,
            on_hand=c["on_hand"], snapshot_date=snapshot_date, as_of=as_of, open_orders=None, calendar=calendar,
            coverage_table=(calibration or {}).get("coverage"),
            validated_horizon_days=(calibration or {}).get("validated_horizon_days", qe.VALIDATED_HORIZON_DAYS))
        if c["lead_assumed"]:
            res.flags.append("lead_time_assumed")
        d = res.to_dict()
        if d["order_qty"] is None:
            skipped["no_quantity"] += 1
            continue
        if d["order_qty"] == 0:
            skipped["already_covered"] += 1
            continue
        recs.append(_record(c, d, as_of, replay))

    trace.append(f"Sized {len(to_size)} of {len(candidates)} triggered SKUs: {len(recs)} recommendations, "
                 f"{skipped['already_covered']} already covered by stock, {skipped['no_quantity']} without enough "
                 f"data for a quantity, {skipped['arrives_too_late']} could not arrive in time; "
                 f"{len(deferred)} deferred (raise `limit` to size them)")
    logger.info("reorder_scan_complete", as_of=str(as_of), triggered=len(candidates), recommended=len(recs),
                deferred=len(deferred))

    return {
        "as_of": as_of.isoformat(), "replay": replay, "snapshot_date": snapshot_date.isoformat(),
        "snapshot_age_days": age, "skus_in_snapshot": int(len(inv_df)), "triggered": len(candidates),
        "sized": len(to_size), "not_assessed": not_assessed, "skipped": skipped,
        "deferred_count": len(deferred), "deferred": [c["sku"] for c in deferred[:DEFERRED_PREVIEW]],
        "recommendations": [], "_new": recs,
        "scan_trace": trace, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def store_list_ids(ids: list[str]) -> list[dict]:
    """The stored version of each id (an acted-on recommendation keeps its human status), urgency order."""
    items = [_store[i] for i in ids if i in _store]
    order = {"late": 0, "now": 1, "upcoming": 2}
    return sorted(items, key=lambda r: (order.get(r["urgency"], 3), r["order_by_date"] or "9999", r["sku_id"]))


def _empty(as_of: date, today: date, why: str) -> dict:
    return {"as_of": as_of.isoformat(), "replay": as_of != today, "snapshot_date": None, "snapshot_age_days": None,
            "skus_in_snapshot": 0, "triggered": 0, "sized": 0, "not_assessed": {}, "skipped": {}, "deferred": [],
            "deferred_count": 0, "persisted": False,
            "recommendations": [], "scan_trace": [why],
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _record(c: dict, d: dict, as_of: date, replay: bool) -> dict:
    ev: Optional[rt.EventTrigger] = c["event"]
    runway: rt.RunwayAssessment = c["runway"]
    if ev is not None:
        order_by = ev.order_by
    elif runway.must_order_in is not None:
        order_by = as_of + timedelta(days=int(runway.must_order_in // 1))   # floor: a fraction of a day is not slack
    else:
        order_by = None
    days_to = (order_by - as_of).days if order_by else 0
    out_of_stock = runway.remaining_runway == 0.0            # an empty shelf is late whatever the formula says
    urg = "late" if out_of_stock and ev is None else rt.urgency(days_to)
    confidence, basis = _confidence(d, ev is not None)

    reasons = []
    if ev is not None:
        reasons.append(f"{ev.name} ({ev.start} to {ev.end}) lifts this product {ev.lift}x in the calendar; "
                       f"order-by {ev.order_by} = event start - {ev.lead_in_days}d lead-in - {c['lead']}d supplier "
                       f"lead time - {rt.ORDER_BUFFER_DAYS}d buffer")
    if runway.triggered:
        reasons.append(runway.trace)
    elif runway.status != "assessed":
        reasons.append(f"runway rule: {runway.trace}")
    arrival = as_of + timedelta(days=c["lead"])
    timing = f"Timing: an order placed {as_of} arrives {arrival} ({c['lead']}d supplier lead time)"
    if ev is not None:
        ramp_start = ev.start - timedelta(days=ev.lead_in_days)
        missed = max(0, (arrival - ramp_start).days)
        timing += (f"; the demand ramp-up begins {ramp_start}" + (
            (f", so all {ev.lead_in_days} of its days will not be covered" if missed >= ev.lead_in_days else
             f", so the first {missed} of its {ev.lead_in_days} days will not be covered")
            if missed > 0 and ev.lead_in_days else
            ", so it arrives in time for the ramp-up" + (
                f" (but the {rt.ORDER_BUFFER_DAYS}-day safety buffer is already used up)" if days_to < 0 else "")))
    trace = [f"Trigger: {c['trigger']} | urgency: {urg}"] + [f"Why: {r}" for r in reasons] + [timing] + list(d["trace"]) + [
        f"Confidence {confidence}: {basis}"]
    if replay:
        trace.insert(0, f"REPLAY: scanned as of {as_of}, not today")

    return {
        "rec_id": rt.rec_id(c["sku"], c["trigger"], ev.name if ev else None, as_of),
        "sku_id": c["sku"], "product_name": c["name"], "category": c["category"],
        "trigger": c["trigger"], "event": ev.name if ev else None, "urgency": urg,
        "order_qty": d["order_qty"], "order_by_date": order_by.isoformat() if order_by else None,
        "days_to_order_by": days_to, "window_start": d["window_start"], "window_end": d["window_end"],
        "forecast_units": d["forecast_units"], "safety_multiplier": d["safety_multiplier"],
        "target_units": d["target_units"], "on_hand_snapshot": d["on_hand_snapshot"],
        "snapshot_date": d["snapshot_date"], "snapshot_age_days": d["snapshot_age_days"],
        "projected_on_hand": d["projected_on_hand"], "open_orders": d["open_orders"],
        "confidence": confidence, "confidence_basis": basis, "flags": d["flags"],
        "headline": _headline(c["sku"], c["name"], d["order_qty"], order_by, ev, as_of, arrival),
        "reasoning_trace": trace, "status": "pending", "as_of": as_of.isoformat(), "replay": replay,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

