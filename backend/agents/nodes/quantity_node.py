"""
agents/nodes/quantity_node.py
------------------------------
Node 5 (Phase 3): the deterministic ORDER-QUANTITY for every candidate SKU. No LLM.

Runs after seasonality (it needs the event and its dates) and before recommendation, which only EXPLAINS the
numbers computed here. The arithmetic and its caveats live in services/quantity_engine.py; this node gathers inputs:

  history     sales_node._load_sales_df()              (censored sales + stockout flag)
  stock       inventory_node._load_inventory_df()      (snapshot; it has NO date column, see below)
  calendar    data/synthetic/uae_retail_calendar.csv   (the same file the seasonality node reads)
  calibration data/quantity_calibration.json           (safety multiplier + measured coverage, from eval/backtest_quantity.py)

Decisions
  * Snapshot date: inventory_snapshot.csv carries no date. The generator writes it as the END STATE of the sales
    simulation, so its date is the last sales date. The trace says so. A real feed must timestamp its snapshot.
  * Window: the event from the seasonality signal when there is one (alert not "none"), else the generic
    lead-time-plus-review-period window. Lead time per SKU from the catalogue; assumed (and flagged) if missing.
  * Fits are LAZY and CACHED (bounded LRU, MAX_CACHED_FORECASTS): a Prophet model per SKU is fast (~0.1-0.5 s) but
    500 resident models would not fit the 4 GB container next to the embedding model. Fits run in a worker thread so
    the event loop stays free, at most FIT_CONCURRENCY at a time. Next architectural step at scale: a scheduled
    batch job writes forecasts to PostgreSQL and the request path only reads them.
  * Nothing here guesses: unknown stock, too little history or no calibration give order_qty = None and a flag.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

from services import quantity_engine as qe

from .. import catalogue
from ..state import AgentState
from . import inventory_node, sales_node, seasonality_node as sn

logger = structlog.get_logger(__name__)

CALIBRATION_PATH = Path(os.getenv("QUANTITY_CALIBRATION_PATH", "data/quantity_calibration.json"))
MASK_MODE = os.getenv("QUANTITY_MASK_MODE", "run")
MAX_CACHED_FORECASTS = int(os.getenv("QUANTITY_MAX_CACHED_FORECASTS", "64"))
FIT_CONCURRENCY = int(os.getenv("QUANTITY_FIT_CONCURRENCY", "2"))
MAX_SKUS_PER_QUERY = 8              # same cap the event-candidates node applies

_forecasts: "OrderedDict[tuple, Optional[qe.Forecast]]" = OrderedDict()
_fit_gate: Optional[asyncio.Semaphore] = None


# ── inputs ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_calibration() -> Optional[dict]:
    """Calibration file as a dict with a SafetyMultiplier under 'safety_obj', or None when missing / unreadable."""
    if not CALIBRATION_PATH.exists():
        logger.warning("quantity_calibration_not_found", path=str(CALIBRATION_PATH))
        return None
    try:
        raw = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        raw["safety_obj"] = qe.SafetyMultiplier.from_dict(raw["safety"])
        return raw
    except (OSError, ValueError, KeyError) as exc:
        logger.error("quantity_calibration_unreadable", error=str(exc))
        return None


@lru_cache(maxsize=1)
def _calendar_df() -> Optional[pd.DataFrame]:
    if not sn.CALENDAR_PATH.exists():
        return None
    return pd.read_csv(sn.CALENDAR_PATH, parse_dates=["start_date", "end_date"])


def clear_caches() -> None:
    """For tests and after a data upload."""
    _forecasts.clear()
    load_calibration.cache_clear()
    _calendar_df.cache_clear()


def _today() -> date:
    return sn._today()


def _lead_days(chunk: dict) -> tuple[int, bool]:
    for raw in (chunk.get("supplier_lead_days"), catalogue.get_row(chunk["sku_id"]).get("supplier_lead_days")):
        try:
            return int(float(raw)), False
        except (TypeError, ValueError):
            continue
    return sn.DEFAULT_LEAD_TIME_DAYS, True


def _event_for_window(signal: dict) -> Optional[dict]:
    """The seasonality signal's event, with its lead-in days read from the calendar (the signal does not carry them)."""
    if not signal or not signal.get("upcoming_event") or signal.get("alert_level", "none") == "none":
        return None
    name, start, end = signal["upcoming_event"], signal.get("event_start"), signal.get("event_end")
    if not (start and end):
        return None
    events, _ = sn._load_events()
    lead_in = next((e.lead_in_days for e in events if e.name == name and e.start.isoformat() == start), 0)
    return {"start": date.fromisoformat(start), "end": date.fromisoformat(end), "lead_in_days": lead_in}


# ── forecasts (lazy, cached, off the event loop) ───────────────────────────────

def _fit_sync(sku_id: str, sales_df: pd.DataFrame, calendar: pd.DataFrame) -> Optional[qe.Forecast]:
    sku = sales_df[sales_df["sku_id"] == sku_id]
    if sku.empty:
        return None
    series = qe.demand_series(sku, MASK_MODE)
    return qe.fit_prophet(series, calendar, MASK_MODE)


async def _get_forecast(sku_id: str, sales_df: pd.DataFrame, calendar: pd.DataFrame) -> Optional[qe.Forecast]:
    global _fit_gate
    key = (sku_id, str(sales_df["date"].max().date()), MASK_MODE)
    if key in _forecasts:
        _forecasts.move_to_end(key)
        return _forecasts[key]
    if _fit_gate is None:
        _fit_gate = asyncio.Semaphore(FIT_CONCURRENCY)
    async with _fit_gate:
        if key in _forecasts:                                   # another request fitted it while we waited
            return _forecasts[key]
        fc = await asyncio.to_thread(_fit_sync, sku_id, sales_df, calendar)
    _forecasts[key] = fc
    while len(_forecasts) > MAX_CACHED_FORECASTS:
        _forecasts.popitem(last=False)
    return fc


# ── node ───────────────────────────────────────────────────────────────────────

async def quantity_node(state: AgentState) -> dict:
    """Compute an order quantity (or an explicit 'no quantity' with the reason) for each candidate SKU."""
    chunks = list(state.get("retrieved_chunks", []))[:MAX_SKUS_PER_QUERY]
    trace = list(state["reasoning_trace"])
    if not chunks:
        return {"quantity_results": [], "reasoning_trace": trace + ["Quantity engine: no candidate SKUs; skipped"]}

    calendar = _calendar_df()
    calibration = load_calibration()
    if calendar is None:
        return {"quantity_results": [], "reasoning_trace": trace + [
            "Quantity engine: calendar file not found; no quantities computed"]}

    try:
        sales_df = sales_node._load_sales_df()
        inv_df = inventory_node._load_inventory_df()
        if sales_df.empty:
            return {"quantity_results": [], "reasoning_trace": trace + [
                "Quantity engine: no sales history loaded; no quantities computed"]}
        snapshot_date = sales_df["date"].max().date()        # the snapshot is the end state of the sales history
        as_of = _today()
        signal = state.get("seasonality_signal") or {}
        event = _event_for_window(signal)

        results = []
        for chunk in chunks:
            sku = chunk["sku_id"]
            lead, lead_assumed = _lead_days(chunk)
            window = qe.cover_window(as_of, lead, event)
            row = inv_df[inv_df["sku_id"] == sku]
            on_hand = None
            if not row.empty:
                raw = inventory_node._read_number(row.iloc[0], inventory_node.STOCK_COLUMNS)
                on_hand = None if raw is None else raw
            forecast = await _get_forecast(sku, sales_df, calendar)
            res = qe.order_quantity(
                sku_id=sku, forecast=forecast, window=window,
                safety=calibration["safety_obj"] if calibration else None,
                on_hand=on_hand, snapshot_date=snapshot_date, as_of=as_of,
                open_orders=None,                             # no dataset holds open purchase orders: unknown, flagged
                calendar=calendar,
                coverage_table=(calibration or {}).get("coverage"),
                validated_horizon_days=(calibration or {}).get("validated_horizon_days", qe.VALIDATED_HORIZON_DAYS),
            )
            if lead_assumed:
                res.flags.append("lead_time_assumed")
            d = res.to_dict()
            # A quantity for a window farther out than the backtest validated is a planning estimate, not an order.
            d["indicative_only"] = "beyond_validated_horizon" in res.flags
            d["order_by_date"] = signal.get("order_by_date") if event else None
            results.append(d)

        made = sum(1 for r in results if r["order_qty"] is not None)
        header = (f"Quantity engine: {made} of {len(results)} SKUs have an order quantity "
                  f"(window: {'event ' + event_name(event, signal) if event else 'lead time + review period'}; "
                  f"stock snapshot dated {snapshot_date} = last sales date, as of {as_of})")
        lines = [header] + [t for r in results for t in r["trace"]]
        logger.info("quantity_node_complete", skus=len(results), with_quantity=made, as_of=str(as_of))
        return {"quantity_results": results, "reasoning_trace": trace + lines}

    except Exception as exc:                                  # degrade like the other nodes: no quantity, no crash
        logger.error("quantity_node_error", error=str(exc))
        return {"quantity_results": [], "error": f"Quantity engine failed: {type(exc).__name__}",
                "reasoning_trace": trace + [f"Quantity engine error: {type(exc).__name__}; no quantities computed"]}


def event_name(event: Optional[dict], signal: dict) -> str:
    return str(signal.get("upcoming_event") or "") if event else ""
