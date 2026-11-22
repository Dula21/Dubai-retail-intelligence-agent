"""
services/reorder_trigger.py
----------------------------
Pure rules for the AUTONOMOUS reorder trigger (Phase 3). No I/O, no agent imports, no LLM.

A SKU triggers a reorder recommendation for one of two reasons, evaluated with no user question:

  RUNWAY  the stock will run out before a replenishment could be arranged.
            remaining_runway = stock_now / recent_daily_rate
            must_order_in    = remaining_runway - supplier_lead_days - ORDER_BUFFER_DAYS
            trigger when must_order_in <= REVIEW_PERIOD_DAYS: the NEXT scheduled review would already be late.
  EVENT   a calendar event that lifts this SKU has its order-by date inside the notice window.
            order_by = event_start - lead_in_days - supplier_lead_days - ORDER_BUFFER_DAYS   (same formula the
            seasonality node reports), trigger when order_by <= as_of + EVENT_NOTICE_DAYS.
            Long "background" seasons (Summer, Winter) never trigger; they are not an ordering deadline.

DECISION: two cheap rules screen every SKU; only triggered SKUs go through the expensive quantity engine.
  Alternative: run the Prophet engine on every SKU every scan (500 fits, minutes of CPU on a free-tier box).
  Trade-off: the screen uses a plain recent-rate runway, which is rougher than the Prophet forecast; it can only
  cause a SKU to be looked at (or missed), never change a quantity. A missed SKU is one whose recent rate understates
  an imminent event; the EVENT rule covers that case independently of the rate.
  Next architectural step at scale: a scheduled batch job writes forecasts to PostgreSQL and the scan reads them.

DECISION: stale stock is a hard fallback for the RUNWAY rule (factual field, architecture principle 10).
  A stock count older than MAX_SNAPSHOT_AGE_DAYS is not used to claim "this SKU runs out in N days"; those SKUs
  are reported as "not assessed", never guessed. The EVENT rule still runs (its quantity is sized from the forecast
  and carries the stale-snapshot flag and the confidence cap).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterable, Optional

import pandas as pd

from . import quantity_engine as qe

REVIEW_PERIOD_DAYS = qe.REVIEW_PERIOD_DAYS          # 14
ORDER_BUFFER_DAYS = qe.ORDER_BUFFER_DAYS            # 5
MAX_SNAPSHOT_AGE_DAYS = qe.MAX_SNAPSHOT_AGE_DAYS    # 7
EVENT_NOTICE_DAYS = 7           # an event trigger fires this many days before its order-by date (and every scan after)
MIN_EVENT_LIFT = 1.15           # same floor the seasonality node uses
BACKGROUND_EVENT_DAYS = 45      # events longer than this are seasons, not deadlines
RATE_WINDOW_DAYS = 28
MIN_RATE_DAYS = 14              # uncensored days needed inside the window to trust a rate


# ── recent sell rate ───────────────────────────────────────────────────────────

def recent_rate(sku_sales: pd.DataFrame, window_days: int = RATE_WINDOW_DAYS,
                min_days: int = MIN_RATE_DAYS, mask_mode: str = "run") -> tuple[Optional[float], int]:
    """
    Mean units/day over the last `window_days` of history, counting only days that are real demand observations
    (stockout-censored days are excluded: they understate demand). Returns (rate, observed_days);
    rate is None when fewer than `min_days` usable days exist (unknown, never a guess).
    """
    if sku_sales.empty:
        return None, 0
    df = sku_sales.sort_values("date")
    stock = df["stockout_occurred"] if "stockout_occurred" in df else pd.Series(0, index=df.index)
    cens = qe.censored_days(df["units_sold"].reset_index(drop=True), stock.reset_index(drop=True), mask_mode).to_numpy()
    tail = df.assign(_c=cens).tail(window_days)
    usable = tail[~tail["_c"]]
    if len(usable) < min_days:
        return None, int(len(usable))
    return float(usable["units_sold"].mean()), int(len(usable))


# ── runway rule ────────────────────────────────────────────────────────────────

@dataclass
class RunwayAssessment:
    status: str                        # "assessed" | "stock_unknown" | "snapshot_stale" | "rate_unknown"
    triggered: bool = False
    rate: Optional[float] = None
    rate_days: int = 0
    stock_now: Optional[float] = None  # snapshot stock minus consumption since the snapshot (floored at 0)
    remaining_runway: Optional[float] = None
    stockout_date: Optional[date] = None
    must_order_in: Optional[float] = None   # days from as_of until the order must be placed (negative = late)
    trace: str = ""


def assess_runway(on_hand: Optional[float], snapshot_date: Optional[date], as_of: date,
                  rate: Optional[float], rate_days: int, lead_days: int) -> RunwayAssessment:
    """The RUNWAY rule. An empty shelf is a trigger even when the rate is unknown."""
    if on_hand is None or on_hand < 0 or snapshot_date is None:
        return RunwayAssessment("stock_unknown", trace="stock unknown (insufficient data): runway rule not applied")
    age = (as_of - snapshot_date).days
    if age > MAX_SNAPSHOT_AGE_DAYS:
        return RunwayAssessment(
            "snapshot_stale", stock_now=None,
            trace=f"stock figure is {age} days old (limit {MAX_SNAPSHOT_AGE_DAYS}); runway rule not applied "
                  f"(insufficient data, not a guess)")
    if on_hand == 0:
        return RunwayAssessment(
            "assessed", True, rate, rate_days, 0.0, 0.0, as_of, 0.0,
            "stock is 0: already out of stock; an order placed today arrives "
            f"in {lead_days} days")
    if rate is None or rate <= 0:
        return RunwayAssessment("rate_unknown", rate=rate, rate_days=rate_days, stock_now=float(on_hand),
                                trace="no usable recent sales rate; runway rule not applied")
    stock_now = max(0.0, float(on_hand) - rate * max(age, 0))
    remaining = stock_now / rate
    must_order_in = remaining - lead_days - ORDER_BUFFER_DAYS
    triggered = must_order_in <= REVIEW_PERIOD_DAYS
    return RunwayAssessment(
        "assessed", triggered, rate, rate_days, stock_now, remaining, as_of + timedelta(days=int(remaining)),
        must_order_in,
        f"stock {stock_now:.0f} at {rate:.1f}/day = {remaining:.1f} days of runway; lead time {lead_days}d + "
        f"{ORDER_BUFFER_DAYS}d buffer -> must order in {must_order_in:.0f} days "
        f"({'at or inside' if triggered else 'outside'} the {REVIEW_PERIOD_DAYS}-day review period)")


# ── event rule ─────────────────────────────────────────────────────────────────

@dataclass
class EventTrigger:
    name: str
    start: date
    end: date
    lead_in_days: int
    lift: float
    order_by: date
    days_to_order_by: int


def next_event_trigger(events: Iterable, lift_of: Callable[[object], float], as_of: date, lead_days: int,
                       notice_days: int = EVENT_NOTICE_DAYS) -> Optional[EventTrigger]:
    """
    The earliest-starting event that lifts this SKU and whose order-by date is inside the notice window.
    `events` need .name .start .end .lead_in_days; `lift_of(event)` is this SKU's lift for it (1.0 = none).
    Events already over, seasons, and events an order placed today could not reach in time (it would arrive after
    the event ends) are skipped.
    """
    for e in sorted(events, key=lambda ev: ev.start):
        if e.end < as_of or (e.end - e.start).days > BACKGROUND_EVENT_DAYS:
            continue
        lift = lift_of(e)
        if lift < MIN_EVENT_LIFT:
            continue
        order_by = e.start - timedelta(days=e.lead_in_days + lead_days + ORDER_BUFFER_DAYS)
        if order_by > as_of + timedelta(days=notice_days):
            continue
        if as_of + timedelta(days=lead_days) > e.end:       # would arrive after the event: nothing to order for
            continue
        return EventTrigger(e.name, e.start, e.end, e.lead_in_days, round(lift, 2), order_by, (order_by - as_of).days)
    return None


# ── presentation helpers ───────────────────────────────────────────────────────

def urgency(days_to_order_by: float) -> str:
    """late: the order-by date has passed; now: today; upcoming: inside the notice window."""
    if days_to_order_by < 0:
        return "late"
    if days_to_order_by < 1:
        return "now"
    return "upcoming"


def rec_id(sku_id: str, trigger: str, event: Optional[str], as_of: date) -> str:
    """Deterministic id: re-scanning the same day gives the same id (idempotent), a new day a new one."""
    raw = f"{sku_id}|{trigger}|{event or ''}|{as_of.isoformat()}"
    return "rec_" + hashlib.sha1(raw.encode()).hexdigest()[:12]
