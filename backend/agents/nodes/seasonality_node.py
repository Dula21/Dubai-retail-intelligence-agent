"""
agents/nodes/seasonality_node.py
----------------------------------
Node 4: UAE retail seasonality detection.

Phase 2: static lookup table (fast, deterministic, no external calls).
Phase 3: replaced by Prophet predictions trained on 24-month synthetic
         history with UAE demand multipliers baked in from Phase 1.

Alert level logic:
  "critical" → event within 7 days
  "act"      → event within 21 days
  "watch"    → event within 45 days
  "none"     → no upcoming event in window

Engineering decision: static table over live Hijri API at Phase 2.
Options considered:
  1. Static Gregorian approximations (chosen) — fast, offline, testable
  2. hijri-converter library — accurate, adds dependency, requires date math
  3. Live Hijri API — accurate, adds network dependency, breaks in offline/CI

Known limitation: Hijri calendar dates are approximated in Gregorian.
Production fix: hijri-converter library. Documented in README.
"""

from __future__ import annotations

from datetime import date

import structlog

from ..state import AgentState, SeasonalitySignal

logger = structlog.get_logger(__name__)

UAE_RETAIL_EVENTS = [
    {
        "name":       "Back-to-School",
        "start":      date(2026, 8, 20),
        "end":        date(2026, 9, 10),
        "multiplier": 1.6,
        "categories": ["fashion", "electronics", "stationery"],
    },
    {
        "name":       "National Day",
        "start":      date(2026, 12, 1),
        "end":        date(2026, 12, 4),
        "multiplier": 1.5,
        "categories": ["fashion", "home", "food", "beauty"],
    },
    {
        "name":       "DSF",
        "start":      date(2026, 12, 15),
        "end":        date(2027, 1, 15),
        "multiplier": 2.1,
        "categories": ["fashion", "electronics", "home", "beauty"],
    },
    {
        "name":       "Ramadan",
        "start":      date(2027, 2, 18),
        "end":        date(2027, 3, 19),
        "multiplier": 1.8,
        "categories": ["food", "beauty", "fashion", "home"],
    },
    {
        "name":       "Eid Al Fitr",
        "start":      date(2027, 3, 20),
        "end":        date(2027, 3, 23),
        "multiplier": 1.9,
        "categories": ["fashion", "food", "beauty"],
    },
    {
        "name":       "Eid Al Adha",
        "start":      date(2027, 6, 6),
        "end":        date(2027, 6, 9),
        "multiplier": 1.7,
        "categories": ["fashion", "food", "home"],
    },
]

ALERT_THRESHOLDS   = {"critical": 7, "act": 21, "watch": 45}
ACTIVE_WINDOW_BEFORE = 7
ACTIVE_WINDOW_AHEAD  = 45


async def seasonality_check_node(state: AgentState) -> dict:
    """
    Node 4: Detect upcoming UAE retail demand events.

    Category-aware: prefers events that match the categories of retrieved
    products, falling back to the closest event by date.

    Args:
        state: Current AgentState

    Returns:
        Partial state dict with seasonality_signal and trace entry
    """
    today = date.today()

    retrieved_categories = {
        c.get("category", "").lower()
        for c in state.get("retrieved_chunks", [])
    }

    best_event:      dict | None = None
    best_days_until: int  | None = None

    for event in UAE_RETAIL_EVENTS:
        days_to_start = (event["start"] - today).days
        in_window     = -ACTIVE_WINDOW_BEFORE <= days_to_start <= ACTIVE_WINDOW_AHEAD

        if not in_window:
            continue

        event_categories = set(event.get("categories", []))
        category_match   = bool(retrieved_categories & event_categories)

        if best_event is None:
            best_event      = event
            best_days_until = days_to_start
        elif category_match and not bool(
            set(best_event.get("categories", [])) & retrieved_categories
        ):
            best_event      = event
            best_days_until = days_to_start
        elif abs(days_to_start) < abs(best_days_until):
            best_event      = event
            best_days_until = days_to_start

    if best_event is None:
        signal = SeasonalitySignal(
            upcoming_event=None,
            days_until_event=None,
            expected_demand_multiplier=1.0,
            alert_level="none",
        )
        trace = "Seasonality: no UAE retail events in 45-day window — baseline demand"
    else:
        days = best_days_until

        if days <= ALERT_THRESHOLDS["critical"]:
            alert = "critical"
        elif days <= ALERT_THRESHOLDS["act"]:
            alert = "act"
        else:
            alert = "watch"

        if days < 0:
            alert = "act" if abs(days) < 14 else "watch"

        signal = SeasonalitySignal(
            upcoming_event=best_event["name"],
            days_until_event=days,
            expected_demand_multiplier=best_event["multiplier"],
            alert_level=alert,
        )
        trace = (
            f"Seasonality: {best_event['name']} in {days} days "
            f"(multiplier={best_event['multiplier']}x, alert={alert}, "
            f"category_match={bool(retrieved_categories & set(best_event.get('categories', [])))})"
        )

    logger.info(
        "seasonality_node_complete",
        upcoming_event=signal["upcoming_event"],
        days_until=signal["days_until_event"],
        alert=signal["alert_level"],
    )

    return {
        "seasonality_signal": signal,
        "reasoning_trace":    state["reasoning_trace"] + [trace],
    }
