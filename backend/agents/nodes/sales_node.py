"""
agents/nodes/sales_node.py
---------------------------
Node 2: Sales analysis for retrieved SKUs.

Reads the synthetic sales history (daily granularity). Data is SYNTHETIC
(see data/generate_synthetic.py), generated from the UAE retail calendar.

Peak-period detection is driven by the calendar file
(data/synthetic/uae_retail_calendar.csv): Ramadan, Eid, Diwali, Christmas,
White Friday, DSF, Back to School, etc. For each event we compare mean daily
units inside the event windows against mean units on days outside ALL event
windows, and report the strongest event with lift >= 1.2.

Fallback: if the calendar file is missing, the old month-name table is used
so the agent still works (and says so in the logs).

Engineering decision: daily granularity over weekly. UAE weekend has been
Saturday-Sunday since 2022; weekly aggregation hides day-level spikes.

Trend: slope sign over the last 30 days. Prophet comes later; slope is enough
for direction and avoids cold-start latency on every query.

Scale ceiling: the CSV is loaded once into memory (~365k rows, tens of MB).
Fine at SME scale. At noon scale: PostgreSQL materialised view + async queries.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from ..state import AgentState, SalesSignal

logger = structlog.get_logger(__name__)

SALES_CSV_PATH    = Path(os.getenv("SALES_DATA_PATH", "data/synthetic/sales_history.csv"))
CALENDAR_CSV_PATH = Path(os.getenv("CALENDAR_DATA_PATH", "data/synthetic/uae_retail_calendar.csv"))
TREND_WINDOW_DAYS      = 30
TREND_STABLE_THRESHOLD = 0.05
PEAK_MIN_LIFT          = 1.2
PEAK_MIN_EVENT_DAYS    = 7

# Fallback only (used when the calendar file is missing).
UAE_PEAK_PERIODS = {
    "dsf":            ["december", "january"],
    "ramadan":        ["february", "march"],
    "national_day":   ["december"],
    "back_to_school": ["august", "september"],
    "eid":            ["march", "april"],
}


def _slug(name: str) -> str:
    """'Eid al-Fitr' -> 'eid_al_fitr'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@lru_cache(maxsize=1)
def _load_sales_df() -> pd.DataFrame:
    """Load sales history once and cache in memory."""
    if not SALES_CSV_PATH.exists():
        logger.warning("sales_csv_not_found", path=str(SALES_CSV_PATH))
        return pd.DataFrame(columns=["sku_id", "date", "units_sold"])

    df = pd.read_csv(SALES_CSV_PATH, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    df["month_name"] = df["date"].dt.strftime("%B").str.lower()
    logger.info("sales_df_loaded", rows=len(df), skus=df["sku_id"].nunique())
    return df


@lru_cache(maxsize=1)
def _load_event_windows() -> dict[str, frozenset]:
    """
    event slug -> set of dates (event start..end, all years in the calendar).
    Empty dict if the calendar file is missing (triggers month-name fallback).
    """
    if not CALENDAR_CSV_PATH.exists():
        logger.warning("calendar_csv_not_found", path=str(CALENDAR_CSV_PATH))
        return {}

    cal = pd.read_csv(CALENDAR_CSV_PATH, parse_dates=["start_date", "end_date"])
    windows: dict[str, set] = {}
    for _, row in cal.iterrows():
        days = pd.date_range(row["start_date"], row["end_date"], freq="D")
        windows.setdefault(_slug(str(row["event"])), set()).update(days)
    logger.info("calendar_loaded", events=len(windows))
    return {k: frozenset(v) for k, v in windows.items()}


def _detect_peak_period_calendar(sku_df: pd.DataFrame, windows: dict[str, frozenset]) -> str | None:
    """Strongest calendar event by lift vs days outside every event window."""
    all_event_days = set().union(*windows.values()) if windows else set()
    outside = sku_df[~sku_df["date"].isin(all_event_days)]
    baseline = outside["units_sold"].mean() if not outside.empty else sku_df["units_sold"].mean()
    if not baseline or baseline <= 0:
        return None

    best_event, best_ratio = None, 1.0
    for event, days in windows.items():
        inside = sku_df[sku_df["date"].isin(days)]
        if len(inside) < PEAK_MIN_EVENT_DAYS:
            continue  # event not (sufficiently) covered by the sales history
        ratio = inside["units_sold"].mean() / baseline
        if ratio > best_ratio:
            best_event, best_ratio = event, ratio

    return best_event if best_ratio >= PEAK_MIN_LIFT else None


def _detect_peak_period_months(sku_df: pd.DataFrame) -> str | None:
    """Old month-name fallback (used only if the calendar file is missing)."""
    overall_mean = sku_df["units_sold"].mean()
    if overall_mean == 0:
        return None

    best_period, best_ratio = None, 1.0
    for period, months in UAE_PEAK_PERIODS.items():
        period_df = sku_df[sku_df["month_name"].isin(months)]
        if period_df.empty:
            continue
        ratio = period_df["units_sold"].mean() / overall_mean
        if ratio > best_ratio:
            best_ratio, best_period = ratio, period
    return best_period if best_ratio >= PEAK_MIN_LIFT else None


def _detect_peak_period(sku_df: pd.DataFrame) -> str | None:
    """Calendar-driven peak detection, month-name fallback if no calendar."""
    if sku_df.empty:
        return None
    windows = _load_event_windows()
    if windows:
        return _detect_peak_period_calendar(sku_df, windows)
    return _detect_peak_period_months(sku_df)


def _compute_trend(sku_df: pd.DataFrame) -> str:
    """Trend direction from the last 30 days: 'rising' | 'falling' | 'stable'."""
    recent = sku_df.nlargest(TREND_WINDOW_DAYS, "date").sort_values("date")
    if len(recent) < 3:
        return "stable"

    x = np.arange(len(recent))
    y = recent["units_sold"].values
    slope = float(np.polyfit(x, y, 1)[0])
    normalized_slope = slope / max(y.mean(), 1)

    if normalized_slope > TREND_STABLE_THRESHOLD:
        return "rising"
    if normalized_slope < -TREND_STABLE_THRESHOLD:
        return "falling"
    return "stable"


async def sales_analysis_node(state: AgentState) -> dict:
    """
    Node 2: analyse sales history for each retrieved SKU.

    Runs only for "reorder" and "analysis" intents (conditional edge in
    retail_agent.py). Returns sales_signals plus a trace entry.
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "sales_signals": [],
            "reasoning_trace": state["reasoning_trace"] + [
                "Sales analysis: no chunks to analyse — skipped"
            ],
        }

    try:
        df = _load_sales_df()
        signals: list[SalesSignal] = []

        for chunk in chunks:
            sku_id = chunk["sku_id"]
            sku_df = df[df["sku_id"] == sku_id].copy()

            if sku_df.empty:
                logger.warning("sku_not_in_sales", sku_id=sku_id)
                signals.append(SalesSignal(
                    sku_id=sku_id,
                    avg_daily_units=0.0,
                    trend_direction="stable",
                    peak_period=None,
                    confidence=0.0,
                ))
                continue

            avg_daily = float(sku_df["units_sold"].mean())
            trend = _compute_trend(sku_df)
            peak = _detect_peak_period(sku_df)
            data_completeness = min(len(sku_df) / 365.0, 1.0)

            signals.append(SalesSignal(
                sku_id=sku_id,
                avg_daily_units=round(avg_daily, 2),
                trend_direction=trend,
                peak_period=peak,
                confidence=round(data_completeness * 0.9, 2),
            ))

            logger.debug("sku_sales_analysed", sku_id=sku_id, avg_daily=avg_daily,
                         trend=trend, peak=peak)

        trace = (
            f"Sales analysis: analysed {len(signals)} SKUs, "
            f"rising={sum(1 for s in signals if s['trend_direction'] == 'rising')}, "
            f"falling={sum(1 for s in signals if s['trend_direction'] == 'falling')}, "
            f"peak_periods={[s['peak_period'] for s in signals if s['peak_period']]}"
        )

        return {
            "sales_signals": signals,
            "reasoning_trace": state["reasoning_trace"] + [trace],
        }
    except Exception as exc:
        logger.error("sales_node_error", error=str(exc))
        return {
            "sales_signals": [],
            "error": f"Sales analysis failed: {exc}",
            "reasoning_trace": state["reasoning_trace"] + [
                f"Sales analysis error: {exc} — continuing with available data"
            ],
        }