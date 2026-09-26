"""
agents/nodes/sales_node.py
---------------------------
Node 2: Sales analysis for retrieved SKUs.

Reads from the synthetic sales history (24 months, daily granularity,
UAE seasonality multipliers baked in). At Phase 2, reads from the CSV
file generated in Phase 1. Phase 3 will migrate to PostgreSQL with
SQLAlchemy async queries.

Engineering decision: daily granularity over weekly.
UAE weekend is Friday–Saturday (not Saturday–Sunday). Weekly aggregation
loses this signal and distorts DSF/Ramadan day-level spike detection.

Trend calculation: simple linear regression slope over the last 30 days.
Not Prophet — Prophet is Phase 3. For trend direction a slope sign is
sufficient and avoids Prophet's cold-start latency on every query.

Performance note: sales CSV loaded once at startup into a pandas DataFrame
and cached in module memory. At SME scale (500 SKUs × 365 days = 182,500
rows) this fits in ~15MB RAM — well within free-tier limits.
At noon scale: PostgreSQL materialised view with async queries.
Documented ceiling, not a hidden assumption.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from ..state import AgentState, SalesSignal

logger = structlog.get_logger(__name__)

SALES_CSV_PATH        = Path(os.getenv("SALES_DATA_PATH", "data/synthetic/sales_history.csv"))
TREND_WINDOW_DAYS     = 30
TREND_STABLE_THRESHOLD = 0.05

UAE_PEAK_PERIODS = {
    "dsf":            ["december", "january"],
    "ramadan":        ["february", "march"],
    "national_day":   ["december"],
    "back_to_school": ["august", "september"],
    "eid":            ["march", "april"],
}


@lru_cache(maxsize=1)
def _load_sales_df() -> pd.DataFrame:
    """
    Load sales history CSV once and cache in memory.
    lru_cache(maxsize=1) ensures a single load regardless of concurrent requests.
    """
    if not SALES_CSV_PATH.exists():
        logger.warning("sales_csv_not_found", path=str(SALES_CSV_PATH))
        return pd.DataFrame(columns=["sku_id", "date", "units_sold"])

    df = pd.read_csv(SALES_CSV_PATH, parse_dates=["date"])
    df["date"]       = pd.to_datetime(df["date"])
    df["month_name"] = df["date"].dt.strftime("%B").str.lower()
    logger.info("sales_df_loaded", rows=len(df), skus=df["sku_id"].nunique())
    return df


def _detect_peak_period(sku_df: pd.DataFrame) -> str | None:
    """
    Identify if a SKU has statistically significant spikes in any UAE
    peak period by comparing period mean to overall mean.
    Returns the peak period name or None if no spike ≥ 20% above baseline.
    """
    if sku_df.empty:
        return None

    overall_mean = sku_df["units_sold"].mean()
    if overall_mean == 0:
        return None

    best_period = None
    best_ratio  = 1.0

    for period, months in UAE_PEAK_PERIODS.items():
        period_df = sku_df[sku_df["month_name"].isin(months)]
        if period_df.empty:
            continue
        ratio = period_df["units_sold"].mean() / overall_mean
        if ratio > best_ratio:
            best_ratio  = ratio
            best_period = period

    return best_period if best_ratio >= 1.2 else None


def _compute_trend(sku_df: pd.DataFrame) -> str:
    """
    Compute trend direction from last 30 days using linear regression slope.
    Returns: "rising" | "falling" | "stable"
    """
    recent = sku_df.nlargest(TREND_WINDOW_DAYS, "date").sort_values("date")
    if len(recent) < 3:
        return "stable"

    x = np.arange(len(recent))
    y = recent["units_sold"].values
    slope            = float(np.polyfit(x, y, 1)[0])
    normalized_slope = slope / max(y.mean(), 1)

    if normalized_slope > TREND_STABLE_THRESHOLD:
        return "rising"
    if normalized_slope < -TREND_STABLE_THRESHOLD:
        return "falling"
    return "stable"


async def sales_analysis_node(state: AgentState) -> dict:
    """
    Node 2: Analyse sales history for each retrieved SKU.

    Runs only for "reorder" and "analysis" intents (routed by conditional
    edge in retail_agent.py). "search" and "trend" queries skip this node.

    Args:
        state: Current AgentState with retrieved_chunks populated

    Returns:
        Partial state dict with sales_signals and trace entry
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
        df      = _load_sales_df()
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

            avg_daily        = float(sku_df["units_sold"].mean())
            trend            = _compute_trend(sku_df)
            peak             = _detect_peak_period(sku_df)
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
            "sales_signals":  signals,
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
