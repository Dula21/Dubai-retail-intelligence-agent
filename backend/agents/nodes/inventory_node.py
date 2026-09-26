"""
agents/nodes/inventory_node.py
--------------------------------
Node 3: Inventory position check and runway calculation.

Reads current stock levels from the synthetic inventory snapshot
(generated in Phase 1: 500 SKUs with current_stock, reorder_point,
supplier_lead_time_days).

Runway calculation:
  runway_days = current_stock / avg_daily_units

  A runway below (CRITICAL_RUNWAY_DAYS + lead_time + buffer) = is_critical.
  Same logic as Logistics Oracle's critical runway override — consistent
  pattern across the portfolio.

Engineering decision: hard fallback for inventory data.
Inventory count is a factual field. Returning a wrong stock number
is unrecoverable — a retailer immediately acts on it. If the inventory
CSV lookup fails for a SKU, we return sentinel -1 rather than guessing.

Scale ceiling: inventory CSV loaded once into memory (~50KB for 500 SKUs).
At noon scale: async PostgreSQL with row-level locking on writes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import structlog

from ..state import AgentState, InventorySignal, SalesSignal

logger = structlog.get_logger(__name__)

INVENTORY_CSV_PATH    = Path(os.getenv("INVENTORY_DATA_PATH", "data/synthetic/inventory_snapshot.csv"))
CRITICAL_RUNWAY_DAYS  = 14
REORDER_LEAD_TIME_BUFFER = 7


@lru_cache(maxsize=1)
def _load_inventory_df() -> pd.DataFrame:
    """Load inventory snapshot CSV once and cache in memory."""
    if not INVENTORY_CSV_PATH.exists():
        logger.warning("inventory_csv_not_found", path=str(INVENTORY_CSV_PATH))
        return pd.DataFrame(
            columns=["sku_id", "current_stock", "reorder_point", "supplier_lead_time_days"]
        )

    df = pd.read_csv(INVENTORY_CSV_PATH)
    logger.info("inventory_df_loaded", rows=len(df), skus=df["sku_id"].nunique())
    return df


def _calculate_runway(current_stock: int, avg_daily_units: float) -> float:
    """
    Calculate days until stockout at current sales rate.
    Returns float('inf') if avg_daily_units is zero (dead stock, not urgent).
    """
    if avg_daily_units <= 0:
        return float("inf")
    return round(current_stock / avg_daily_units, 1)


def _sku_sales_map(sales_signals: list[SalesSignal]) -> dict[str, float]:
    """Build a sku_id → avg_daily_units lookup from sales signals."""
    return {s["sku_id"]: s["avg_daily_units"] for s in sales_signals}


async def inventory_check_node(state: AgentState) -> dict:
    """
    Node 3: Check inventory position for each retrieved SKU.

    Cross-references retrieved chunks with sales signals from Node 2
    to compute runway. If sales signals are missing (Node 2 was skipped),
    uses a conservative default daily rate of 1.0 units/day.

    Args:
        state: Current AgentState with retrieved_chunks and sales_signals

    Returns:
        Partial state dict with inventory_signals and trace entry
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "inventory_signals": [],
            "reasoning_trace": state["reasoning_trace"] + [
                "Inventory check: no chunks — skipped"
            ],
        }

    sales_map = _sku_sales_map(state.get("sales_signals", []))

    try:
        inv_df  = _load_inventory_df()
        signals: list[InventorySignal] = []

        for chunk in chunks:
            sku_id = chunk["sku_id"]
            row    = inv_df[inv_df["sku_id"] == sku_id]

            if row.empty:
                logger.warning("sku_not_in_inventory", sku_id=sku_id)
                # Hard fallback: sentinel -1, never guess a stock level
                signals.append(InventorySignal(
                    sku_id=sku_id,
                    current_stock=-1,
                    reorder_point=-1,
                    runway_days=-1.0,
                    is_critical=False,   # unknown ≠ critical
                ))
                continue

            r             = row.iloc[0]
            current_stock = int(r.get("current_stock", 0))
            reorder_point = int(r.get("reorder_point", 0))
            lead_time     = int(r.get("supplier_lead_time_days", 7))

            avg_daily = sales_map.get(sku_id, 1.0)
            runway    = _calculate_runway(current_stock, avg_daily)

            effective_threshold = CRITICAL_RUNWAY_DAYS + lead_time + REORDER_LEAD_TIME_BUFFER
            is_critical = (
                runway != float("inf") and
                runway <= effective_threshold
            )

            signals.append(InventorySignal(
                sku_id=sku_id,
                current_stock=current_stock,
                reorder_point=reorder_point,
                runway_days=runway if runway != float("inf") else 9999.0,
                is_critical=is_critical,
            ))

            logger.debug("sku_inventory_checked", sku_id=sku_id,
                         current_stock=current_stock, runway=runway, is_critical=is_critical)

        critical_count = sum(1 for s in signals if s["is_critical"])
        unknown_count  = sum(1 for s in signals if s["current_stock"] == -1)

        trace = (
            f"Inventory check: {len(signals)} SKUs checked, "
            f"critical={critical_count}, "
            f"data_missing={unknown_count}, "
            f"min_runway={min((s['runway_days'] for s in signals if s['runway_days'] > 0), default='N/A')}"
        )

        return {
            "inventory_signals": signals,
            "reasoning_trace":   state["reasoning_trace"] + [trace],
        }

    except Exception as exc:
        logger.error("inventory_node_error", error=str(exc))
        return {
            "inventory_signals": [],
            "error": f"Inventory check failed: {exc}",
            "reasoning_trace": state["reasoning_trace"] + [
                f"Inventory check error: {exc} — continuing with available data"
            ],
        }
