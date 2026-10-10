"""
agents/nodes/inventory_node.py
--------------------------------
Node 3: Inventory position check and runway calculation.

Reads current stock levels from the synthetic inventory snapshot
(data/synthetic/inventory_snapshot.csv). Columns used:
  sku_id, current_stock_units, reorder_point_units, days_of_stock_left
(the older names current_stock / reorder_point are also accepted).
Data is SYNTHETIC (see data/generate_synthetic.py).

Runway calculation:
  runway_days = current_stock / avg_daily_units          (sales signal)
  fallback    = days_of_stock_left from the snapshot     (no sales signal)
  otherwise   = -1.0 (unknown) -- never a guessed number

  A runway below (CRITICAL_RUNWAY_DAYS + lead_time + buffer) = is_critical.
  Same logic as Logistics Oracle's critical runway override.

Engineering decision: hard fallback for inventory data (factual field).
Returning a wrong stock number is unrecoverable -- a retailer acts on it
immediately. If a SKU, column or value is missing we return sentinel -1
("insufficient data"), never 0 and never a guess. Unknown is not critical.

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

INVENTORY_CSV_PATH       = Path(os.getenv("INVENTORY_DATA_PATH", "data/synthetic/inventory_snapshot.csv"))
CRITICAL_RUNWAY_DAYS     = 14
REORDER_LEAD_TIME_BUFFER = 7
DEFAULT_LEAD_TIME_DAYS   = 7

STOCK_COLUMNS         = ("current_stock_units", "current_stock")
REORDER_POINT_COLUMNS = ("reorder_point_units", "reorder_point")
DAYS_LEFT_COLUMNS     = ("days_of_stock_left",)
LEAD_TIME_COLUMNS     = ("supplier_lead_days", "supplier_lead_time_days")


@lru_cache(maxsize=1)
def _load_inventory_df() -> pd.DataFrame:
    """Load inventory snapshot CSV once and cache in memory."""
    if not INVENTORY_CSV_PATH.exists():
        logger.warning("inventory_csv_not_found", path=str(INVENTORY_CSV_PATH))
        return pd.DataFrame(columns=["sku_id", *STOCK_COLUMNS, *REORDER_POINT_COLUMNS])

    df = pd.read_csv(INVENTORY_CSV_PATH)
    logger.info("inventory_df_loaded", rows=len(df), skus=df["sku_id"].nunique())
    return df


def _read_number(row, names: tuple[str, ...]) -> float | None:
    """
    First column in `names` that exists and holds a real value, else None.
    Missing column or empty value is None -- never silently 0.
    """
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    return None


def _lead_time_days(row, chunk: dict) -> int:
    """Lead time from the inventory row, then chunk metadata, then default."""
    value = _read_number(row, LEAD_TIME_COLUMNS)
    if value is not None:
        return int(value)
    meta = chunk.get("metadata") or {}
    for name in LEAD_TIME_COLUMNS:
        for source in (chunk, meta):
            raw = source.get(name)
            if raw is not None and str(raw).strip() != "":
                try:
                    return int(float(raw))
                except (TypeError, ValueError):
                    pass
    return DEFAULT_LEAD_TIME_DAYS


def _calculate_runway(current_stock: int, avg_daily_units: float) -> float:
    """
    Days until stockout at the given sales rate.
    Returns float('inf') if avg_daily_units is zero (dead stock, not urgent).
    """
    if avg_daily_units <= 0:
        return float("inf")
    return round(current_stock / avg_daily_units, 1)


def _sku_sales_map(sales_signals: list[SalesSignal]) -> dict[str, float]:
    """Build a sku_id -> avg_daily_units lookup from sales signals."""
    return {s["sku_id"]: s["avg_daily_units"] for s in sales_signals}


def _unknown_signal(sku_id: str, current_stock: int = -1, reorder_point: int = -1) -> InventorySignal:
    """Sentinel signal: insufficient data. Unknown is not critical."""
    return InventorySignal(
        sku_id=sku_id,
        current_stock=current_stock,
        reorder_point=reorder_point,
        runway_days=-1.0,
        is_critical=False,
    )


async def inventory_check_node(state: AgentState) -> dict:
    """
    Node 3: Check inventory position for each retrieved SKU.

    Cross-references retrieved chunks with sales signals from Node 2 to
    compute runway. With no sales signal for a SKU, falls back to the
    snapshot's days_of_stock_left; if that is missing too, runway is
    unknown (-1) rather than a guess.

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
        inv_df = _load_inventory_df()
        signals: list[InventorySignal] = []

        for chunk in chunks:
            sku_id = chunk["sku_id"]
            row = inv_df[inv_df["sku_id"] == sku_id]

            if row.empty:
                logger.warning("sku_not_in_inventory", sku_id=sku_id)
                signals.append(_unknown_signal(sku_id))
                continue

            r = row.iloc[0]
            stock_raw = _read_number(r, STOCK_COLUMNS)
            rop_raw = _read_number(r, REORDER_POINT_COLUMNS)

            if stock_raw is None:
                logger.warning("stock_value_missing", sku_id=sku_id)
                signals.append(_unknown_signal(
                    sku_id, reorder_point=int(rop_raw) if rop_raw is not None else -1
                ))
                continue

            current_stock = int(stock_raw)
            reorder_point = int(rop_raw) if rop_raw is not None else -1
            lead_time = _lead_time_days(r, chunk)

            # Runway: sales-signal rate first, snapshot value second, else unknown.
            if sku_id in sales_map:
                runway = _calculate_runway(current_stock, sales_map[sku_id])
            else:
                days_left = _read_number(r, DAYS_LEFT_COLUMNS)
                runway = round(days_left, 1) if days_left is not None else -1.0

            effective_threshold = CRITICAL_RUNWAY_DAYS + lead_time + REORDER_LEAD_TIME_BUFFER
            is_critical = (
                runway >= 0
                and runway != float("inf")
                and runway <= effective_threshold
            )

            signals.append(InventorySignal(
                sku_id=sku_id,
                current_stock=current_stock,
                reorder_point=reorder_point,
                runway_days=runway if runway != float("inf") else 9999.0,
                is_critical=is_critical,
            ))

            logger.debug("sku_inventory_checked", sku_id=sku_id,
                         current_stock=current_stock, runway=runway,
                         lead_time=lead_time, is_critical=is_critical)

        critical_count = sum(1 for s in signals if s["is_critical"])
        unknown_count = sum(1 for s in signals if s["current_stock"] == -1)
        known_runways = [s["runway_days"] for s in signals if s["runway_days"] >= 0]
        min_runway = min(known_runways) if known_runways else "N/A"

        trace = (
            f"Inventory check: {len(signals)} SKUs checked, "
            f"critical={critical_count}, "
            f"data_missing={unknown_count}, "
            f"min_runway={min_runway}"
        )

        return {
            "inventory_signals": signals,
            "reasoning_trace": state["reasoning_trace"] + [trace],
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