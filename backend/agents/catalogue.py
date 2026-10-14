"""
agents/catalogue.py
--------------------
Read-only access to the product catalogue and inventory snapshot CSVs (SYNTHETIC data, see
data/generate_synthetic.py) for nodes that work from the catalogue itself:

  * the seasonality node enriches retrieved chunks with fields the RAG chunk does not carry
    (supplier_lead_days, tags_en), so lead times and keyword lifts are real, not assumed;
  * the event-candidates node scans the catalogue for SKUs an event lifts.

Loaded once and cached. If a file is missing the functions return empty results and the callers say
so in the reasoning trace; nothing here guesses.

Scale ceiling: whole-catalogue CSV in memory is fine for hundreds to low thousands of SKUs.
At noon scale this becomes an indexed catalogue table in PostgreSQL.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

CATALOGUE_PATH = Path(os.getenv("CATALOGUE_DATA_PATH", "data/synthetic/product_catalogue.csv"))
INVENTORY_PATH = Path(os.getenv("INVENTORY_DATA_PATH", "data/synthetic/inventory_snapshot.csv"))


@lru_cache(maxsize=1)
def _catalogue() -> dict[str, dict]:
    if not CATALOGUE_PATH.exists():
        logger.warning("catalogue_csv_not_found", path=str(CATALOGUE_PATH))
        return {}
    with open(CATALOGUE_PATH, newline="", encoding="utf-8") as f:
        return {r["sku_id"]: r for r in csv.DictReader(f)}


@lru_cache(maxsize=1)
def _days_left() -> dict[str, float]:
    if not INVENTORY_PATH.exists():
        return {}
    out: dict[str, float] = {}
    with open(INVENTORY_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[r["sku_id"]] = float(r["days_of_stock_left"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def get_row(sku_id: str) -> dict:
    """Catalogue row for a SKU (all values are strings), or {} if unknown / catalogue missing."""
    return _catalogue().get(sku_id, {})


def all_rows() -> list[dict]:
    return list(_catalogue().values())


def days_of_stock_left(sku_id: str) -> float | None:
    """Days of stock left from the inventory snapshot, or None when unknown."""
    return _days_left().get(sku_id)


def clear_caches() -> None:
    """For tests and after a data upload."""
    _catalogue.cache_clear()
    _days_left.cache_clear()