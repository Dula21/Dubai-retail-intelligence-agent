"""
Regression test: the inventory node must report the REAL stock from
data/synthetic/inventory_snapshot.csv, and must never turn missing data
into a confident 0.

Bug this guards against: the node read columns `current_stock` /
`reorder_point` while the CSV has `current_stock_units` /
`reorder_point_units`; the default of 0 made every SKU look out of stock.
Data is synthetic (see data/generate_synthetic.py).
"""

import asyncio
from pathlib import Path

import pandas as pd

from backend.agents.nodes import inventory_node

CSV = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "inventory_snapshot.csv"


def _run(sku_ids):
    state = {
        "retrieved_chunks": [{"sku_id": s} for s in sku_ids],
        "sales_signals": [],
        "reasoning_trace": [],
    }
    return asyncio.run(inventory_node.inventory_check_node(state))


def test_inventory_node_reports_real_stock_from_csv():
    inv = pd.read_csv(CSV).set_index("sku_id")
    # a spread of SKUs: lowest stock, highest stock, and a middle one
    ordered = inv.sort_values("current_stock_units")
    sample = [ordered.index[0], ordered.index[len(ordered) // 2], ordered.index[-1]]

    result = _run(sample)

    by_sku = {s["sku_id"]: s for s in result["inventory_signals"]}
    for sku in sample:
        assert by_sku[sku]["current_stock"] == int(inv.loc[sku, "current_stock_units"])
        assert by_sku[sku]["reorder_point"] == int(inv.loc[sku, "reorder_point_units"])


def test_inventory_node_not_all_zero_stock():
    inv = pd.read_csv(CSV)
    result = _run(list(inv["sku_id"].head(25)))
    stocks = [s["current_stock"] for s in result["inventory_signals"]]
    assert max(stocks) > 0, "every SKU reported 0 stock: column-name regression"


def test_unknown_sku_is_insufficient_data_not_zero_or_critical():
    result = _run(["NOPE-9999"])
    sig = result["inventory_signals"][0]
    assert sig["current_stock"] == -1
    assert sig["runway_days"] == -1.0
    assert sig["is_critical"] is False