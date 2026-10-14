"""
tests/test_sales_event.py
--------------------------
Sales node on planning questions: it reports the observed lift of the event that was ASKED about,
not each SKU's strongest event ever. Synthetic frames only (no CSV needed).
"""
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.nodes import sales_node as sa  # noqa: E402

DAYS = pd.date_range("2025-01-01", periods=100, freq="D")
RAMADAN = frozenset(DAYS[20:41])      # 21 days
HOLI = frozenset(DAYS[60:71])         # 11 days
WINDOWS = {"ramadan": RAMADAN, "holi": HOLI}


def frame(sku, event_units=2.0, holi_units=2.0):
    units = [2.0] * len(DAYS)
    for i, d in enumerate(DAYS):
        if d in RAMADAN:
            units[i] = event_units
        elif d in HOLI:
            units[i] = holi_units
    return pd.DataFrame({"date": DAYS, "sku_id": sku, "units_sold": units})


def make_df():
    return pd.concat([frame("A", event_units=6.0, holi_units=8.0),   # strong Ramadan, even stronger Holi
                      frame("B")])                                    # flat


def chunk(sku, **extra):
    return {"sku_id": sku, "product_name_en": sku, "product_name_ar": "", "category": "home",
            "similarity_score": 1.0, "content": "", **extra}


def run(monkeypatch, chunks):
    monkeypatch.setattr(sa, "_load_sales_df", make_df)
    monkeypatch.setattr(sa, "_load_event_windows", lambda: WINDOWS)
    state = {"retrieved_chunks": chunks, "reasoning_trace": ["Agent started"]}
    return asyncio.run(sa.sales_analysis_node(state))


def test_observed_lift_is_event_mean_over_outside_mean():
    assert sa._observed_event_lift(frame("A", event_units=6.0), WINDOWS, "ramadan") == 3.0


def test_too_few_event_days_gives_none():
    short = {"ramadan": frozenset(DAYS[20:24])}
    assert sa._observed_event_lift(frame("A", event_units=6.0), short, "ramadan") is None


def test_planning_reports_only_the_asked_event(monkeypatch):
    chunks = [chunk("A", event_name="Ramadan", event_lift=3.5), chunk("B", event_name="Ramadan", event_lift=3.5)]
    out = run(monkeypatch, chunks)
    by_sku = {s["sku_id"]: s for s in out["sales_signals"]}
    assert by_sku["A"]["peak_period"] == "ramadan"          # not "holi", although Holi is A's biggest peak
    assert by_sku["A"]["observed_event_lift"] == 3.0
    assert by_sku["B"]["peak_period"] is None                # flat SKU: no Ramadan peak
    assert by_sku["B"]["observed_event_lift"] == 1.0
    trace = out["reasoning_trace"][-1]
    assert "Sales analysis (Ramadan)" in trace and "calendar table says 3.5x" in trace
    assert "1 of 2 at or above 1.2x" in trace
    assert "holi" not in trace.lower()


def test_ordinary_query_keeps_strongest_event_and_counts_peaks(monkeypatch):
    out = run(monkeypatch, [chunk("A"), chunk("B")])
    by_sku = {s["sku_id"]: s for s in out["sales_signals"]}
    assert by_sku["A"]["peak_period"] == "holi"              # old behaviour: strongest event overall
    assert "observed_event_lift" not in by_sku["A"]
    assert "peak_periods={'holi': 1}" in out["reasoning_trace"][-1]


def test_unknown_sku_does_not_break_event_summary(monkeypatch):
    out = run(monkeypatch, [chunk("A", event_name="Ramadan", event_lift=3.5),
                            chunk("ZZZ", event_name="Ramadan", event_lift=3.5)])
    assert len(out["sales_signals"]) == 2
    assert out["sales_signals"][1]["confidence"] == 0.0