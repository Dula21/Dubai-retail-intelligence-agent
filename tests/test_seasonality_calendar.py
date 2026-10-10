"""
tests/test_seasonality_calendar.py
-----------------------------------
The seasonality node must read the UAE retail calendar (synthetic, see data/make_calendar.py),
name real upcoming events, and give an order-by date derived from supplier lead time.
Dates are pinned by patching the node's _today(), so these tests do not depend on the real clock.
"""
import asyncio
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.agents.nodes import seasonality_node as sn

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data/synthetic/uae_retail_calendar.csv").exists(), reason="calendar CSV missing (generate it first)")

HOME = [{"sku_id": "HOM-1", "category": "home", "name_en": "Diwali Diya Set", "supplier_lead_days": 21}]
FASHION = [{"sku_id": "FSH-1", "category": "fashion", "name_en": "Summer Dress", "supplier_lead_days": 10}]
FOOD = [{"sku_id": "FD-1", "category": "food", "name_en": "Medjool Dates 1kg", "supplier_lead_days": 12}]


def run(query, chunks, today, monkeypatch):
    monkeypatch.setattr(sn, "_today", lambda: today)
    state = {"query": query, "retrieved_chunks": chunks, "reasoning_trace": []}
    out = asyncio.run(sn.seasonality_check_node(state))
    return out["seasonality_signal"], out["reasoning_trace"]


def test_every_alias_targets_a_real_calendar_event():
    events, source = sn._load_events()
    assert source == "calendar"
    names = {e.name for e in events}
    assert set(sn.EVENT_ALIASES) <= names
    assert {n for ns in sn.GENERIC_ALIASES.values() for n in ns} <= names


def test_named_event_diwali_gets_dates_and_order_by(monkeypatch):
    sig, trace = run("what should I stock before Diwali", HOME, date(2026, 10, 10), monkeypatch)
    assert sig["upcoming_event"] == "Diwali" and sig["named_in_query"] is True
    start = date.fromisoformat(sig["event_start"])
    assert date(2026, 10, 25) <= start <= date(2026, 11, 8)
    assert sig["expected_demand_multiplier"] > 1.5
    assert any("order-by" in line for line in trace)


def test_order_by_is_start_minus_lead_in_minus_supplier_lead_minus_buffer(monkeypatch):
    sig, _ = run("before Diwali", HOME, date(2026, 10, 10), monkeypatch)
    ev = next(e for e in sn._load_events()[0] if e.name == "Diwali" and e.start.year == 2026)
    expected = ev.start - timedelta(days=ev.lead_in_days + 21 + sn.ORDER_BUFFER_DAYS)
    assert sig["order_by_date"] == expected.isoformat()
    assert sig["lead_time_days"] == 21 and sig["lead_time_assumed"] is False


def test_order_by_already_passed_is_critical(monkeypatch):
    sig, trace = run("before Diwali", HOME, date(2026, 10, 10), monkeypatch)
    assert sig["days_to_order_by"] < 0 and sig["alert_level"] == "critical"
    assert any("passed" in line for line in trace)


def test_arabic_query_finds_ramadan_beyond_90_days(monkeypatch):
    sig, _ = run("ما الذي يجب أن أخزنه قبل رمضان", FOOD, date(2026, 10, 10), monkeypatch)
    assert sig["upcoming_event"] == "Ramadan" and sig["event_start"] == "2027-02-08"
    assert sig["days_until_event"] > sn.HORIZON_DAYS


def test_past_occurrence_is_never_offered(monkeypatch):
    sig, _ = run("stock before ramadan", FOOD, date(2026, 4, 1), monkeypatch)    # Ramadan 2026 already over
    assert sig["event_start"] == "2027-02-08"


def test_eid_is_not_matched_inside_other_words(monkeypatch):
    assert sn._named_event_names("show me side tables and provided items") == set()
    assert sn._named_event_names("eid dress") == {"Eid al-Fitr", "Eid al-Adha"}
    assert sn._named_event_names("eid al-adha gifts") == {"Eid al-Adha"}


def test_no_event_named_picks_a_relevant_upcoming_event_within_horizon(monkeypatch):
    today = date(2026, 10, 10)
    sig, trace = run("what should I restock", FASHION, today, monkeypatch)
    assert sig["upcoming_event"] is not None and sig["named_in_query"] is False
    assert date.fromisoformat(sig["event_start"]) <= today + timedelta(days=sn.HORIZON_DAYS)
    assert sig["expected_demand_multiplier"] >= sn.MIN_LIFT
    assert sig["other_upcoming"]                      # runners-up are reported too


def test_strong_event_outranks_mild_event_that_starts_sooner(monkeypatch):
    sig, _ = run("", [], date(2026, 10, 10), monkeypatch)            # Navratri (mild) is running; Diwali is strong
    assert sig["expected_demand_multiplier"] >= sn.STRONG_LIFT


def test_missing_supplier_data_uses_default_and_says_so(monkeypatch):
    sig, trace = run("before Diwali", [{"sku_id": "X", "category": "home"}], date(2026, 10, 10), monkeypatch)
    assert sig["lead_time_days"] == sn.DEFAULT_LEAD_TIME_DAYS and sig["lead_time_assumed"] is True
    assert any("lead time assumed" in line for line in trace)


def test_no_lift_for_these_categories_means_no_invented_event(monkeypatch):
    # a category that no calendar row lifts must not get a made-up event
    chunks = [{"sku_id": "Z-1", "category": "garden", "name_en": "Spade", "supplier_lead_days": 10}]
    sig, trace = run("", chunks, date(2026, 10, 10), monkeypatch)
    assert sig["upcoming_event"] is None and sig["alert_level"] == "none"
    assert sig["expected_demand_multiplier"] == 1.0


def test_calendar_missing_falls_back_and_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(sn, "CALENDAR_PATH", tmp_path / "nope.csv")
    sn._load_events.cache_clear()
    try:
        sig, trace = run("", FASHION, date(2026, 11, 20), monkeypatch)
        assert any("static fallback" in line for line in trace)
        assert sig["alert_level"] in {"none", "watch", "act", "critical"}
    finally:
        sn._load_events.cache_clear()


def test_signal_keeps_the_original_four_fields(monkeypatch):
    sig, _ = run("", FASHION, date(2026, 10, 10), monkeypatch)
    for key in ("upcoming_event", "days_until_event", "expected_demand_multiplier", "alert_level"):
        assert key in sig