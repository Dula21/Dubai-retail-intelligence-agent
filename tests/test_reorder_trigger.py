"""
tests/test_reorder_trigger.py
------------------------------
Phase 3 autonomous trigger: the pure rules (services/reorder_trigger.py), the scan (agents/reorder_scan.py) and its
API endpoints. No model, no network; the Prophet fit is replaced by a flat forecast.
"""
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from agents import reorder_scan as rs  # noqa: E402
from agents.nodes import inventory_node, quantity_node as qn, sales_node, seasonality_node as sn  # noqa: E402
from services import quantity_engine as qe  # noqa: E402
from services import reorder_trigger as rt  # noqa: E402

AS_OF = date(2026, 10, 8)


def ev(name, start, end, lead_in=0):
    return sn.Event(name, start, end, lead_in, "fixed", ())


# ── recent_rate ────────────────────────────────────────────────────────────────

def sales(units, stockout=None, end=AS_OF):
    n = len(units)
    return pd.DataFrame({"date": pd.date_range(end=pd.Timestamp(end), periods=n), "sku_id": "A",
                         "units_sold": units, "stockout_occurred": stockout if stockout is not None else [0] * n})


def test_recent_rate_is_the_mean_of_the_last_28_days():
    rate, n = rt.recent_rate(sales([1] * 100 + [4] * 28))
    assert rate == 4.0 and n == 28


def test_recent_rate_ignores_stockout_censored_days():
    units = [5] * 14 + [0] * 7 + [5] * 7
    flags = [0] * 14 + [1] + [0] * 6 + [0] * 7                 # a stockout run: the zeros are not demand
    rate, n = rt.recent_rate(sales(units, flags))
    assert rate == 5.0 and n == 21


def test_recent_rate_is_unknown_with_too_few_usable_days():
    rate, n = rt.recent_rate(sales([5] * 10))
    assert rate is None and n == 10
    assert rt.recent_rate(sales([]).iloc[0:0]) == (None, 0)


# ── runway rule ────────────────────────────────────────────────────────────────

def test_runway_triggers_inside_the_review_period():
    # 60 units at 2/day = 30 days; lead 7 + buffer 5 -> must order in 18 days: outside the 14-day period
    a = rt.assess_runway(60, AS_OF, AS_OF, 2.0, 28, lead_days=7)
    assert a.status == "assessed" and not a.triggered and a.must_order_in == pytest.approx(18)
    # 50 units -> 25 days -> must order in 13: inside
    b = rt.assess_runway(50, AS_OF, AS_OF, 2.0, 28, lead_days=7)
    assert b.triggered and b.must_order_in == pytest.approx(13) and b.remaining_runway == pytest.approx(25)


def test_an_empty_shelf_triggers_even_without_a_rate():
    a = rt.assess_runway(0, AS_OF, AS_OF, None, 0, lead_days=7)
    assert a.triggered and a.remaining_runway == 0.0


def test_stale_stock_is_a_hard_fallback_not_a_guess():
    a = rt.assess_runway(5, AS_OF - timedelta(days=rt.MAX_SNAPSHOT_AGE_DAYS + 1), AS_OF, 2.0, 28, lead_days=7)
    assert a.status == "snapshot_stale" and not a.triggered and a.remaining_runway is None
    assert "insufficient data" in a.trace


def test_unknown_stock_or_rate_never_triggers():
    assert rt.assess_runway(-1, AS_OF, AS_OF, 2.0, 28, 7).status == "stock_unknown"
    assert rt.assess_runway(None, AS_OF, AS_OF, 2.0, 28, 7).status == "stock_unknown"
    c = rt.assess_runway(40, AS_OF, AS_OF, None, 3, 7)
    assert c.status == "rate_unknown" and not c.triggered


def test_stock_is_run_down_by_consumption_since_a_recent_snapshot():
    a = rt.assess_runway(20, AS_OF - timedelta(days=5), AS_OF, 2.0, 28, lead_days=7)
    assert a.stock_now == pytest.approx(10.0) and a.remaining_runway == pytest.approx(5.0)


# ── event rule ─────────────────────────────────────────────────────────────────

LIFT = lambda e: 2.0      # noqa: E731


def test_event_triggers_inside_the_notice_window_and_not_before():
    e = ev("DSF", date(2026, 12, 15), date(2027, 1, 15), lead_in=7)        # order-by = Dec 15 - 7 - 21 - 5 = Nov 12
    assert rt.next_event_trigger([e], LIFT, date(2026, 10, 1), 21) is None
    t = rt.next_event_trigger([e], LIFT, date(2026, 11, 6), 21)            # 6 days before order-by: inside 7-day notice
    assert t and t.order_by == date(2026, 11, 12) and t.days_to_order_by == 6
    assert rt.next_event_trigger([e], LIFT, date(2026, 11, 5), 21).days_to_order_by == 7      # exactly 7 days: fires
    assert rt.next_event_trigger([e], LIFT, date(2026, 11, 4), 21) is None                    # 8 days: not yet


def test_an_overdue_event_still_triggers_while_an_order_can_arrive_in_time():
    e = ev("DSF", date(2026, 12, 15), date(2027, 1, 15), lead_in=7)
    late = rt.next_event_trigger([e], LIFT, date(2026, 11, 20), 21)
    assert late and late.days_to_order_by == -8
    assert rt.next_event_trigger([e], LIFT, date(2026, 12, 20), 21) is not None     # arrives Jan 10, event ends Jan 15
    assert rt.next_event_trigger([e], LIFT, date(2027, 1, 1), 21) is None           # arrives Jan 22, after the end


def test_an_order_that_would_arrive_after_the_event_does_not_trigger():
    e = ev("National Day", date(2026, 12, 1), date(2026, 12, 4))
    assert rt.next_event_trigger([e], LIFT, date(2026, 11, 25), 21) is None


def test_seasons_weak_lifts_and_finished_events_never_trigger():
    season = ev("Summer", date(2026, 6, 1), date(2026, 9, 30))
    assert rt.next_event_trigger([season], LIFT, date(2026, 7, 1), 7) is None
    e = ev("DSF", date(2026, 12, 15), date(2027, 1, 15))
    assert rt.next_event_trigger([e], lambda _: 1.1, date(2026, 12, 1), 7) is None
    assert rt.next_event_trigger([e], LIFT, date(2027, 2, 1), 7) is None


def test_the_earliest_triggering_event_wins():
    a, b = ev("A", date(2026, 12, 1), date(2026, 12, 4)), ev("B", date(2026, 12, 15), date(2027, 1, 15))
    assert rt.next_event_trigger([b, a], LIFT, date(2026, 11, 20), 7).name == "A"


def test_urgency_and_rec_id():
    assert (rt.urgency(-1), rt.urgency(0), rt.urgency(3)) == ("late", "now", "upcoming")
    a = rt.rec_id("S1", "runway", None, AS_OF)
    assert a == rt.rec_id("S1", "runway", None, AS_OF) and a != rt.rec_id("S1", "runway", None, AS_OF + timedelta(days=1))
    assert a != rt.rec_id("S2", "runway", None, AS_OF)


# ── the scan ───────────────────────────────────────────────────────────────────

CALIBRATION = {"safety_obj": qe.SafetyMultiplier(0.9, (), (0.20,), (500,), 0.20, 500),
               "coverage": {"overall": 0.9, "events_with_history": 0.94, "events_without_history": 0.78},
               "validated_horizon_days": 30}
CAL = pd.DataFrame([dict(event="DSF", start_date=pd.Timestamp("2026-11-20"), end_date=pd.Timestamp("2026-12-20"),
                         lead_in_days=0, dip_days=0, dip_factor=1.0)])
CATALOGUE = {
    "LOW":  {"name_en": "Low stock", "category": "home", "tags_en": "", "supplier_lead_days": "7"},
    "OK":   {"name_en": "Healthy", "category": "home", "tags_en": "", "supplier_lead_days": "7"},
    "OUT":  {"name_en": "Empty", "category": "home", "tags_en": "", "supplier_lead_days": "7"},
}


def flat(per_day=10.0):
    return qe.Forecast(method="flat", train_end=pd.Timestamp(AS_OF), n_train_days=400, masked_share=0.0,
                       _predict=lambda idx: np.full(len(idx), float(per_day)))


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    """The container sets DATABASE_URL; these tests must run on the in-memory store, never on the real database."""
    monkeypatch.delenv("DATABASE_URL", raising=False)


@pytest.fixture
def scan_env(monkeypatch):
    qn.clear_caches()
    rs.clear_state()
    rows = []
    for sku, per_day in (("LOW", 5), ("OK", 1), ("OUT", 5)):
        d = pd.DataFrame({"date": pd.date_range(end=pd.Timestamp(AS_OF), periods=60), "sku_id": sku,
                          "units_sold": per_day, "stockout_occurred": 0})
        rows.append(d)
    sales_df = pd.concat(rows, ignore_index=True)
    inv = pd.DataFrame({"sku_id": ["LOW", "OK", "OUT"], "current_stock_units": [40, 900, 0]})

    async def fake_forecast(sku, s, c):
        return flat(5.0)

    monkeypatch.setattr(sales_node, "_load_sales_df", lambda: sales_df)
    monkeypatch.setattr(inventory_node, "_load_inventory_df", lambda: inv)
    monkeypatch.setattr(qn, "_calendar_df", lambda: CAL)
    monkeypatch.setattr(qn, "load_calibration", lambda: CALIBRATION)
    monkeypatch.setattr(qn, "_get_forecast", fake_forecast)
    monkeypatch.setattr(qn, "_lead_days", lambda chunk: (7, False))
    monkeypatch.setattr(rs.catalogue, "get_row", lambda sku: CATALOGUE.get(sku, {}))
    monkeypatch.setattr(sn, "_load_events", lambda: (
        (ev("DSF", date(2026, 11, 20), date(2026, 12, 20)),), "calendar"))
    monkeypatch.setattr(sn, "lift_for", lambda e, cat, text: 1.0)       # no event lift unless a test sets one
    monkeypatch.setattr(sn, "_today", lambda: AS_OF)
    yield monkeypatch
    rs.clear_state()
    qn._forecasts.clear()           # (clear_caches() would hit the stubs above, which are still patched here)


def scan(**kw):
    return asyncio.run(rs.run_scan(**kw))


def test_scan_raises_recommendations_for_low_and_empty_stock_without_a_question(scan_env):
    out = scan()
    by = {r["sku_id"]: r for r in out["recommendations"]}
    assert set(by) == {"LOW", "OUT"}                                   # the healthy SKU is left alone
    assert by["OUT"]["urgency"] == "late" and by["OUT"]["trigger"] == "runway"
    assert out["triggered"] == 2 and out["snapshot_age_days"] == 0


def test_every_recommendation_has_a_trace_a_confidence_and_a_pending_status(scan_env):
    for r in scan()["recommendations"]:
        assert r["status"] == "pending" and 0 <= r["confidence"] <= 1
        assert r["reasoning_trace"] and r["reasoning_trace"][0].startswith("Trigger:")
        assert any(line.startswith("Why:") for line in r["reasoning_trace"])
        assert any(line.startswith("Confidence") for line in r["reasoning_trace"])
        assert r["headline"].startswith("Reorder ") and str(r["order_qty"]) in r["headline"]


def test_a_runway_only_recommendation_is_not_given_borrowed_confidence(scan_env):
    for r in scan()["recommendations"]:
        assert r["event"] is None
        assert r["confidence"] == 0.4 and "non_event_window" in r["flags"]


def test_an_event_recommendation_carries_the_measured_coverage(scan_env):
    scan_env.setattr(sn, "lift_for", lambda e, cat, text: 2.0)
    scan_env.setattr(sn, "_today", lambda: date(2026, 11, 14))              # DSF order-by = Nov 20 - 7 - 5 = Nov 8
    scan_env.setattr(sales_node, "_load_sales_df", sales_node._load_sales_df)  # unchanged
    out = scan(as_of=date(2026, 11, 14))
    ev_recs = [r for r in out["recommendations"] if r["event"] == "DSF"]
    assert ev_recs, out["scan_trace"]
    assert all(r["order_by_date"] == "2026-11-08" and r["urgency"] == "late" for r in ev_recs)
    assert all("stock_snapshot_stale" in r["flags"] and r["confidence"] == 0.4 for r in ev_recs)   # 37-day-old stock


def test_stale_stock_skips_the_runway_rule_and_says_so(scan_env):
    scan_env.setattr(sn, "_today", lambda: AS_OF + timedelta(days=30))
    out = scan()
    assert out["recommendations"] == [] and out["not_assessed"]["snapshot_stale"] == 3
    assert any("stale stock" in line for line in out["scan_trace"])


def test_limit_defers_the_rest_instead_of_dropping_them(scan_env):
    out = scan(limit=1)
    assert out["sized"] == 1 and len(out["recommendations"]) == 1 and len(out["deferred"]) == 1
    assert out["recommendations"][0]["sku_id"] == "OUT"                    # the empty shelf is sized first


def test_a_replay_is_labelled_and_the_future_is_refused(scan_env):
    scan_env.setattr(sn, "_today", lambda: AS_OF + timedelta(days=1))
    out = scan(as_of=AS_OF)
    assert out["replay"] is True and out["recommendations"][0]["reasoning_trace"][0].startswith("REPLAY")
    with pytest.raises(ValueError):
        scan(as_of=AS_OF + timedelta(days=5))


def test_rescanning_is_idempotent_and_never_overwrites_a_human_decision(scan_env):
    first = scan()["recommendations"]
    ids = {r["rec_id"] for r in first}
    rs._store[first[0]["rec_id"]]["status"] = "approved"                   # a human acted on it
    second = scan()["recommendations"]
    assert {r["rec_id"] for r in second} == ids and len(rs.store_list()) == len(ids)
    assert rs._store[first[0]["rec_id"]]["status"] == "approved"


def test_the_store_is_bounded(scan_env):
    scan_env.setattr(rs, "MAX_STORED", 2)
    for i in range(5):
        rs.store_put({"rec_id": f"r{i}", "status": "pending", "sku_id": "A", "urgency": "now", "order_by_date": None})
    assert len(rs._store) == 2


def test_a_second_scan_while_one_is_running_is_refused(scan_env):
    async def go():
        rs._scan_lock = asyncio.Lock()
        await rs._scan_lock.acquire()
        with pytest.raises(RuntimeError):
            await rs.run_scan()
    asyncio.run(go())
    rs._scan_lock = None


# ── API ────────────────────────────────────────────────────────────────────────

def _client(monkeypatch, run_scan):
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend import agent_router as ar              # same import path the app uses (relative imports inside)
    monkeypatch.setattr(ar.reorder_scan, "run_scan", run_scan)
    app = FastAPI()
    app.include_router(ar.router, prefix="/api/v1/agent")
    return TestClient(app), ar


def test_scan_endpoint_passes_its_arguments_through(monkeypatch):
    seen = {}

    async def fake(as_of=None, limit=None):
        seen.update(as_of=as_of, limit=limit)
        return {"recommendations": [], "triggered": 0}

    c, _ = _client(monkeypatch, fake)
    assert c.post("/api/v1/agent/scan", json={"as_of": "2026-10-08", "limit": 5}).status_code == 200
    assert seen == {"as_of": date(2026, 10, 8), "limit": 5}
    assert c.post("/api/v1/agent/scan").status_code == 200                       # body is optional
    assert c.post("/api/v1/agent/scan", json={"limit": 0}).status_code == 422     # validated


def test_scan_endpoint_maps_errors_to_409_and_422(monkeypatch):
    async def busy(as_of=None, limit=None):
        raise RuntimeError("a scan is already running")

    async def future(as_of=None, limit=None):
        raise ValueError("as_of cannot be in the future")

    c, _ = _client(monkeypatch, busy)
    assert c.post("/api/v1/agent/scan", json={}).status_code == 409
    c, _ = _client(monkeypatch, future)
    assert c.post("/api/v1/agent/scan", json={}).status_code == 422


def test_recommendations_endpoint_filters_and_summarises(monkeypatch):
    async def noop(as_of=None, limit=None):
        return {}

    c, ar = _client(monkeypatch, noop)
    ar.reorder_scan.clear_state()
    for i, (sku, st) in enumerate((("A", "pending"), ("B", "approved"), ("A", "approved"))):
        ar.reorder_scan.store_put({"rec_id": f"r{i}", "status": st, "sku_id": sku, "urgency": "now", "order_by_date": None})
    assert c.get("/api/v1/agent/recommendations").json()["count"] == 3
    assert c.get("/api/v1/agent/recommendations?status=approved").json()["count"] == 2
    assert c.get("/api/v1/agent/recommendations?sku_id=A&status=approved").json()["count"] == 1
    assert "last_scan" in c.get("/api/v1/agent/recommendations").json()
    ar.reorder_scan.clear_state()
