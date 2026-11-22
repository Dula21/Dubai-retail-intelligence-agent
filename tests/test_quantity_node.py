"""
tests/test_quantity_node.py
----------------------------
Phase 3 wiring: the quantity node feeds the engine from the loaders, the graph runs it between seasonality and
recommendation, the recommendation prompt carries the engine's numbers, and the API exposes them. No model, no network
(the Prophet fit is replaced by a flat forecast; one test uses the real data and real Prophet and is skipped without them).
"""
import asyncio
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from agents.nodes import inventory_node, quantity_node as qn, recommendation_node as rn, sales_node  # noqa: E402
from agents.nodes import seasonality_node as sn  # noqa: E402
from services import quantity_engine as qe  # noqa: E402

DATA_END = date(2026, 10, 8)
TODAY = date(2026, 10, 9)
CAL = pd.DataFrame([dict(event="Diwali", start_date=pd.Timestamp("2026-11-08"), end_date=pd.Timestamp("2026-11-10"),
                         lead_in_days=10, dip_days=3, dip_factor=0.9)])
CALIBRATION = {"safety_obj": qe.SafetyMultiplier(0.9, (), (0.20,), (500,), 0.20, 500),
               "coverage": {"overall": 0.9, "events_with_history": 0.94, "events_without_history": 0.78},
               "validated_horizon_days": 30}
CHUNK = {"sku_id": "HOM-1", "product_name_en": "X", "product_name_ar": "س", "category": "home",
         "similarity_score": 1.0, "content": "", "supplier_lead_days": 10}


def flat(per_day=10.0):
    return qe.Forecast(method="flat", train_end=pd.Timestamp(DATA_END), n_train_days=400, masked_share=0.0,
                       _predict=lambda idx: np.full(len(idx), float(per_day)))


@pytest.fixture
def wired(monkeypatch):
    qn.clear_caches()
    sales = pd.DataFrame({"sku_id": ["HOM-1"], "date": [pd.Timestamp(DATA_END)], "units_sold": [5]})
    inv = pd.DataFrame({"sku_id": ["HOM-1"], "current_stock_units": [200], "reorder_point_units": [50]})

    async def fake_forecast(sku, s, c):
        return flat()
    monkeypatch.setattr(sales_node, "_load_sales_df", lambda: sales)
    monkeypatch.setattr(inventory_node, "_load_inventory_df", lambda: inv)
    monkeypatch.setattr(qn, "_calendar_df", lambda: CAL)
    monkeypatch.setattr(qn, "load_calibration", lambda: CALIBRATION)
    monkeypatch.setattr(qn, "_get_forecast", fake_forecast)
    monkeypatch.setattr(qn, "_today", lambda: TODAY)
    monkeypatch.setattr(sn, "_load_events", lambda: ((sn.Event("Diwali", date(2026, 11, 8), date(2026, 11, 10), 10, "fixed", ()),), "calendar"))
    return monkeypatch


def state(chunks=(CHUNK,), signal=None):
    return {"query": "q", "retrieved_chunks": list(chunks), "reasoning_trace": ["start"],
            "seasonality_signal": signal or {"upcoming_event": None, "days_until_event": None,
                                             "expected_demand_multiplier": 1.0, "alert_level": "none"}}


def run(st):
    return asyncio.run(qn.quantity_node(st))


def test_no_candidates_skips_cleanly(wired):
    out = run(state(chunks=()))
    assert out["quantity_results"] == [] and "skipped" in out["reasoning_trace"][-1]


def test_without_an_event_the_window_is_arrival_plus_review_period(wired):
    r = run(state())["quantity_results"][0]
    assert r["window_start"] == "2026-10-19"                      # as of 9 Oct + 10 days lead
    assert r["window_end"] == "2026-11-01"                        # + 14 days of cover
    # forecast 14 days x 10 = 140, x1.2 = 168; stock 200 snapshot 8 Oct, depletion 9..18 Oct = 100 -> projected 100
    assert r["forecast_units"] == 140 and r["target_units"] == 168
    assert r["projected_on_hand"] == 100 and r["order_qty"] == 68
    assert "open_orders_unknown" in r["flags"]                    # no dataset holds open orders


def test_event_window_uses_the_lead_in_read_from_the_calendar(wired):
    sig = {"upcoming_event": "Diwali", "days_until_event": 30, "expected_demand_multiplier": 1.8, "alert_level": "watch",
           "event_start": "2026-11-08", "event_end": "2026-11-10"}
    r = run(state(signal=sig))["quantity_results"][0]
    assert (r["window_start"], r["window_end"]) == ("2026-10-29", "2026-11-10")      # 8 Nov - 10 days lead-in .. event end
    assert "beyond_validated_horizon" not in r["flags"] and r["horizon_days"] == 21


def test_window_beyond_the_validated_horizon_is_flagged_and_has_no_coverage_estimate(wired):
    wired.setattr(qn, "_today", lambda: date(2026, 10, 9))
    sig = {"upcoming_event": "Diwali", "days_until_event": 30, "expected_demand_multiplier": 1.8, "alert_level": "watch",
           "event_start": "2027-02-08", "event_end": "2027-03-09"}
    wired.setattr(sn, "_load_events", lambda: ((sn.Event("Diwali", date(2027, 2, 8), date(2027, 3, 9), 14, "fixed", ()),), "calendar"))
    sig["order_by_date"] = "2027-01-10"
    r = run(state(signal=sig))["quantity_results"][0]
    assert "beyond_validated_horizon" in r["flags"] and r["coverage_estimate"] is None
    assert r["indicative_only"] is True and r["order_by_date"] == "2027-01-10"


def test_validated_event_window_gets_the_measured_coverage(wired):
    sig = {"upcoming_event": "Diwali", "days_until_event": 30, "expected_demand_multiplier": 1.8, "alert_level": "act",
           "event_start": "2026-11-08", "event_end": "2026-11-10"}
    r = run(state(signal=sig))["quantity_results"][0]
    assert r["coverage_estimate"] == 0.94 and "held-out" in r["coverage_basis"]


def test_a_window_with_no_event_has_no_measured_coverage(wired):
    """Every backtest case was an event window, so a generic lead-time window must not borrow that coverage."""
    r = run(state())["quantity_results"][0]
    assert "non_event_window" in r["flags"] and r["coverage_estimate"] is None
    assert "no event" in r["coverage_basis"]


def test_stale_snapshot_is_flagged_in_the_result_and_the_trace(wired):
    wired.setattr(qn, "_today", lambda: date(2026, 11, 7))
    out = run(state())
    assert "stock_snapshot_stale" in out["quantity_results"][0]["flags"]
    assert any("snapshot dated 2026-10-08" in line for line in out["reasoning_trace"])


def test_unknown_sku_stock_gives_no_quantity(wired):
    wired.setattr(inventory_node, "_load_inventory_df", lambda: pd.DataFrame({"sku_id": ["OTHER"], "current_stock_units": [1]}))
    r = run(state())["quantity_results"][0]
    assert r["order_qty"] is None and "stock_unknown" in r["flags"]


def test_missing_calibration_gives_no_quantity(wired):
    wired.setattr(qn, "load_calibration", lambda: None)
    r = run(state())["quantity_results"][0]
    assert r["order_qty"] is None and "no_calibration" in r["flags"]


def test_missing_calendar_degrades_without_crashing(wired):
    wired.setattr(qn, "_calendar_df", lambda: None)
    out = run(state())
    assert out["quantity_results"] == [] and "calendar file not found" in out["reasoning_trace"][-1]


def test_a_failure_inside_the_node_degrades_and_sets_error(wired):
    def boom():
        raise RuntimeError("disk")
    wired.setattr(sales_node, "_load_sales_df", boom)
    out = run(state())
    assert out["quantity_results"] == [] and out["error"].startswith("Quantity engine failed")


def test_forecast_cache_is_bounded(monkeypatch):
    qn.clear_caches()
    monkeypatch.setattr(qn, "MAX_CACHED_FORECASTS", 3)
    monkeypatch.setattr(qn, "_fit_sync", lambda sku, s, c: flat())
    sales = pd.DataFrame({"sku_id": ["A"], "date": [pd.Timestamp(DATA_END)], "units_sold": [1]})
    for i in range(6):
        asyncio.run(qn._get_forecast(f"S{i}", sales, CAL))
    assert len(qn._forecasts) == 3
    qn.clear_caches()


# ── recommendation prompt ──────────────────────────────────────────────────────

def test_prompt_section_carries_the_engine_numbers_and_the_caveats(wired):
    r = run(state())["quantity_results"][0]
    text = rn._format_quantities([r])
    for part in ("ORDER 68 units", "forecast 140.0 units x safety 1.2 = target 168.0", "open orders 0",
                 "open purchase orders are unknown and assumed to be zero",
                 "this window has no event, and the backtest only validated event windows"):
        assert part in text
    assert "_" not in text.split("caveats in plain words:")[1]          # no raw flag codes for the model to echo


def test_far_window_quantity_is_presented_as_indicative_not_as_an_order():
    r = {"sku_id": "X-1", "order_qty": 1300, "indicative_only": True, "order_by_date": "2027-09-09", "window_start": "2027-10-05",
         "window_end": "2027-10-31", "forecast_units": 1007.2, "safety_multiplier": 1.29, "target_units": 1299.0,
         "on_hand_snapshot": 0, "snapshot_date": "2026-10-08", "snapshot_age_days": 39, "projected_on_hand": 0.0,
         "open_orders": 0, "coverage_estimate": None, "coverage_basis": "window starts beyond the horizon",
         "horizon_days": 360, "flags": ["stock_snapshot_stale", "beyond_validated_horizon", "open_orders_unknown"]}
    text = rn._format_quantities([r])
    assert "INDICATIVE ONLY, about 1300 units" in text and "NOT an order to place now" in text and "2027-09-09" in text
    assert "ORDER 1300" not in text
    assert "the stock figure is 39 days old" in text and "360 days after the last data" in text
    assert "_" not in text.split("caveats in plain words:")[1]


def test_confidence_is_capped_by_what_the_backtest_measured():
    ok = {"order_qty": 10, "coverage_estimate": 0.94}
    low = {"order_qty": 10, "coverage_estimate": 0.78}
    none = {"order_qty": 10, "coverage_estimate": None}
    assert rn._apply_quantity_confidence(0.80, []) == 0.80                       # no quantities: unchanged
    assert rn._apply_quantity_confidence(0.80, [{"order_qty": None}]) == 0.80
    assert rn._apply_quantity_confidence(0.99, [ok]) == 0.94
    assert rn._apply_quantity_confidence(0.99, [ok, low]) == 0.78
    assert rn._apply_quantity_confidence(0.60, [ok]) == 0.60                     # never raised
    assert rn._apply_quantity_confidence(0.80, [ok, none]) == rn.UNVALIDATED_CONFIDENCE_CAP == 0.4


def test_prompt_rules_block_order_now_for_indicative_quantities():
    assert "INDICATIVE ONLY" in rn.SYSTEM_PROMPT and "never claim an order will fix today's stock" in rn.SYSTEM_PROMPT.replace("\n    ", " ")
    assert "Never print raw caveat codes" in rn.SYSTEM_PROMPT


def test_prompt_says_no_quantity_when_the_engine_has_none():
    text = rn._format_quantities([{"sku_id": "X-1", "order_qty": None, "flags": ["stock_unknown"]}])
    assert "NO QUANTITY (stock unknown)" in text and "ORDER" not in text.replace("NO QUANTITY", "")
    assert "No quantity computed" in rn._format_quantities([])


def test_prompt_forbids_the_model_from_inventing_a_quantity():
    assert "never compute your own" in rn.RECOMMENDATION_PROMPT
    assert "copied exactly" in rn.SYSTEM_PROMPT and "State no order quantities at all" not in rn.SYSTEM_PROMPT


def test_a_bad_max_tokens_env_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("GROQ_MAX_TOKENS", "not-a-number")
    assert rn._max_tokens() == rn.DEFAULT_MAX_TOKENS
    monkeypatch.setenv("GROQ_MAX_TOKENS", "900")
    assert rn._max_tokens() == 900


# ── graph + API ────────────────────────────────────────────────────────────────

def test_graph_runs_the_quantity_node_between_seasonality_and_recommendation():
    pytest.importorskip("langgraph")
    from agents.retail_agent import retail_agent
    g = retail_agent.get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert ("seasonality_check_node", "quantity_node") in edges and ("quantity_node", "recommendation_node") in edges
    assert ("seasonality_check_node", "recommendation_node") not in edges


def test_api_response_exposes_the_quantities_and_the_cache_key_changed():
    pytest.importorskip("fastapi")
    from models.agent_models import AgentQueryResponse, OrderQuantityResponse
    resp = AgentQueryResponse(confidence=0.5, order_recommendations=[OrderQuantityResponse(sku_id="X", order_qty=5)])
    assert resp.order_recommendations[0].order_qty == 5
    assert AgentQueryResponse(confidence=0.5).order_recommendations == []          # old shape still valid


# ── real data, real Prophet ────────────────────────────────────────────────────

REAL = all((ROOT / p).exists() for p in ("data/synthetic/sales_history.csv", "data/synthetic/inventory_snapshot.csv",
                                         "data/synthetic/uae_retail_calendar.csv", "data/synthetic/product_catalogue.csv",
                                         "data/quantity_calibration.json"))


@pytest.mark.skipif(not REAL, reason="synthetic data or calibration file missing")
def test_real_data_end_to_end_gives_a_quantity_with_its_inputs(monkeypatch):
    pytest.importorskip("prophet")
    qn.clear_caches()
    sales_node._load_sales_df.cache_clear()
    sku = pd.read_csv(ROOT / "data/synthetic/product_catalogue.csv")["sku_id"].iloc[0]
    last = sales_node._load_sales_df()["date"].max().date()
    monkeypatch.setattr(qn, "_today", lambda: last)
    out = run(state(chunks=({**CHUNK, "sku_id": sku, "supplier_lead_days": 7},)))
    r = out["quantity_results"][0]
    assert isinstance(r["order_qty"], int) and r["order_qty"] >= 0, r
    assert r["forecast_units"] > 0 and r["snapshot_date"] == str(last) and "stock_snapshot_stale" not in r["flags"]
    assert r["coverage_estimate"] is None and "non_event_window" in r["flags"]     # generic window: not validated
    qn.clear_caches()


def test_confidence_and_formatting_tolerate_quantity_results_none():
    """LangGraph hands nodes None for a state key no earlier node wrote (e.g. the search path that skips quantity_node)."""
    from agents.nodes import recommendation_node as rn
    assert rn._apply_quantity_confidence(0.45, None or []) == 0.45
    assert rn._format_quantities(None or []) is not None


# ── deterministic additions to the model's answer ─────────────────────────────

def _res(sku="FD-0366", qty=1300, indicative=True, cov=None, order_by="2027-09-09", **kw):
    base = {"sku_id": sku, "order_qty": qty, "window_start": "2027-10-05", "window_end": "2027-10-31",
            "indicative_only": indicative, "order_by_date": order_by, "coverage_estimate": cov, "flags": [],
            "forecast_units": 1007.2, "safety_multiplier": 1.29, "target_units": 1299.0, "on_hand_snapshot": 0,
            "snapshot_date": "2026-10-08", "snapshot_age_days": 44, "projected_on_hand": 0.0, "open_orders": 0,
            "coverage_basis": "n/a"}
    base.update(kw)
    return base


def test_look_alike_hyphens_and_spaces_are_replaced_but_en_dashes_stay():
    from agents.nodes import recommendation_node as rn
    assert rn._normalize_text("2027‑0‑09 and 332 days left, 0–12 units") == \
        "2027-0-09 and 332 days left, 0–12 units"


def test_quantity_block_is_built_from_the_engine_numbers():
    from agents.nodes import recommendation_node as rn
    block = rn._quantities_block([_res(), _res("HOM-0264", 281, indicative=False, cov=0.94, order_by="2026-12-01")])
    assert "from the forecasting engine, not the language model" in block
    assert "- FD-0366: **1,300 units** (indicative only; recomputed by 2027-09-09; window 2027-10-05 to 2027-10-31; coverage not validated)" in block
    assert "- HOM-0264: **281 units** (order by 2026-12-01; window 2027-10-05 to 2027-10-31; coverage 94%)" in block


def test_quantity_block_handles_no_quantity_arabic_and_the_line_cap():
    from agents.nodes import recommendation_node as rn
    none = rn._quantities_block([_res(qty=None, flags=["stock_unknown"])])
    assert "- FD-0366: no quantity" in none
    assert "كميات الطلب" in rn._quantities_block([_res()], "ar") and "وحدة" in rn._quantities_block([_res()], "ar")
    many = rn._quantities_block([_res(f"S{i}") for i in range(rn.MAX_QUANTITY_LINES + 3)])
    assert many.count("units**") == rn.MAX_QUANTITY_LINES and "and 3 more" in many
    assert rn._quantities_block([]) == ""


def test_the_final_answer_keeps_the_model_text_then_the_engine_list():
    from agents.nodes import recommendation_node as rn
    out = rn._finalize_text("Monitor.‑ MONITOR", [_res()], "en")
    assert out.startswith("Monitor.- MONITOR") and out.rstrip().endswith("coverage not validated)")
    assert rn._finalize_text("No SKUs.", [], "en") == "No SKUs."


def test_the_node_appends_the_list_even_when_the_model_omits_the_numbers(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from agents.nodes import recommendation_node as rn
    choice = MagicMock(); choice.message.content = "Prepare orders before 2027‑09‑09. MONITOR"
    resp = MagicMock(); resp.choices = [choice]; resp.usage.total_tokens = 10
    client = AsyncMock(); client.chat.completions.create.return_value = resp
    monkeypatch.setattr(rn, "get_groq_client", lambda: client)
    st = {**state(), "sales_signals": [], "inventory_signals": [], "retrieval_confidence": 0.8,
          "query_language": "en", "quantity_results": [_res()]}
    out = asyncio.run(rn.recommendation_node(st))
    text = out["recommendation"]
    assert "2027-09-09" in text and "‑" not in text           # normalised
    assert "**1,300 units**" in text                                  # the number the model never wrote


# ── action label is decided by the engine, not the model ─────────────────────────

def _lbl_state(level="watch", critical=False, stale=False, signals=True):
    inv = [{"sku_id": "A", "current_stock": 0 if critical else 50, "is_critical": critical}] if signals else []
    return {"seasonality_signal": {"alert_level": level}, "inventory_signals": inv,
            "quantity_results": [{"sku_id": "A", "flags": ["stock_snapshot_stale"] if stale else []}]}


def test_label_watch_with_stale_critical_stock_is_investigate():
    assert rn.action_label(_lbl_state("watch", critical=True, stale=True)) == "INVESTIGATE"


def test_label_fresh_critical_stock_is_reorder_now():
    assert rn.action_label(_lbl_state("watch", critical=True, stale=False)) == "REORDER NOW"


def test_label_act_and_critical_alerts_reorder_now():
    assert rn.action_label(_lbl_state("act")) == "REORDER NOW"
    assert rn.action_label(_lbl_state("critical")) == "REORDER NOW"


def test_label_watch_ok_stock_is_monitor_and_none_is_no_action():
    assert rn.action_label(_lbl_state("watch")) == "MONITOR"
    assert rn.action_label(_lbl_state("none")) == "NO ACTION NEEDED"
    assert rn.action_label(_lbl_state("none", signals=False)) == "INVESTIGATE"


def test_model_label_is_replaced_by_system_label():
    text = "Stock looks empty.\n\n**Action label**  \nREORDER NOW"
    out = rn._finalize_text(text, [], "en", label="INVESTIGATE")
    assert "REORDER NOW" not in out
    assert out.rstrip().endswith("**Action:** INVESTIGATE")
    out2 = rn._finalize_text("Fine.\n\nMONITOR", [], "en", label="NO ACTION NEEDED")
    assert "MONITOR" not in out2 and "**Action:** NO ACTION NEEDED" in out2


# ── the recommendation sentence is written by code (fix6) ────────────────────────

def _full_state(level="watch", critical=True, stale=True, until=331, left=291):
    return {"seasonality_signal": {"alert_level": level, "upcoming_event": "Diwali", "order_by_date": "2027-09-09",
                                   "days_until_event": until, "days_to_order_by": left},
            "inventory_signals": [{"sku_id": "FD-1", "current_stock": 0, "is_critical": critical}],
            "quantity_results": [{"sku_id": "FD-1", "snapshot_age_days": 45,
                                  "flags": ["stock_snapshot_stale"] if stale else []}]}


def test_investigate_sentence_states_age_and_both_day_counts():
    st = _full_state()
    s = rn.recommendation_sentence(st, rn.action_label(st))
    assert "45 days old" in s and "confirm current stock before ordering" in s
    assert "starts in 331 days" in s and "(291 days from now)" in s and "2027-09-09" in s
    assert "Place the order now" not in s and "replenishment order now" not in s


def test_monitor_sentence_says_no_order_yet():
    st = _full_state(critical=False)
    assert rn.action_label(st) == "MONITOR"
    assert rn.recommendation_sentence(st, "MONITOR").startswith("No order is needed yet.")


def test_reorder_now_sentences_event_and_stock():
    st = _full_state(level="critical", left=-3)
    s = rn.recommendation_sentence(st, "REORDER NOW")
    assert s.startswith("Place the order now") and "passed 3 days ago" in s
    st2 = _full_state(critical=True, stale=False)
    assert "place a replenishment order now" in rn.recommendation_sentence(st2, "REORDER NOW")


def test_event_running_and_no_event_and_arabic():
    assert "running now" in rn.recommendation_sentence(_full_state(level="act", until=0, left=-30), "REORDER NOW")
    st = {"seasonality_signal": {}, "inventory_signals": [{"sku_id": "A", "current_stock": 9, "is_critical": False}]}
    assert rn.recommendation_sentence(st, "NO ACTION NEEDED").startswith("No action needed")
    ar = rn.recommendation_sentence(_full_state(), "INVESTIGATE", "ar")
    assert "45" in ar and "2027-09-09" in ar and "Place" not in ar


def test_model_written_recommendation_section_is_cut_and_replaced():
    model = ("Stock is old.\n\n**Recommendation**  \nPlace purchase orders now, 1300 units.\n\n"
             "**Action label**  \nREORDER NOW")
    st = _full_state()
    out = rn._finalize_text(model, [], "en", label="INVESTIGATE", state=st)
    assert "1300" not in out and "Place purchase orders now" not in out and "REORDER NOW" not in out
    assert out.startswith("Stock is old.")
    assert "**Recommendation**\nStock reads critical" in out and out.rstrip().endswith("**Action:** INVESTIGATE")


def test_prose_mentioning_recommendation_midline_is_kept():
    out = rn._cut_model_recommendation("The recommendation engine shows 3 SKUs low.\nStock is old.")
    assert out.startswith("The recommendation engine shows")
