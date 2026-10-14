"""
tests/test_recommendation_node.py
----------------------------------
Recommendation node with a fake Groq client (no network, no key needed).
Guards: the model comes from env, empty/failed LLM replies degrade cleanly, no raw provider error
reaches the caller or the reasoning trace, and the prompt carries the calendar seasonality facts
(event dates, supplier lead time, order-by date, "PASSED") plus the anti-invention rules.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.nodes import recommendation_node as rn  # noqa: E402

DIWALI = {
    "upcoming_event": "Diwali", "days_until_event": 19, "expected_demand_multiplier": 2.6,
    "alert_level": "critical", "event_start": "2026-10-29", "event_end": "2026-11-10",
    "order_by_date": "2026-09-19", "days_to_order_by": -21, "lead_time_days": 21,
    "lead_time_assumed": False, "date_basis": "computed_holidays_lib", "named_in_query": True,
    "other_upcoming": [{"name": "White Friday", "start": "2026-11-23", "multiplier": 2.8,
                        "order_by": "2026-10-28"}],
}


def make_state(seasonality=None):
    return {
        "query": "how many units left of FSH-0014", "query_language": "en", "intent": "reorder",
        "retrieved_chunks": [], "sales_signals": [], "inventory_signals": [],
        "seasonality_signal": seasonality or {}, "retrieval_confidence": 0.8,
        "reasoning_trace": ["Agent started"],
    }


class FakeClient:
    def __init__(self, content="MONITOR", exc=None):
        self.kwargs, self._content, self._exc = None, content, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        if self._exc:
            raise self._exc
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def run(monkeypatch, client, state=None):
    monkeypatch.setattr(rn, "get_groq_client", lambda: client)
    return asyncio.run(rn.recommendation_node(state or make_state()))


def test_success_uses_model_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some/test-model")
    client = FakeClient(content="Stock is fine.\nMONITOR")
    out = run(monkeypatch, client)
    assert client.kwargs["model"] == "some/test-model"
    assert out["recommendation"].endswith("MONITOR")
    assert "some/test-model" in out["reasoning_trace"][-1]
    assert "error" not in out


def test_provider_error_never_reaches_caller(monkeypatch):
    client = FakeClient(exc=RuntimeError("Error code: 404 model_not_found secret-detail"))
    out = run(monkeypatch, client)
    assert "secret-detail" not in out["recommendation"]
    assert "secret-detail" not in " ".join(out["reasoning_trace"])
    assert out["recommendation_confidence"] == 0.0
    assert out["error"]                      # set, so the API does not cache it


def test_empty_model_reply_is_a_degraded_answer_not_a_blank(monkeypatch):
    out = run(monkeypatch, FakeClient(content=""))
    assert out["recommendation"].strip()
    assert out["error"]


def test_prompt_carries_event_dates_lead_time_and_passed_order_by(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, make_state(DIWALI))
    prompt = client.kwargs["messages"][1]["content"]
    assert "Diwali 2026-10-29 to 2026-11-10" in prompt
    assert "supplier lead time: 21 days" in prompt
    assert "order-by date: 2026-09-19 (PASSED 21 days ago)" in prompt
    assert "White Friday" in prompt                      # runners-up reach the model too


def test_system_prompt_forbids_invented_rules_and_quantities(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client)
    system = client.kwargs["messages"][0]["content"]
    assert "Never invent rules, formulas, safety-stock percentages or order quantities" in system
    assert "PASSED" in system


def test_assumed_lead_time_and_estimated_dates_are_flagged():
    text = rn._format_seasonality({**DIWALI, "lead_time_assumed": True, "date_basis": "estimate_verify"})
    assert "ASSUMED" in text and "ESTIMATE" in text


def test_running_event_does_not_say_passed():
    running = {**DIWALI, "days_until_event": -3, "days_to_order_by": -40}
    text = rn._format_seasonality(running)
    assert "running now" in text and "event already running" in text and "PASSED" not in text


def test_old_four_field_signal_still_formats():
    old = {"upcoming_event": "DSF", "days_until_event": 12, "expected_demand_multiplier": 2.1,
           "alert_level": "act"}
    text = rn._format_seasonality(old)
    assert "DSF" in text and "starts in 12 days" in text and "ACT" in text


def test_no_event_says_baseline():
    assert "baseline demand" in rn._format_seasonality({})
    assert "90 days" in rn._format_seasonality({})


# ── planning answers: stock today vs event deadline, no quantities, honest confidence ──────────

WATCH = {
    "upcoming_event": "Ramadan", "days_until_event": 119, "expected_demand_multiplier": 3.5,
    "alert_level": "watch", "event_start": "2027-02-08", "event_end": "2027-03-08",
    "order_by_date": "2027-01-02", "days_to_order_by": 82, "lead_time_days": 18,
    "lead_time_assumed": False, "date_basis": "computed_umm_al_qura", "named_in_query": True,
    "other_upcoming": [],
}


def planning_state(seasonality, critical=True):
    chunk = {"sku_id": "HOM-0311", "product_name_en": "Ramadan Lantern 7", "product_name_ar": "x",
             "category": "home", "similarity_score": 1.0, "content": "",
             "event_name": "Ramadan", "event_lift": 3.5}
    inv = {"sku_id": "HOM-0311", "current_stock": 1, "reorder_point": 80,
           "runway_days": 0.0, "is_critical": critical}
    state = make_state(seasonality)
    state.update({"intent": "event_planning", "query": "what should I stock before Ramadan",
                  "retrieved_chunks": [chunk], "inventory_signals": [inv],
                  "sales_signals": [{"sku_id": "HOM-0311", "avg_daily_units": 5.6,
                                     "trend_direction": "stable", "peak_period": "ramadan",
                                     "confidence": 0.9}],
                  "retrieval_confidence": 1.0})
    return state


def prompt_of(client):
    return "\n".join(m["content"] for m in client.kwargs["messages"])


def test_watch_alert_separates_stock_problem_from_event_deadline(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, planning_state(WATCH))
    p = prompt_of(client)
    assert "stock today: 1 of 1 listed SKUs are CRITICAL right now" in p
    assert "alert WATCH" in p and "NOT urgent for the event yet" in p
    assert "order-by date: 2027-01-02 (82 days left)" in p


def test_non_critical_stock_is_not_reported_as_a_problem(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, planning_state(WATCH, critical=False))
    assert "none of the 1 listed SKUs is critical" in prompt_of(client)


def test_event_chunks_show_lift_not_a_fake_similarity(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, planning_state(WATCH))
    p = prompt_of(client)
    assert "selected by Ramadan lift table: expected lift 3.5x" in p
    assert "Relevance: 1.000" not in p


def test_system_prompt_forbids_quantities_and_urgency_blending():
    assert "NOT an order quantity" in rn.SYSTEM_PROMPT
    assert "Never describe a WATCH deadline as urgent" in rn.SYSTEM_PROMPT


def test_event_selection_confidence_is_capped_below_one(monkeypatch):
    out = run(monkeypatch, FakeClient(), planning_state(WATCH))
    assert out["recommendation_confidence"] == 0.8      # 0.2 selection + 0.3 sales + 0.2 inventory + 0.1


def test_assumed_lead_time_lowers_event_confidence(monkeypatch):
    assumed = dict(WATCH, lead_time_assumed=True)
    out = run(monkeypatch, FakeClient(), planning_state(assumed))
    assert out["recommendation_confidence"] == 0.7


def test_retrieval_path_confidence_formula_unchanged(monkeypatch):
    state = make_state(WATCH)
    state.update({"retrieval_confidence": 0.8, "sales_signals": [{"x": 1}], "inventory_signals": [{"x": 1}]})
    assert rn._compute_confidence(state) == round(0.8 * 0.4 + 0.3 + 0.2 + 0.1, 2)


def test_sales_line_shows_observed_event_lift_not_other_peaks(monkeypatch):
    state = planning_state(WATCH)
    state["sales_signals"][0].update({"event_name": "Ramadan", "observed_event_lift": 1.9, "peak_period": "ramadan"})
    client = FakeClient()
    run(monkeypatch, client, state)
    p = prompt_of(client)
    assert "observed Ramadan lift in past years: 1.9x" in p
    assert "peaks during" not in p


def test_observed_range_line_and_rule_keep_expected_and_observed_apart(monkeypatch):
    state = planning_state(WATCH)
    state["sales_signals"][0].update({"event_name": "Ramadan", "observed_event_lift": 1.9})
    client = FakeClient()
    run(monkeypatch, client, state)
    p = prompt_of(client)
    assert "Observed Ramadan lift in past years across these SKUs: 1.9x to 1.9x" in p
    assert "an expectation, not a past result" in p
    assert "Never call the expected multiplier a past, observed or historical lift" in rn.SYSTEM_PROMPT