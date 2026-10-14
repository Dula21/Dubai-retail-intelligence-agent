"""
tests/test_agent_router_planning.py
------------------------------------
API layer for the event-planning route: calendar-selected SKUs report relevance_score = null
(there is no similarity), retrieval-selected SKUs keep their score, the cache key carries a code
version, and /health lists the new node. The agent itself is stubbed: no model, no network.
"""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend import agent_router as ar  # noqa: E402

BASE = {"product_name_en": "X", "product_name_ar": "س", "category": "home", "content": ""}


def fake_state(chunks, intent="event_planning"):
    return {
        "recommendation": "ok", "recommendation_confidence": 0.8, "reasoning_trace": ["t"],
        "retrieved_chunks": chunks, "fallback_triggered": False, "error": None,
        "seasonality_signal": {"upcoming_event": "Ramadan", "days_until_event": 118,
                               "expected_demand_multiplier": 3.5, "alert_level": "watch"},
        "intent": intent,
    }


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(ar.router, prefix="/api/v1/agent")
    return TestClient(app)


def post(client, monkeypatch, state, query="what should I stock before Ramadan"):
    async def fake_run_agent(**kwargs):
        return state
    monkeypatch.setattr(ar, "run_agent", fake_run_agent)
    return client.post("/api/v1/agent/query", json={"query": query})


def test_calendar_selected_sku_has_null_relevance_and_lift(client, monkeypatch):
    chunk = {**BASE, "sku_id": "HOM-0311", "similarity_score": 1.0, "event_name": "Ramadan", "event_lift": 3.5}
    body = post(client, monkeypatch, fake_state([chunk])).json()
    product = body["retrieved_products"][0]
    assert product["relevance_score"] is None
    assert product["selection_basis"] == "calendar_lift"
    assert product["event_lift"] == 3.5
    assert body["intent"] == "event_planning"


def test_retrieved_sku_keeps_its_similarity_score(client, monkeypatch):
    chunk = {**BASE, "sku_id": "FSH-0014", "similarity_score": 0.83}
    body = post(client, monkeypatch, fake_state([chunk], intent="search"),
                query="show me summer dresses").json()
    product = body["retrieved_products"][0]
    assert product["relevance_score"] == 0.83
    assert product["selection_basis"] == "similarity"
    assert product["event_lift"] is None


def test_cache_key_changes_with_the_code_version(monkeypatch):
    monkeypatch.setattr(ar, "CACHE_VERSION", "v1")
    a = ar._cache_key("what should I stock before Ramadan")
    assert a == ar._cache_key("  What should I stock before Ramadan ")      # still normalised
    monkeypatch.setattr(ar, "CACHE_VERSION", "v2")
    assert ar._cache_key("what should I stock before Ramadan") != a


def test_health_lists_the_event_candidates_node(client):
    body = client.get("/api/v1/agent/health").json()
    assert "event_candidates_node" in body["nodes"]