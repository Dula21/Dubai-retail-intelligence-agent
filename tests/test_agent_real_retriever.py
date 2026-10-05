"""
tests/test_agent_real_retriever.py
-----------------------------------
Real retriever + real RAG node + real router. No mocks.

Why: the Phase 2 unit tests mocked the retriever in the shape the node expected, so a shape
mismatch with the real RetailRetriever went unnoticed and every real query fell back.
These tests are the guard against that. They also turn eval/smoke_agent.py (which only printed)
into assertions.

Needs the local catalogue CSV and the Chroma index (.chroma_db). If either is missing the
tests SKIP instead of failing, so a fresh clone stays green. Loading the embedding model
takes ~10 s, once per test run (module-scoped fixture).

Covered here: RAG node -> route_after_retrieval for 8 queries, plus the full agent graph on the
fallback path (which ends right after the RAG node, so no LLM or sales data is involved).
Not covered: the reorder/recommendation nodes on real data (they call downstream services).
"""
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from agents import retail_agent  # noqa: E402
from agents.nodes import rag_node  # noqa: E402


@pytest.fixture(scope="module")
def retriever():
    from backend.rag.retriever import RetailRetriever
    from eval.run_baseline_v2 import CATALOGUE_CSV, PERSIST_DIR

    csv_path, index_dir = ROOT / CATALOGUE_CSV, ROOT / PERSIST_DIR
    if not csv_path.exists() or not index_dir.exists():
        pytest.skip(f"needs {csv_path} and the Chroma index {index_dir}")
    rt = RetailRetriever(str(csv_path), None, str(index_dir))
    rt.initialize()
    rag_node.set_retriever(rt)
    return rt


def run_rag(query, intent):
    state = {"query": query, "intent": intent, "reasoning_trace": [],
             "retrieved_chunks": [], "retrieval_confidence": 0.0, "retrieval_margin": 0.0,
             "fallback_triggered": False}
    update = asyncio.run(rag_node.rag_retrieval_node(state))
    state.update(update)
    return state


# (query, intent, expected route, expected product name prefix of the top chunk or None)
CASES = [
    ("summer dress",                    "search",  "recommendation_node", "Summer Dress"),
    ("فستان صيفي",                      "search",  "recommendation_node", "Summer Dress"),
    ("kandura أبيض",                    "search",  "recommendation_node", "Kandura White"),
    ("عباية",                           "search",  "recommendation_node", None),  # two real abayas tie; both are right
    ("Nike shoes",                      "search",  "end_fallback",        None),  # not in the catalogue
    ("ايفون 15",                        "search",  "end_fallback",        None),  # not in the catalogue
    ("FSH-0014",                        "search",  "recommendation_node", "Printed T-Shirt"),
    ("how many units left of FSH-0014", "reorder", "sales_analysis_node", "Printed T-Shirt"),
]


@pytest.mark.parametrize("query,intent,route,top_name", CASES)
def test_real_retriever_through_node_and_router(retriever, query, intent, route, top_name):
    state = run_rag(query, intent)

    assert not state.get("error"), f"RAG node errored on real retriever output: {state.get('error')}"
    assert retail_agent.route_after_retrieval(state) == route

    if top_name is not None:
        assert state["retrieved_chunks"], "expected at least one chunk"
        assert state["retrieved_chunks"][0]["product_name_en"].startswith(top_name)


def test_abaya_near_tie_returns_abayas_not_a_fallback(retriever):
    # عباية is the customer spelling of the catalogue's عباءة; the two abaya products score within 0.01.
    state = run_rag("عباية", "search")
    assert state["fallback_triggered"] is False
    names = " | ".join(c["product_name_en"] for c in state["retrieved_chunks"])
    assert "Abaya" in names


def test_exact_sku_uses_catalogue_lookup(retriever):
    state = run_rag("FSH-0014", "search")
    assert state["retrieval_confidence"] == 1.0
    assert [c["sku_id"] for c in state["retrieved_chunks"]] == ["FSH-0014"]
    assert "exact SKU match" in state["reasoning_trace"][-1]


def test_unknown_sku_is_insufficient_data(retriever):
    state = run_rag("price of FSH-9999", "reorder")
    assert state["fallback_triggered"] is True
    assert "not found" in state["reasoning_trace"][-1]


def test_full_agent_graph_on_fallback_path(retriever):
    # A product that is not in the catalogue ends the graph right after the RAG node:
    # no LLM, no sales data, deterministic.
    result = asyncio.run(retail_agent.run_agent("Nike shoes", "en", "search"))
    assert result["fallback_triggered"] is True
    assert not result.get("recommendation")
    assert result["reasoning_trace"][0].startswith("Agent started")
    assert any("RAG retrieval" in step for step in result["reasoning_trace"])