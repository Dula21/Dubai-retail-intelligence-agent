"""
tests/test_fallback_node.py
----------------------------
The fallback answer (principle 10): hard "insufficient data" for factual intents, partial answer with
provenance for narrative intents. No LLM, no real index: unit tests on the node, plus graph tests
through run_agent() with a fake retriever that returns the REAL retriever output shape.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents import retail_agent  # noqa: E402
from agents.nodes import rag_node  # noqa: E402
from agents.nodes.fallback_node import fallback_node  # noqa: E402

CHUNKS = [
    {"sku_id": "ELC-0124", "product_name_en": "Bluetooth Speaker", "product_name_ar": "مكبر صوت بلوتوث",
     "category": "electronics", "similarity_score": 0.7592, "content": "Bluetooth Speaker"},
    {"sku_id": "FOD-0315", "product_name_en": "Sumac Spice 150g", "product_name_ar": "سماق",
     "category": "food", "similarity_score": 0.7637, "content": "Sumac Spice"},
]


def node(**overrides):
    state = {"query": "سبيكر", "query_language": "en", "intent": "search", "retrieved_chunks": CHUNKS,
             "retrieval_confidence": 0.7637, "retrieval_margin": 0.0045, "error": None,
             "reasoning_trace": ["Agent started"]}
    state.update(overrides)
    return asyncio.run(fallback_node(state))


# ── node: policy ──────────────────────────────────────────────────────────────

def test_factual_intent_is_a_hard_fallback_with_no_products_or_numbers():
    for intent in ("reorder", "analysis"):
        out = node(intent=intent)
        assert out["recommendation"].startswith("Insufficient data")
        for c in CHUNKS:
            assert c["product_name_en"] not in out["recommendation"]
            assert c["sku_id"] not in out["recommendation"]
        assert not any(ch.isdigit() for ch in out["recommendation"])


def test_narrative_intent_gets_a_partial_answer_with_provenance_and_confidence_label():
    for intent in ("search", "trend"):
        rec = node(intent=intent)["recommendation"]
        assert "product catalogue" in rec                       # provenance
        assert "LOW" in rec                                     # confidence label
        assert "partial" in rec and "verify before acting" in rec
        assert "Bluetooth Speaker" in rec and "ELC-0124" in rec and "0.759" in rec


def test_no_candidates_is_insufficient_data_for_any_intent():
    for intent in ("search", "reorder"):
        assert node(intent=intent, retrieved_chunks=[])["recommendation"].startswith("Insufficient data")


def test_unknown_sku_names_the_sku():
    out = node(intent="reorder", query="price of FSH-9999", retrieved_chunks=[])
    assert "FSH-9999" in out["recommendation"] and "not found" in out["recommendation"]


def test_technical_error_never_shows_candidates_even_for_narrative_intents():
    out = node(intent="search", error="RAG retrieval failed: chroma down")
    assert out["recommendation"].startswith("Insufficient data")
    assert "Bluetooth Speaker" not in out["recommendation"]
    assert "chroma down" not in out["recommendation"]           # internal error text is not shown to users


def test_confidence_is_zero_and_trace_is_appended():
    out = node()
    assert out["recommendation_confidence"] == 0.0
    assert out["reasoning_trace"][0] == "Agent started"
    assert "Fallback: partial_answer" in out["reasoning_trace"][-1] and "no LLM" in out["reasoning_trace"][-1]


def test_arabic_and_mixed_queries_get_arabic_wording():
    for lang in ("ar", "mixed"):
        assert "لا تتوفر بيانات كافية" in node(intent="reorder", query_language=lang)["recommendation"]
        partial = node(intent="search", query_language=lang)["recommendation"]
        assert "ملاحظة" in partial and "Bluetooth Speaker" in partial   # names/SKUs stay as they are


# ── graph: real output shape from the retriever, fallback path only (no LLM needed) ──

CATALOGUE = [
    {"sku_id": "ELC-0124", "name_en": "Bluetooth Speaker", "name_ar": "مكبر صوت بلوتوث", "category": "electronics"},
    {"sku_id": "FOD-0315", "name_en": "Sumac Spice 150g",  "name_ar": "سماق",            "category": "food"},
    {"sku_id": "FSH-0007", "name_en": "Sports Leggings",   "name_ar": "ليغنز رياضي",      "category": "fashion"},
]


def _hit(row, score):
    return {**row, "similarity_score": score,
            "document_text_preview": f"{row['name_en']} | {row['name_ar']}", "price_aed": 10.0, "sales": {}}


class _Loader:
    def load_catalogue(self):
        return CATALOGUE


class _Retriever:
    """Flat scores and no query term in any result: the margin fallback fires."""
    loader = _Loader()

    def search(self, query, n_results=5, **kw):
        flat = [_hit(CATALOGUE[0], 0.7640), _hit(CATALOGUE[1], 0.7638), _hit(CATALOGUE[2], 0.7636)]
        return {"query": query, "results": flat, "result_count": 3}


def run_agent(query, language, intent):
    rag_node.set_retriever(_Retriever())
    return asyncio.run(retail_agent.run_agent(query, language, intent))


def test_graph_search_fallback_returns_partial_answer():
    result = run_agent("Nike shoes", "en", "search")
    assert result["fallback_triggered"] is True
    assert "partial" in result["recommendation"]
    assert "Bluetooth Speaker" in result["recommendation"]
    assert result["recommendation_confidence"] == 0.0
    assert result["reasoning_trace"][-1].startswith("Fallback:")


def test_graph_reorder_fallback_is_hard_insufficient_data():
    result = run_agent("how many Nike shoes are left", "en", "reorder")
    assert result["fallback_triggered"] is True
    assert result["recommendation"].startswith("Insufficient data")
    assert "Bluetooth Speaker" not in result["recommendation"]


def test_graph_unknown_sku_is_insufficient_data():
    result = run_agent("price of FSH-9999", "en", "reorder")
    assert result["fallback_triggered"] is True
    assert "FSH-9999" in result["recommendation"] and "not found" in result["recommendation"]


def test_graph_arabic_fallback_answers_in_arabic():
    result = run_agent("ايفون 15", "ar", "search")
    assert result["fallback_triggered"] is True
    assert "ملاحظة" in result["recommendation"]