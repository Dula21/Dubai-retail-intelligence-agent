"""
tests/test_rag_node_integration.py
-----------------------------------
Tests the RAG node against the REAL retriever output shape.

Why this file exists: the Phase 2 unit tests mocked the retriever with a bare list
(product_name_en / content). The real RetailRetriever.search() returns a dict
({"results": [...]}) with name_en / document_text_preview, so with real data every query
crashed inside the node and fell back. These tests use the real shape, so that
mismatch cannot return unnoticed.

Plain asyncio.run() is used (no pytest-asyncio marker needed).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.nodes import rag_node  # noqa: E402
from services.sku_lookup import build_family_map, extract_sku_candidates  # noqa: E402

CATALOGUE = [
    {"sku_id": "FSH-0001", "name_en": "Summer Dress",     "name_ar": "فستان صيفي",   "category": "fashion",     "tags_en": "summer", "tags_ar": "صيف"},
    {"sku_id": "FSH-0016", "name_en": "Summer Dress Pro", "name_ar": "فستان صيفي برو", "category": "fashion",   "tags_en": "summer", "tags_ar": "صيف"},
    {"sku_id": "FSH-0002", "name_en": "Abaya Classic",   "name_ar": "عباءة كلاسيكية", "category": "fashion",   "tags_en": "abaya",  "tags_ar": "عباءة"},
    {"sku_id": "FSH-0012", "name_en": "Embroidered Abaya", "name_ar": "عباءة مطرزة",  "category": "fashion",    "tags_en": "abaya",  "tags_ar": "عباءة"},
    {"sku_id": "FSH-0004", "name_en": "Linen Shirt",      "name_ar": "قميص كتان",    "category": "fashion",     "tags_en": "linen",  "tags_ar": "كتان"},
    {"sku_id": "FSH-0019", "name_en": "Linen Shirt Pro",  "name_ar": "قميص كتان برو", "category": "fashion",    "tags_en": "linen",  "tags_ar": "كتان"},
    {"sku_id": "FSH-0006", "name_en": "Kandura White",    "name_ar": "كندورة بيضاء", "category": "fashion",     "tags_en": "men",    "tags_ar": "رجالي"},
    {"sku_id": "ELC-0121", "name_en": "Wireless Earbuds", "name_ar": "سماعات لاسلكية", "category": "electronics", "tags_en": "audio", "tags_ar": "صوت"},
    {"sku_id": "FOD-0301", "name_en": "Medjool Dates 1kg", "name_ar": "تمر مجدول كيلو", "category": "food",    "tags_en": "dates",  "tags_ar": "تمر"},
]


def hit(sku, score):
    """One result in the REAL retriever shape."""
    row = next(r for r in CATALOGUE if r["sku_id"] == sku)
    return {
        "sku_id": sku, "name_en": row["name_en"], "name_ar": row["name_ar"],
        "category": row["category"], "similarity_score": score,
        "document_text_preview": f"{row['name_en']} | {row['name_ar']} | {row['tags_en']} | {row['tags_ar']}",
        "price_aed": 100.0, "sales": {},
    }


class FakeLoader:
    def __init__(self, rows):
        self.rows = rows

    def load_catalogue(self):
        return self.rows


class FakeRetriever:
    """Mimics RetailRetriever: synchronous search() returning a DICT, plus .loader."""

    def __init__(self, results, rows=CATALOGUE):
        self._results = results
        self.loader = FakeLoader(list(rows))
        self.calls = []

    def search(self, query, n_results=5, **kwargs):
        self.calls.append((query, n_results))
        return {"query": query, "query_language": "en", "results": self._results,
                "result_count": len(self._results), "confidence_gate": 0.72, "filters_applied": {}}


def run(query, retriever, intent="search"):
    rag_node.set_retriever(retriever)
    state = {"query": query, "intent": intent, "reasoning_trace": []}
    return asyncio.run(rag_node.rag_retrieval_node(state))


# ── the bug: real retriever shape must work ───────────────────────────────────

def test_real_dict_shape_works_and_maps_fields():
    r = FakeRetriever([hit("FSH-0001", 0.88), hit("FSH-0004", 0.84), hit("FSH-0006", 0.83)])
    out = run("summer dress", r)
    assert "error" not in out
    assert out["fallback_triggered"] is False
    top = out["retrieved_chunks"][0]
    assert top["sku_id"] == "FSH-0001"
    assert top["product_name_en"] == "Summer Dress"      # name_en -> product_name_en
    assert "Summer Dress" in top["content"]              # document_text_preview -> content
    assert abs(out["retrieval_margin"] - 0.04) < 1e-9


def test_legacy_list_shape_is_still_accepted():
    legacy = [{"sku_id": "FSH-0001", "product_name_en": "Summer Dress", "product_name_ar": "فستان صيفي",
               "category": "fashion", "similarity_score": 0.90, "content": "Summer Dress | summer"},
              {"sku_id": "FSH-0004", "product_name_en": "Linen Shirt", "product_name_ar": "قميص كتان",
               "category": "fashion", "similarity_score": 0.80, "content": "Linen Shirt | linen"}]
    class BareList(FakeRetriever):
        def search(self, query, n_results=5, **kw):
            self.calls.append((query, n_results))
            return self._results            # a bare LIST, the shape the Phase 2 mocks used

    out = run("summer dress", BareList(legacy, rows=[]))
    assert "error" not in out and out["fallback_triggered"] is False
    assert out["retrieved_chunks"][0]["product_name_en"] == "Summer Dress"


def test_unknown_result_shape_falls_back_without_crashing():
    class Weird(FakeRetriever):
        def search(self, query, n_results=5, **kw):
            return "not a dict or list"
    out = run("summer dress", Weird([]))
    assert out["fallback_triggered"] is True


# ── margin on distinct products ───────────────────────────────────────────────

def test_variants_are_collapsed_before_margin():
    # Same-family variants score 0.870 / 0.869; next different product 0.830.
    # Per-SKU top1-top3 would be 0.002 (looks uncertain); per-product margin is 0.040.
    r = FakeRetriever([hit("FSH-0001", 0.870), hit("FSH-0016", 0.869), hit("FSH-0004", 0.830), hit("FSH-0006", 0.820)])
    out = run("nike shoes", r)   # entity absent, so only the margin can save it
    assert abs(out["retrieval_margin"] - 0.040) < 1e-9
    assert out["fallback_triggered"] is False
    skus = [c["sku_id"] for c in out["retrieved_chunks"]]
    assert skus == ["FSH-0001", "FSH-0004", "FSH-0006"]   # one SKU per product


def test_flat_scores_and_missing_entity_fall_back():
    r = FakeRetriever([hit("FSH-0001", 0.800), hit("FSH-0004", 0.799), hit("FSH-0006", 0.798)])
    out = run("Nike shoes", r)
    assert out["fallback_triggered"] is True


def test_flat_scores_but_entity_present_does_not_fall_back():
    # Principle 2: BOTH conditions are required.
    r = FakeRetriever([hit("FSH-0001", 0.800), hit("FSH-0004", 0.799), hit("FSH-0006", 0.798)])
    out = run("summer dress", r)
    assert out["retrieval_margin"] < rag_node.MARGIN_THRESHOLD
    assert out["fallback_triggered"] is False


def test_below_floor_falls_back_even_with_big_margin():
    r = FakeRetriever([hit("FSH-0001", 0.70), hit("FSH-0004", 0.50), hit("FSH-0006", 0.40)])
    assert run("summer dress", r)["fallback_triggered"] is True


def test_three_letter_arabic_term_counts_as_entity():
    r = FakeRetriever([hit("FOD-0301", 0.800), hit("FSH-0004", 0.799), hit("FSH-0006", 0.798)])
    out = run("تمر", r)   # 3 letters: the old len>3 rule ignored it
    assert out["fallback_triggered"] is False


def test_gulf_spelling_of_abaya_matches_catalogue_spelling():
    # Smoke test on the real catalogue: "عباية" (customer spelling) vs "عباءة" (catalogue) was not
    # recognised as the same word, so a near-tie between two real abayas became a fallback.
    r = FakeRetriever([hit("FSH-0012", 0.792), hit("FSH-0002", 0.789), hit("FSH-0004", 0.774)])
    out = run("عباية", r)
    assert out["retrieval_margin"] < rag_node.MARGIN_THRESHOLD
    assert out["fallback_triggered"] is False


def test_product_not_in_catalogue_still_falls_back_in_arabic():
    r = FakeRetriever([hit("FSH-0002", 0.785), hit("FSH-0001", 0.784), hit("FSH-0004", 0.777)])
    assert run("ايفون 15", r)["fallback_triggered"] is True


# ── exact SKU route ───────────────────────────────────────────────────────────

def test_exact_sku_is_looked_up_without_vector_search():
    r = FakeRetriever([])
    out = run("how many units left of fsh-0004", r, intent="reorder")
    assert out["fallback_triggered"] is False
    assert out["retrieval_confidence"] == 1.0
    assert out["retrieved_chunks"][0]["sku_id"] == "FSH-0004"
    assert r.calls == []   # vector search was not used


def test_unknown_sku_is_a_hard_fallback_when_catalogue_is_loaded():
    r = FakeRetriever([hit("FSH-0001", 0.9)])
    out = run("price of FSH-9999", r, intent="reorder")
    assert out["fallback_triggered"] is True
    assert "not found" in out["reasoning_trace"][-1]
    assert r.calls == []


def test_sku_route_is_skipped_when_catalogue_is_unavailable():
    # A test double without a catalogue must not turn every SKU query into a fallback.
    r = FakeRetriever([hit("FSH-0001", 0.88), hit("FSH-0004", 0.84), hit("FSH-0006", 0.83)], rows=[])
    out = run("why is SKU-4421 underperforming", r, intent="analysis")
    assert r.calls, "should have fallen through to semantic search"
    assert out["fallback_triggered"] is False


# ── robustness ────────────────────────────────────────────────────────────────

def test_search_exception_triggers_fallback_with_error():
    class Boom(FakeRetriever):
        def search(self, query, n_results=5, **kw):
            raise RuntimeError("chroma down")
    out = run("summer dress", Boom([]))
    assert out["fallback_triggered"] is True
    assert "chroma down" in out["error"]


def test_search_runs_off_the_event_loop():
    import threading
    seen = {}

    class Spy(FakeRetriever):
        def search(self, query, n_results=5, **kw):
            seen["thread"] = threading.current_thread()
            return super().search(query, n_results, **kw)

    run("summer dress", Spy([hit("FSH-0001", 0.88), hit("FSH-0004", 0.84)]))
    assert seen["thread"] is not threading.main_thread()


# ── sku_lookup helpers ────────────────────────────────────────────────────────

def test_extract_sku_candidates():
    assert extract_sku_candidates("stock of fsh-0014 and ELC_0123") == ["FSH-0014", "ELC-0123"]
    assert extract_sku_candidates("why is SKU-4421 underperforming") == ["SKU-4421"]
    assert extract_sku_candidates("USB-C hub and T-Shirt") == []


def test_family_map_groups_variants_within_a_category():
    fam = build_family_map(CATALOGUE)
    assert fam["FSH-0016"] == fam["FSH-0001"] == "Summer Dress"
    assert fam["FSH-0019"] == "Linen Shirt"
    assert fam["FSH-0006"] == "Kandura White"   # no shorter prefix: its own family