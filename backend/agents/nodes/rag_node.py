"""
agents/nodes/rag_node.py
-------------------------
Node 1: RAG retrieval over the bilingual product catalogue.

Wraps the Phase 1 RetailRetriever with margin-based confidence routing
and the field-type fallback split (architecture principle 10).

CHANGES vs the Phase 2 version (all driven by eval/RESULTS.md and eval/smoke_agent.py):

  1. ADAPTER FIX (the bug). RetailRetriever.search() returns a DICT
     {"results": [...], ...} with fields name_en / document_text_preview.
     The old node treated it as a LIST with product_name_en / content, so with real
     data every query crashed ('str' object has no attribute 'get') and fell back.
     The unit tests passed because their mock returned the old list shape.
     _normalise_results() now accepts both shapes.

  2. MARGIN IS COMPUTED ON DISTINCT PRODUCTS. Variants of one product
     ("Summer Dress", "Summer Dress Pro") are collapsed to a single product first.
     Without this the top-1 vs top-3 gap is ~0.005 for right AND wrong answers.
     Margin = best product's score minus the NEXT DIFFERENT product's score.
     (Principle 2 says top-1 vs top-3; the evidence we measured is top-1 vs next
     distinct product. Same idea, product-level instead of SKU-level.)

  3. MARGIN_THRESHOLD 0.05 -> 0.01. 0.05 would flag nearly every query on this catalogue.
     0.01 is a HYPOTHESIS from 75 queries tuned on the same data it was scored on
     (kept ~78% of correct answers, caught ~90% of wrong / not-in-catalogue ones).
     Re-check it on a held-out set before quoting it.

  4. EXACT SKU ROUTE. SKU codes are not in the embedded text, so vector search cannot
     find them. A catalogue SKU in the query is looked up exactly (confidence 1.0).
     An unknown SKU is a hard fallback. Only active when the catalogue could be loaded.

  5. Entity check counts 3-letter terms (تمر, دلة, oud ...) and normalises Arabic letters.
     The old rule (len > 3) ignored them.

  6. search() is synchronous and runs the embedding model, so it now runs in a worker
     thread instead of blocking the event loop.

Fallback requires BOTH conditions (architecture principle 2):
  margin below threshold (uncertain) AND the queried entity absent from the top results.
  The 0.72 floor stays as an additional minimum (it never fired in measurements).
"""

from __future__ import annotations

import asyncio
import re

import structlog

from ..state import AgentState, RetrievedChunk
from services.intent_classifier import extract_sku_references
from services.sku_lookup import (
    CatalogueIndex,
    build_catalogue_index,
    extract_sku_candidates,
)

logger = structlog.get_logger(__name__)

# ── Configuration constants ────────────────────────────────────────────────────

CONFIDENCE_FLOOR = 0.72
MARGIN_THRESHOLD = 0.01   # hypothesis, see module docstring (was 0.05)
TOP_K_RETRIEVAL  = 20     # fetch enough SKUs that variant-collapsing still leaves >= 3 products
CHUNKS_TO_RETURN = 3      # distinct products returned

# ── Module-level retriever singleton ──────────────────────────────────────────

_retriever = None
_index_cache: dict = {"retriever": None, "index": None}


def set_retriever(retriever) -> None:
    """
    Store the Phase 1 RetailRetriever instance for use by this node.
    Called once from main.py lifespan after retriever.initialize() completes.
    Also resets the catalogue index, so calling it again after a data upload refreshes it.
    """
    global _retriever
    _retriever = retriever
    _index_cache["retriever"] = None
    _index_cache["index"] = None
    logger.info("rag_node_retriever_registered", retriever_type=type(retriever).__name__)


def get_retriever():
    """Return the stored retriever. Raises RuntimeError if set_retriever() was never called."""
    if _retriever is None:
        raise RuntimeError(
            "RetailRetriever not initialised. "
            "Ensure set_retriever() is called in main.py lifespan."
        )
    return _retriever


def _get_index(retriever) -> CatalogueIndex:
    """
    Build (once) the SKU lookup and product-family map from the retriever's catalogue.
    If the catalogue cannot be read (e.g. a test double), return an empty index: the
    exact-SKU route and variant collapsing are then skipped and behaviour degrades to
    plain per-SKU retrieval.
    """
    if _index_cache["retriever"] is retriever and _index_cache["index"] is not None:
        return _index_cache["index"]
    try:
        index = build_catalogue_index(list(retriever.loader.load_catalogue()))
    except Exception as exc:  # noqa: BLE001 - degrade, never crash retrieval
        logger.warning("rag_catalogue_index_unavailable", error=str(exc))
        index = CatalogueIndex()
    _index_cache["retriever"] = retriever
    _index_cache["index"] = index
    logger.info("rag_catalogue_index_ready", skus=len(index.by_sku), families=len(set(index.family.values())))
    return index


# ── Helper functions ───────────────────────────────────────────────────────────

_ARABIC_MARKS = re.compile(r"[ً-ْـ]")   # diacritics + tatweel
_WORD = re.compile(r"[\w؀-ۿ]+")

_STOPWORDS = {
    # English
    "the", "for", "and", "with", "how", "many", "much", "what", "which", "are", "any", "have",
    "left", "units", "unit", "stock", "need", "show", "give", "why", "from", "that", "this",
    "can", "you", "buy", "get", "all", "was", "our", "about", "price", "cost",
    # Arabic
    "من", "في", "على", "عن", "هل", "ما", "ماذا", "كم", "هذا", "هذه", "مع", "او", "أو",
    "ابغى", "ابي", "أبي", "أبغى", "عندكم", "عندك", "شنو", "وين", "اللي", "الي", "سعر", "كمية",
}


def _norm(text: str) -> str:
    """Lowercase + light Arabic normalisation (diacritics, alef forms, ya, ta marbuta, hamza)."""
    t = _ARABIC_MARKS.sub("", (text or "").lower())
    t = re.sub("[أإآ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    # hamza seats: the catalogue spells عباءة, customers type عباية (same word). Entity check only.
    return t.replace("ئ", "ي").replace("ؤ", "و").replace("ء", "ي")


def _normalise_results(raw) -> list[dict]:
    """
    Accept both retriever output shapes and return a list of canonical dicts:
      real:   {"results": [{"name_en", "name_ar", "document_text_preview", ...}], ...}
      legacy: [{"product_name_en", "product_name_ar", "content", ...}]
    Sorted best-first.
    """
    if isinstance(raw, dict):
        items = raw.get("results") or []
    elif isinstance(raw, (list, tuple)):
        items = raw
    else:
        logger.warning("rag_unexpected_result_shape", result_type=type(raw).__name__)
        items = []

    out: list[dict] = []
    for r in items:
        if not isinstance(r, dict):
            continue
        out.append({
            "sku_id":           str(r.get("sku_id", "")),
            "name_en":          r.get("name_en") or r.get("product_name_en") or "",
            "name_ar":          r.get("name_ar") or r.get("product_name_ar") or "",
            "category":         r.get("category", ""),
            "similarity_score": float(r.get("similarity_score", 0.0) or 0.0),
            "content":          r.get("content") or r.get("document_text_preview") or "",
        })
    out.sort(key=lambda r: r["similarity_score"], reverse=True)
    return out


def _collapse_by_product(results: list[dict], index: CatalogueIndex) -> list[tuple[str, dict, int]]:
    """
    Collapse variants into products. Returns [(product, best_result, variants_seen), ...]
    best first. Without a catalogue index the key falls back to the English name, then the SKU.
    """
    best: dict[str, list] = {}
    for r in results:  # already best-first
        key = index.family.get(r["sku_id"]) or r["name_en"] or r["sku_id"]
        if key in best:
            best[key][1] += 1
        else:
            best[key] = [r, 1]
    return [(k, v[0], v[1]) for k, v in best.items()]


def _entity_present(query: str, sku_refs: list[str], results: list[dict]) -> bool:
    """Does the queried entity (SKU or a significant term) appear in the top results?"""
    haystack = _norm(" ".join(
        f"{r['sku_id']} {r['name_en']} {r['name_ar']} {r['content']}" for r in results
    ))
    if any(_norm(s) in haystack for s in sku_refs):
        return True
    terms = [t for t in _WORD.findall(_norm(query)) if len(t) >= 3 and t not in _STOPWORDS]
    return any(t in haystack for t in terms)


def _chunk(r: dict, score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        sku_id=r["sku_id"],
        product_name_en=r["name_en"],
        product_name_ar=r["name_ar"],
        category=r["category"],
        similarity_score=round(r["similarity_score"] if score is None else score, 4),
        content=r["content"],
    )


def _fallback(state: AgentState, trace: str, error: str | None = None) -> dict:
    update = {
        "retrieved_chunks":     [],
        "retrieval_confidence": 0.0,
        "retrieval_margin":     0.0,
        "fallback_triggered":   True,
        "reasoning_trace":      state["reasoning_trace"] + [trace],
    }
    if error:
        update["error"] = error
    return update


# ── Node implementation ────────────────────────────────────────────────────────

async def rag_retrieval_node(state: AgentState) -> dict:
    """
    Node 1: Retrieve relevant product catalogue chunks.

    Returns a partial AgentState update - LangGraph merges it with existing state.
    """
    query  = state["query"]
    intent = state["intent"]

    try:
        retriever = get_retriever()
        index     = _get_index(retriever)

        sku_refs   = list(dict.fromkeys(extract_sku_references(query) + extract_sku_candidates(query)))
        candidates = extract_sku_candidates(query)
        logger.info("rag_node_start", query=query[:60], intent=intent, sku_refs=sku_refs)

        # ── Exact SKU route (only when the real catalogue is loaded) ───────
        if candidates and index:
            hits = [index.by_sku[c] for c in candidates if c in index.by_sku]
            if hits:
                chunks = [
                    _chunk({
                        "sku_id":           str(row["sku_id"]),
                        "name_en":          row.get("name_en", ""),
                        "name_ar":          row.get("name_ar", ""),
                        "category":         row.get("category", ""),
                        "similarity_score": 1.0,
                        "content":          " | ".join(
                            str(row.get(k, "")) for k in ("name_en", "name_ar", "category", "tags_en", "tags_ar")
                            if row.get(k)
                        ),
                    })
                    for row in hits[:CHUNKS_TO_RETURN]
                ]
                trace = (
                    f"RAG retrieval: exact SKU match {[c['sku_id'] for c in chunks]} "
                    f"(catalogue lookup, no vector search), fallback=no"
                )
                logger.info("rag_node_exact_sku", skus=[c["sku_id"] for c in chunks])
                return {
                    "retrieved_chunks":     chunks,
                    "retrieval_confidence": 1.0,
                    "retrieval_margin":     1.0,
                    "fallback_triggered":   False,
                    "reasoning_trace":      state["reasoning_trace"] + [trace],
                }
            logger.info("rag_node_unknown_sku", skus=candidates)
            return _fallback(
                state,
                f"RAG retrieval: SKU {candidates} not found in catalogue - insufficient data",
            )

        # ── Semantic route ─────────────────────────────────────────────────
        raw = await asyncio.to_thread(retriever.search, query, n_results=TOP_K_RETRIEVAL)
        results = _normalise_results(raw)

        if not results:
            logger.warning("rag_no_results", query=query[:60])
            return _fallback(state, "RAG retrieval: no results returned - fallback triggered")

        products = _collapse_by_product(results, index)
        scores   = [p[1]["similarity_score"] for p in products]
        top1     = scores[0]
        top2     = scores[1] if len(scores) > 1 else CONFIDENCE_FLOOR  # one distinct product = unambiguous
        top3     = scores[2] if len(scores) > 2 else top2
        margin   = top1 - top2

        top_results      = [p[1] for p in products[:CHUNKS_TO_RETURN]]
        below_floor      = top1 < CONFIDENCE_FLOOR
        uncertain_margin = margin < MARGIN_THRESHOLD
        entity_present   = _entity_present(query, sku_refs, top_results)

        # Both conditions required to trigger the margin fallback (principle 2)
        fallback = below_floor or (uncertain_margin and not entity_present)

        chunks = [_chunk(r) for r in top_results]
        trace = (
            f"RAG retrieval: top1={top1:.3f}, top2={top2:.3f}, top3={top3:.3f}, "
            f"margin={margin:.3f} (best product vs next different product), "
            f"entity_present={entity_present}, fallback={'yes' if fallback else 'no'}, "
            f"chunks_returned={len(chunks)}, "
            f"duplicates_removed={len(results) - len(products)}"
        )
        logger.info("rag_node_complete", top1=top1, margin=margin, fallback=fallback, chunks=len(chunks))

        return {
            "retrieved_chunks":     chunks,
            "retrieval_confidence": top1,
            "retrieval_margin":     margin,
            "fallback_triggered":   fallback,
            "reasoning_trace":      state["reasoning_trace"] + [trace],
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("rag_node_error", error=str(exc), query=query[:60])
        return _fallback(
            state,
            f"RAG retrieval error: {exc}",
            error=f"RAG retrieval failed: {exc}",
        )