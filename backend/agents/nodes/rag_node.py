"""
agents/nodes/rag_node.py
-------------------------
Node 1: RAG retrieval over the bilingual product catalogue.

Wraps the Phase 1 RetailRetriever with margin-based confidence routing
and the field-type fallback split (architecture principle 10).

FIX vs earlier version:
  set_retriever() and get_retriever() were MISSING from the pasted version.
  main.py calls set_retriever(retriever) in its lifespan — without these
  functions, the entire application crashes on startup with ImportError.
  These are restored here as the module-level singleton pattern.

Margin-based routing rationale (from Rajat Shrivastav feedback):
  Absolute cosine thresholds drift with query length and phrasing.
  A 3-word Arabic query and a 15-word English query produce different
  score ranges for identical retrieval quality. The margin (top-1 minus
  top-3 gap) is query-length invariant — a small gap means the retriever
  is uncertain regardless of the absolute score level.

Fallback requires BOTH conditions:
  1. Margin below threshold (uncertain retrieval), AND
  2. None of top-3 chunks contain the specific queried entity
  Either condition alone is not enough — prevents over-triggering fallback.

Deduplication before margin calculation:
  If the same content appears in multiple source documents, top-3 scores
  look artificially flat. Deduplicate by content hash before computing
  margin — otherwise good retrievals get flagged as uncertain.
"""

from __future__ import annotations

import hashlib

import structlog

from ..state import AgentState, RetrievedChunk
from services.intent_classifier import extract_sku_references, is_factual_intent

logger = structlog.get_logger(__name__)

# ── Configuration constants ────────────────────────────────────────────────────

CONFIDENCE_FLOOR = 0.72
MARGIN_THRESHOLD = 0.05
TOP_K_RETRIEVAL  = 5
CHUNKS_TO_RETURN = 3

# ── Module-level retriever singleton ──────────────────────────────────────────
# Set once at startup via set_retriever() called from main.py lifespan.
# All requests share this single instance — embedding model loaded once.

_retriever = None


def set_retriever(retriever) -> None:
    """
    Store the Phase 1 RetailRetriever instance for use by this node.
    Called once from main.py lifespan after retriever.initialize() completes.

    Engineering decision: module-level singleton over dependency injection.
    FastAPI's Depends() system works well for stateless dependencies but
    adds boilerplate for a heavy stateful object like an embedding model.
    At SME scale, one global instance is correct and simple.
    """
    global _retriever
    _retriever = retriever
    logger.info("rag_node_retriever_registered", retriever_type=type(retriever).__name__)


def get_retriever():
    """
    Return the stored retriever instance.
    Raises RuntimeError if set_retriever() was never called (startup bug).
    """
    if _retriever is None:
        raise RuntimeError(
            "RetailRetriever not initialised. "
            "Ensure set_retriever() is called in main.py lifespan."
        )
    return _retriever


# ── Helper functions ───────────────────────────────────────────────────────────

def _content_hash(content: str) -> str:
    """Stable hash of first 200 chars — sufficient for deduplication."""
    return hashlib.md5(content[:200].encode()).hexdigest()


def _entity_present_in_chunks(
    query: str,
    sku_refs: list[str],
    chunks: list[dict],
) -> bool:
    """
    Check whether the queried entity (SKU or product term) appears in
    any of the top-3 retrieved chunks. Used as second fallback condition.
    """
    query_lower = query.lower()
    for chunk in chunks:
        content_lower = chunk.get("content", "").lower()
        if any(sku.lower() in content_lower for sku in sku_refs):
            return True
        significant_terms = [w for w in query_lower.split() if len(w) > 3]
        if significant_terms and any(t in content_lower for t in significant_terms):
            return True
    return False


# ── Node implementation ────────────────────────────────────────────────────────

async def rag_retrieval_node(state: AgentState) -> dict:
    """
    Node 1: Retrieve relevant product catalogue chunks.

    Returns a partial AgentState update — LangGraph merges this with
    existing state. Only return keys you are updating.

    Args:
        state: Current AgentState

    Returns:
        Partial state dict with retrieval results and trace entry
    """
    query   = state["query"]
    intent  = state["intent"]
    sku_refs = extract_sku_references(query)

    logger.info("rag_node_start", query=query[:60], intent=intent, sku_refs=sku_refs)

    try:
        retriever    = get_retriever()
        raw_results: list[dict] = await retriever.retrieve(query, top_k=TOP_K_RETRIEVAL)

        if not raw_results:
            logger.warning("rag_no_results", query=query[:60])
            return {
                "retrieved_chunks":    [],
                "retrieval_confidence": 0.0,
                "retrieval_margin":     0.0,
                "fallback_triggered":   True,
                "reasoning_trace": state["reasoning_trace"] + [
                    "RAG retrieval: no results returned — fallback triggered"
                ],
            }

        # ── Deduplicate before margin calculation ──────────────────────────
        seen_hashes:    set[str]  = set()
        unique_results: list[dict] = []
        for r in raw_results:
            h = _content_hash(r.get("content", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_results.append(r)

        # ── Margin-based routing ───────────────────────────────────────────
        scores     = [r.get("similarity_score", 0.0) for r in unique_results]
        top1_score = scores[0] if scores else 0.0
        top3_score = scores[min(2, len(scores) - 1)]
        margin     = top1_score - top3_score

        below_floor      = top1_score < CONFIDENCE_FLOOR
        uncertain_margin = margin < MARGIN_THRESHOLD
        entity_present   = _entity_present_in_chunks(
            query, sku_refs, unique_results[:CHUNKS_TO_RETURN]
        )

        # Both conditions required to trigger fallback (architecture principle 2)
        fallback = below_floor or (uncertain_margin and not entity_present)

        # ── Build chunk objects ────────────────────────────────────────────
        chunks: list[RetrievedChunk] = [
            RetrievedChunk(
                sku_id=r.get("sku_id", ""),
                product_name_en=r.get("product_name_en", ""),
                product_name_ar=r.get("product_name_ar", ""),
                category=r.get("category", ""),
                similarity_score=round(r.get("similarity_score", 0.0), 4),
                content=r.get("content", ""),
            )
            for r in unique_results[:CHUNKS_TO_RETURN]
        ]

        trace = (
            f"RAG retrieval: top1={top1_score:.3f}, top3={top3_score:.3f}, "
            f"margin={margin:.3f}, entity_present={entity_present}, "
            f"fallback={'yes' if fallback else 'no'}, "
            f"chunks_returned={len(chunks)}, "
            f"duplicates_removed={len(raw_results) - len(unique_results)}"
        )

        logger.info(
            "rag_node_complete",
            top1=top1_score,
            margin=margin,
            fallback=fallback,
            chunks=len(chunks),
        )

        return {
            "retrieved_chunks":    chunks,
            "retrieval_confidence": top1_score,
            "retrieval_margin":     margin,
            "fallback_triggered":   fallback,
            "reasoning_trace": state["reasoning_trace"] + [trace],
        }

    except Exception as exc:
        logger.error("rag_node_error", error=str(exc), query=query[:60])
        return {
            "retrieved_chunks":    [],
            "retrieval_confidence": 0.0,
            "retrieval_margin":     0.0,
            "fallback_triggered":   True,
            "error": f"RAG retrieval failed: {exc}",
            "reasoning_trace": state["reasoning_trace"] + [
                f"RAG retrieval error: {exc}"
            ],
        }
