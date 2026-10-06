"""
agents/nodes/fallback_node.py
------------------------------
Deterministic answer for the fallback path (architecture principle 10, field-type fallback split).

Before this node existed, a fallback ended the graph right after the RAG node and the caller got
recommendation=None: the "insufficient data" reply (factual) and the partial answer with provenance
(narrative) were specified but never produced.

Policy:
  FACTUAL intents (reorder, analysis): hard fallback. "Insufficient data" and one hint on how to
      ask again. No product names, no numbers: a wrong confident number is unrecoverable.
  NARRATIVE intents (search, trend): partial answer with provenance and a LOW confidence label:
      "Based on [source], [closest items]. Note: this is partial - verify before acting."
      The right product is in the top 3 for every query we measured (eval/RESULTS.md, section 2),
      so showing the closest distinct products lets the retailer pick, instead of a dead end.
  Technical error, unknown SKU, or no results: "insufficient data" for every intent.

No LLM is used here on purpose: it is instant, cannot hallucinate, and works when the Groq
free-tier rate limit is hit.

Arabic wording (query_language "ar" or "mixed") is plain Modern Standard Arabic and has not been
reviewed by a native Gulf Arabic speaker. Phase 4 is where bilingual response polish belongs.
"""

from __future__ import annotations

import structlog

from services.intent_classifier import is_factual_intent
from services.sku_lookup import extract_sku_candidates

from ..state import AgentState

logger = structlog.get_logger(__name__)

_TEXT = {
    "en": {
        "insufficient": "Insufficient data to answer reliably. Please give the exact product name or SKU.",
        "error":        "Insufficient data: a technical error stopped the catalogue search, so no answer was generated.",
        "sku_missing":  "Insufficient data: SKU {sku} was not found in the catalogue.",
        "none":         "Insufficient data: nothing in the catalogue matched this query.",
        "partial_head": (
            "I could not confirm an exact match for your query. Based on the product catalogue "
            "(semantic search), the closest items are (confidence: LOW, top relevance {top:.2f}, "
            "gap to the next product {margin:.3f}):"
        ),
        "partial_note": "Note: this is partial - verify before acting.",
    },
    "ar": {
        "insufficient": "لا تتوفر بيانات كافية للإجابة بشكل موثوق. يرجى تحديد اسم المنتج أو رمز SKU بدقة.",
        "error":        "لا تتوفر بيانات كافية: حدث خطأ تقني أثناء البحث في الكتالوج، لذلك لم يتم إنشاء إجابة.",
        "sku_missing":  "لا تتوفر بيانات كافية: الرمز {sku} غير موجود في الكتالوج.",
        "none":         "لا تتوفر بيانات كافية: لم يتطابق أي منتج في الكتالوج مع هذا الطلب.",
        "partial_head": (
            "لم أتمكن من تأكيد تطابق دقيق مع طلبك. بناءً على كتالوج المنتجات (بحث دلالي)، "
            "أقرب المنتجات هي (مستوى الثقة: منخفض، أعلى درجة تطابق {top:.2f}، "
            "الفارق عن المنتج التالي {margin:.3f}):"
        ),
        "partial_note": "ملاحظة: هذه إجابة جزئية - يرجى التحقق قبل اتخاذ أي إجراء.",
    },
}


def _format_candidates(chunks: list) -> str:
    return "\n".join(
        f"  • {c['product_name_en']} ({c['product_name_ar']}) | SKU: {c['sku_id']} "
        f"| relevance {c['similarity_score']:.3f}"
        for c in chunks
    )


async def fallback_node(state: AgentState) -> dict:
    """Produce the user-facing answer when retrieval was not confident enough."""
    text   = _TEXT["ar" if state.get("query_language", "en") in ("ar", "mixed") else "en"]
    intent = state.get("intent", "search")
    chunks = state.get("retrieved_chunks") or []
    skus   = extract_sku_candidates(state.get("query", ""))

    if state.get("error"):
        kind, answer = "technical_error", text["error"]
    elif skus and not chunks:
        kind, answer = "unknown_sku", text["sku_missing"].format(sku=", ".join(skus))
    elif is_factual_intent(intent):
        kind, answer = "hard_fallback", text["insufficient"]
    elif chunks:
        kind = "partial_answer"
        answer = "\n".join([
            text["partial_head"].format(
                top=state.get("retrieval_confidence", 0.0),
                margin=state.get("retrieval_margin", 0.0),
            ),
            _format_candidates(chunks),
            "",
            text["partial_note"],
        ])
    else:
        kind, answer = "no_results", text["none"]

    logger.info("fallback_node", kind=kind, intent=intent, candidates=len(chunks))
    trace = f"Fallback: {kind} (intent={intent}); deterministic answer, no LLM used"
    return {
        "recommendation":            answer,
        "recommendation_confidence": 0.0,
        "reasoning_trace":           state["reasoning_trace"] + [trace],
    }