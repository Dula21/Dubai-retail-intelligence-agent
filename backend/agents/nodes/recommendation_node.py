"""
agents/nodes/recommendation_node.py
-------------------------------------
Node 5: Generate final recommendation via Groq LLM.

Synthesises all upstream signals into a human-readable recommendation.

Hallucination prevention:
  All data passed to the LLM is injected from AgentState. The prompt
  explicitly instructs the LLM not to use numbers or facts not present
  in the provided data — same grounding pattern as Dubai Property Intelligence.

Groq free-tier rate limit handling:
  On RateLimitError (HTTP 429), degrades gracefully: returns the reasoning
  trace as the recommendation rather than a 500 error. Lower confidence
  score signals to the caller that synthesis was unavailable.

Bilingual output:
  query_language="ar"    → response in Arabic
  query_language="mixed" → Arabic with English SKU codes and numbers
  query_language="en"    → English
"""

from __future__ import annotations

import os

import structlog
from groq import AsyncGroq, RateLimitError

from ..state import AgentState, InventorySignal, SalesSignal, SeasonalitySignal

logger = structlog.get_logger(__name__)

# ── Groq client singleton ──────────────────────────────────────────────────────

_groq_client: AsyncGroq | None = None


def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set")
        _groq_client = AsyncGroq(api_key=api_key)
    return _groq_client


# ── Prompt templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an autonomous retail merchandising AI for Dubai mid-market retailers.
Your job is to generate clear, actionable reorder and stock recommendations.

Rules you must follow without exception:
1. Only use numbers and facts present in the DATA section below — never invent figures
2. If data for a field is missing or marked -1, state "insufficient data" for that field
3. Cite which data source supports each claim (e.g. "Based on 30-day sales average")
4. Keep the recommendation under 200 words — retailers are busy
5. End with a clear action: REORDER NOW / MONITOR / INVESTIGATE / NO ACTION NEEDED
6. If responding in Arabic, keep SKU codes, AED values, and numbers in their original form
"""

RECOMMENDATION_PROMPT = """\
QUERY: {query}

DATA:

RETRIEVED PRODUCTS:
{products_text}

SALES SIGNALS (last 30 days):
{sales_text}

INVENTORY STATUS:
{inventory_text}

UAE SEASONALITY:
{seasonality_text}

RETRIEVAL CONFIDENCE: {confidence:.2f} (threshold: 0.72)

Based on the data above, generate a retail merchandising recommendation in {language}.
Structure your response:
1. What the data shows (2–3 sentences, citing sources)
2. The recommendation (specific and actionable)
3. The action label on the last line: REORDER NOW / MONITOR / INVESTIGATE / NO ACTION NEEDED
"""


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _format_products(chunks: list) -> str:
    if not chunks:
        return "No products retrieved"
    return "\n".join(
        f"  • {c['product_name_en']} ({c['product_name_ar']}) "
        f"| SKU: {c['sku_id']} | Category: {c['category']} "
        f"| Relevance: {c['similarity_score']:.3f}"
        for c in chunks
    )


def _format_sales(signals: list[SalesSignal]) -> str:
    if not signals:
        return "No sales data available"
    return "\n".join(
        f"  • {s['sku_id']}: {s['avg_daily_units']:.1f} units/day "
        f"| trend={s['trend_direction']}"
        + (f", peaks during {s['peak_period']}" if s["peak_period"] else "")
        + f" | data confidence={s['confidence']:.2f}"
        for s in signals
    )


def _format_inventory(signals: list[InventorySignal]) -> str:
    if not signals:
        return "No inventory data available"
    lines = []
    for i in signals:
        if i["current_stock"] == -1:
            lines.append(f"  • {i['sku_id']}: insufficient data")
        else:
            critical_flag = " ⚠ CRITICAL" if i["is_critical"] else ""
            runway        = f"{i['runway_days']:.0f}" if i["runway_days"] < 9999 else "∞"
            lines.append(
                f"  • {i['sku_id']}: {i['current_stock']} units in stock "
                f"| {runway} days runway "
                f"| reorder point={i['reorder_point']}{critical_flag}"
            )
    return "\n".join(lines)


def _format_seasonality(signal: SeasonalitySignal) -> str:
    if not signal or not signal.get("upcoming_event"):
        return "No UAE retail events in the next 45 days — baseline demand"
    return (
        f"  • {signal['upcoming_event']} in {signal['days_until_event']} days "
        f"| expected demand: {signal['expected_demand_multiplier']}x baseline "
        f"| alert level: {signal['alert_level'].upper()}"
    )


def _select_language_label(query_language: str) -> str:
    if query_language == "ar":
        return "Arabic"
    if query_language == "mixed":
        return "Arabic (keep SKU codes and numbers in English)"
    return "English"


def _compute_confidence(state: AgentState) -> float:
    """
    Composite confidence score for the final recommendation.
    Weights: retrieval 40%, sales data 30%, inventory data 20%, no error 10%.
    """
    retrieval_score = state.get("retrieval_confidence", 0.0) * 0.4
    has_sales       = 0.3 if state.get("sales_signals")     else 0.0
    has_inventory   = 0.2 if state.get("inventory_signals") else 0.0
    no_error        = 0.1 if not state.get("error")         else 0.0
    return round(min(retrieval_score + has_sales + has_inventory + no_error, 1.0), 2)


# ── Node implementation ────────────────────────────────────────────────────────

async def recommendation_node(state: AgentState) -> dict:
    """
    Node 5: Generate final recommendation via Groq LLM.

    Args:
        state: Full AgentState with all upstream node outputs

    Returns:
        Partial state dict with recommendation, confidence, and trace entry
    """
    confidence = _compute_confidence(state)
    language   = _select_language_label(state.get("query_language", "en"))

    prompt = RECOMMENDATION_PROMPT.format(
        query=state["query"],
        products_text=_format_products(state.get("retrieved_chunks", [])),
        sales_text=_format_sales(state.get("sales_signals", [])),
        inventory_text=_format_inventory(state.get("inventory_signals", [])),
        seasonality_text=_format_seasonality(state.get("seasonality_signal", {})),
        confidence=state.get("retrieval_confidence", 0.0),
        language=language,
    )

    try:
        client   = get_groq_client()
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=400,
            temperature=0.15,
        )
        recommendation = response.choices[0].message.content.strip()

        logger.info(
            "recommendation_generated",
            confidence=confidence,
            language=language,
            tokens_used=response.usage.total_tokens if response.usage else "unknown",
        )
        trace = (
            f"Recommendation generated via Groq llama-3.1-8b-instant "
            f"| language={language} | confidence={confidence:.2f}"
        )

    except RateLimitError:
        logger.warning("groq_rate_limit_hit", query=state["query"][:50])
        trace_summary  = " → ".join(state.get("reasoning_trace", []))
        recommendation = (
            f"[Rate limit reached — LLM synthesis unavailable. "
            f"Agent reasoning trace:]\n\n{trace_summary}\n\n"
            f"Action: Review the trace above and consult inventory data directly."
        )
        confidence = max(confidence - 0.2, 0.0)
        trace      = "Recommendation: Groq rate limit hit — trace-only fallback returned"

    except Exception as exc:
        logger.error("recommendation_node_error", error=str(exc))
        recommendation = (
            f"Recommendation generation failed: {exc}. "
            f"Agent trace: {' → '.join(state.get('reasoning_trace', []))}"
        )
        confidence = 0.0
        trace      = f"Recommendation error: {exc}"

    return {
        "recommendation":            recommendation,
        "recommendation_confidence": confidence,
        "reasoning_trace":           state["reasoning_trace"] + [trace],
    }
