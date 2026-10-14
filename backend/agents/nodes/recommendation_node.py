"""
agents/nodes/recommendation_node.py
-------------------------------------
Node 5: Generate final recommendation via Groq LLM.

Synthesises all upstream signals into a human-readable recommendation.

Seasonality now arrives from the calendar-driven node with event dates, the supplier lead time and
an ORDER-BY date. The prompt shows the model all of it, including "order-by date PASSED N days ago",
so the answer can say plainly that the retailer is late instead of describing a comfortable window.

Planning answers keep two questions apart (TIMING section of the prompt): is stock critical today, and
is the EVENT order deadline urgent. A WATCH alert is never described as urgent; critical stock is
reported as a separate, current problem. No order quantities are stated: a reorder point is a
trigger level, and the deterministic quantity engine is Phase 3.

Hallucination prevention:
  All data passed to the LLM is injected from AgentState. The prompt
  explicitly instructs the LLM not to use numbers or facts not present
  in the provided data — same grounding pattern as Dubai Property Intelligence.

Model configuration:
  GROQ_MODEL       model id (default openai/gpt-oss-20b). Groq retires models, so this
                   is an env var: llama-3.1-8b-instant and llama-3.3-70b-versatile were
                   shut down on 2026-08-16 and now return 404 model_not_found.
  GROQ_MAX_TOKENS  completion budget (default 1500). gpt-oss models are reasoning models:
                   reasoning tokens count against this budget, so 400 can leave the
                   visible answer empty or cut off.

Failure handling (no raw provider errors reach the caller):
  RateLimitError (HTTP 429) or any other error -> a degraded answer: the deterministic
  reasoning trace plus a plain-language reason. The full exception goes to the server log
  only. `error` is set in the returned state, so the API does not cache a degraded answer.

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

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_MAX_TOKENS = 1500

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


def _model_name() -> str:
    return os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def _max_tokens() -> int:
    try:
        return int(os.environ.get("GROQ_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    except ValueError:
        return DEFAULT_MAX_TOKENS


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
7. Dates in the answer must come from the DATA section. If the order-by date has PASSED, say so
   plainly and recommend expediting; never describe a passed deadline as upcoming
8. Never invent rules, formulas, safety-stock percentages or order quantities. If DATA contains no
   suggested quantity, say the quantity must be confirmed by the retailer
9. If DATA says the supplier lead time is assumed, or that event dates are an estimate, say so
10. A reorder point is a trigger level, NOT an order quantity. Never write "reorder N units" using a
    reorder point, a sales rate or the demand multiplier. State no order quantities at all
11. Keep two questions apart, using the TIMING section: (a) is stock critical TODAY, whatever the
    event, and (b) is the EVENT order deadline urgent. If the alert level is WATCH, say the event
    deadline is not yet urgent and give the date; a critical stock level is then a separate,
    current problem. Never describe a WATCH deadline as urgent, and never describe stock that
    is not critical as a problem
12. Two different numbers must never be mixed up. The EXPECTED demand multiplier comes from the
    calendar lift table (UAE SEASONALITY). The OBSERVED lift comes from past sales (SALES SIGNALS).
    Never call the expected multiplier a past, observed or historical lift, and never cite
    SALES SIGNALS for the calendar multiplier
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

TIMING (stock today and the event deadline are separate questions):
{timing_text}

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
    lines = []
    for c in chunks:
        line = (f"  • {c['product_name_en']} ({c['product_name_ar']}) "
                f"| SKU: {c['sku_id']} | Category: {c['category']}")
        if c.get("event_lift"):
            # Chosen by the calendar lift table, not by vector search: there is no similarity score to show.
            line += (f" | selected by {c.get('event_name', 'event')} lift table: "
                     f"expected lift {c['event_lift']}x")
        else:
            line += f" | Relevance: {c['similarity_score']:.3f}"
        lines.append(line)
    return "\n".join(lines)


def _format_sales(signals: list[SalesSignal]) -> str:
    if not signals:
        return "No sales data available"
    lines = []
    event = next((s["event_name"] for s in signals if s.get("event_name")), None)
    if event:
        observed = [s["observed_event_lift"] for s in signals if s.get("observed_event_lift") is not None]
        if observed:
            lines.append(f"  Observed {event} lift in past years across these SKUs: {min(observed)}x to "
                         f"{max(observed)}x (measured from sales history; the calendar multiplier in UAE "
                         "SEASONALITY is an expectation, not a past result)")
        else:
            lines.append(f"  Observed {event} lift: insufficient history for every SKU")
    for s in signals:
        line = f"  • {s['sku_id']}: {s['avg_daily_units']:.1f} units/day | trend={s['trend_direction']}"
        if s.get("event_name"):
            observed = s.get("observed_event_lift")
            line += (f" | observed {s['event_name']} lift in past years: {observed}x"
                     if observed is not None else
                     f" | observed {s['event_name']} lift: insufficient history")
        elif s["peak_period"]:
            line += f", peaks during {s['peak_period']}"
        line += f" | data confidence={s['confidence']:.2f}"
        lines.append(line)
    return "\n".join(lines)


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


def _order_by_line(signal: SeasonalitySignal) -> str | None:
    order_by = signal.get("order_by_date")
    if not order_by:
        return None
    days_left = signal.get("days_to_order_by")
    days_until = signal.get("days_until_event")
    if days_until is not None and days_until <= 0:
        status = "event already running"
    elif days_left is None:
        status = ""
    elif days_left < 0:
        status = f"PASSED {-days_left} days ago"
    elif days_left == 0:
        status = "today"
    else:
        status = f"{days_left} days left"
    return f"order-by date: {order_by}" + (f" ({status})" if status else "")


def _format_seasonality(signal: SeasonalitySignal) -> str:
    if not signal or not signal.get("upcoming_event"):
        return "No UAE retail event lifting these products in the next 90 days: baseline demand"

    days_until = signal.get("days_until_event")
    timing = ("running now" if days_until is not None and days_until <= 0
              else f"starts in {days_until} days")
    dates = (f" {signal['event_start']} to {signal['event_end']}"
             if signal.get("event_start") and signal.get("event_end") else "")
    lines = [
        f"  • {signal['upcoming_event']}{dates} ({timing})",
        f"    expected demand: {signal['expected_demand_multiplier']}x baseline "
        f"(from the synthetic calendar lift table)",
    ]
    if signal.get("lead_time_days") is not None:
        assumed = " (ASSUMED: no supplier data on the retrieved products)" if signal.get("lead_time_assumed") else ""
        lines.append(f"    supplier lead time: {signal['lead_time_days']} days{assumed}")
    order_line = _order_by_line(signal)
    if order_line:
        lines.append(f"    {order_line}")
    lines.append(f"    alert level: {signal['alert_level'].upper()}")
    if signal.get("date_basis") == "estimate_verify":
        lines.append("    note: event dates are an ESTIMATE until officially announced: verify")
    others = signal.get("other_upcoming") or []
    if others:
        lines.append("    also coming up: " + "; ".join(
            f"{o['name']} ({o['start']}, {o['multiplier']}x, order by {o['order_by']})" for o in others))
    return "\n".join(lines)


def _format_timing(state: AgentState) -> str:
    """
    Two separate facts, computed here so the model does not blend them:
      * stock today: how many of the listed SKUs are critical (independent of any event);
      * event deadline: what the alert level says about the ORDER-BY date.
    """
    signals = state.get("inventory_signals") or []
    known = [i for i in signals if i["current_stock"] != -1]
    critical = [i["sku_id"] for i in known if i["is_critical"]]
    if not known:
        stock = "stock today: insufficient data"
    elif critical:
        stock = (f"stock today: {len(critical)} of {len(known)} listed SKUs are CRITICAL right now "
                 f"({', '.join(critical)}); this is a current stock problem, whatever the event")
    else:
        stock = f"stock today: none of the {len(known)} listed SKUs is critical"

    seasonality = state.get("seasonality_signal") or {}
    level = (seasonality.get("alert_level") or "none").lower()
    meaning = {
        "critical": "the order-by date has PASSED: the event order is late, expedite it",
        "act": "the order-by date is close or the event is running: order now",
        "watch": "the order-by date is still ahead: NOT urgent for the event yet, give the date",
        "none": "no event lifts these products: no event deadline",
    }.get(level, "unknown")
    event = f"event deadline ({seasonality.get('upcoming_event') or 'none'}): alert {level.upper()}: {meaning}"
    return f"  • {stock}\n  • {event}"


def _select_language_label(query_language: str) -> str:
    if query_language == "ar":
        return "Arabic"
    if query_language == "mixed":
        return "Arabic (keep SKU codes and numbers in English)"
    return "English"


def _is_event_selection(state: AgentState) -> bool:
    return any(c.get("event_name") for c in state.get("retrieved_chunks", []))


def _compute_confidence(state: AgentState) -> float:
    """
    Data-coverage score for the recommendation (NOT a calibrated probability; rework planned in Phase 3).

    Retrieval path:  retrieval similarity 40%, sales data 30%, inventory data 20%, no error 10%.
    Event-planning path: SKUs were chosen by the calendar lift table, not by similarity, so there is no
    retrieval score to credit. That 40% becomes at most 20% (lift-table selection, synthetic) and the
    score can never reach 1.0; a supplier lead time that had to be assumed costs a further 10%.
    """
    has_sales     = 0.3 if state.get("sales_signals")     else 0.0
    has_inventory = 0.2 if state.get("inventory_signals") else 0.0
    no_error      = 0.1 if not state.get("error")         else 0.0
    if _is_event_selection(state):
        selection = 0.2 if state.get("retrieved_chunks") else 0.0
        assumed = (state.get("seasonality_signal") or {}).get("lead_time_assumed")
        penalty = 0.1 if assumed else 0.0
        return round(max(selection + has_sales + has_inventory + no_error - penalty, 0.0), 2)
    retrieval = state.get("retrieval_confidence", 0.0) * 0.4
    return round(min(retrieval + has_sales + has_inventory + no_error, 1.0), 2)


def _degraded_answer(state: AgentState, headline: str) -> str:
    """Trace-only answer used when the LLM cannot produce one. No provider error text."""
    trace_summary = " → ".join(state.get("reasoning_trace", []))
    return (
        f"[{headline} Agent reasoning trace:]\n\n{trace_summary}\n\n"
        f"Action: Review the trace above and consult inventory data directly."
    )


# ── Node implementation ────────────────────────────────────────────────────────

async def recommendation_node(state: AgentState) -> dict:
    """
    Node 5: Generate final recommendation via Groq LLM.

    Args:
        state: Full AgentState with all upstream node outputs

    Returns:
        Partial state dict with recommendation, confidence, and trace entry.
        On a degraded answer it also sets `error`, so callers can skip caching it.
    """
    confidence = _compute_confidence(state)
    language   = _select_language_label(state.get("query_language", "en"))
    model      = _model_name()
    error: str | None = None

    prompt = RECOMMENDATION_PROMPT.format(
        query=state["query"],
        products_text=_format_products(state.get("retrieved_chunks", [])),
        sales_text=_format_sales(state.get("sales_signals", [])),
        inventory_text=_format_inventory(state.get("inventory_signals", [])),
        seasonality_text=_format_seasonality(state.get("seasonality_signal", {})),
        timing_text=_format_timing(state),
        confidence=state.get("retrieval_confidence", 0.0),
        language=language,
    )

    try:
        client   = get_groq_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=_max_tokens(),
            temperature=0.15,
        )
        recommendation = (response.choices[0].message.content or "").strip()
        if not recommendation:
            raise ValueError("empty model response (token budget used up before an answer?)")

        logger.info(
            "recommendation_generated",
            model=model,
            confidence=confidence,
            language=language,
            tokens_used=response.usage.total_tokens if response.usage else "unknown",
        )
        trace = (
            f"Recommendation generated via Groq {model} "
            f"| language={language} | confidence={confidence:.2f}"
        )

    except RateLimitError:
        logger.warning("groq_rate_limit_hit", query=state["query"][:50])
        recommendation = _degraded_answer(
            state, "Rate limit reached — LLM synthesis unavailable."
        )
        confidence = max(confidence - 0.2, 0.0)
        trace      = "Recommendation: Groq rate limit hit — trace-only fallback returned"
        error      = "recommendation: rate limited"

    except Exception as exc:
        # Full detail goes to the server log only. Callers get a generic reason.
        logger.error("recommendation_node_error", model=model, error=str(exc))
        recommendation = _degraded_answer(
            state, "LLM synthesis unavailable — the recommendation service could not respond."
        )
        confidence = 0.0
        trace      = f"Recommendation: LLM unavailable ({type(exc).__name__}) — trace-only fallback returned"
        error      = f"recommendation: {type(exc).__name__}"

    update = {
        "recommendation":            recommendation,
        "recommendation_confidence": confidence,
        "reasoning_trace":           state["reasoning_trace"] + [trace],
    }
    if error:
        update["error"] = error
    return update