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
reported as a separate, current problem. Order quantities come ONLY from the deterministic quantity
engine (ORDER QUANTITIES section): this node explains them and never computes one.

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
import re

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


def _reasoning_effort() -> str:
    value = os.environ.get("GROQ_REASONING_EFFORT", "low").lower()
    return value if value in ("low", "medium", "high") else "low"


# ── Prompt templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an autonomous retail merchandising AI for Dubai mid-market retailers.
Your job is to generate clear, actionable reorder and stock recommendations.

Rules you must follow without exception:
1. Only use numbers and facts present in the DATA section below — never invent figures
2. If data for a field is missing or marked -1, state "insufficient data" for that field
3. Cite which data source supports each claim (e.g. "Based on 30-day sales average")
4. Keep the recommendation under 200 words — retailers are busy
5. Write ONLY what the data shows. The system writes the recommendation and the action label (rule 17)
6. If responding in Arabic, keep SKU codes, AED values, and numbers in their original form
7. Dates in the answer must come from the DATA section. If the order-by date has PASSED, say so
   plainly and recommend expediting; never describe a passed deadline as upcoming
8. Never invent rules, formulas, safety-stock percentages or order quantities. If ORDER QUANTITIES has no
   quantity for a SKU, say the quantity cannot be computed and give the reason shown there
9. If DATA says the supplier lead time is assumed, or that event dates are an estimate, say so
10. A reorder point is a trigger level, NOT an order quantity. Never derive a quantity from a reorder
    point, a sales rate or the demand multiplier. The only quantities you may state are those listed
    under ORDER QUANTITIES, copied exactly. Never add, round, split or adjust them
11. Keep two questions apart, using the TIMING section: (a) is stock critical TODAY, whatever the
    event, and (b) is the EVENT order deadline urgent. If the alert level is WATCH, say the event
    deadline is not yet urgent and give the date; a critical stock level is then a separate,
    current problem. Never describe a WATCH deadline as urgent, and never describe stock that
    is not critical as a problem
12. Two different numbers must never be mixed up. The EXPECTED demand multiplier comes from the
    calendar lift table (UAE SEASONALITY). The OBSERVED lift comes from past sales (SALES SIGNALS).
    Never call the expected multiplier a past, observed or historical lift, and never cite
    SALES SIGNALS for the calendar multiplier
13. For every quantity, state its caveats in plain words, using the sentences given in ORDER QUANTITIES.
    Never print raw caveat codes (words with underscores). State the coverage estimate exactly as written;
    if it says "none", say it was not validated. Never call it a probability of being right: it is the
    share of past held-out windows the target covered
14. A quantity marked INDICATIVE ONLY is a planning estimate, never an order to place now: say it will be
    recomputed on the order-by date and give that date. Never use it to justify the REORDER NOW label.
    Order quantities size the EVENT window; never claim an order will fix today's stock level, bring
    stock above a reorder point, or cover a lift, beyond what ORDER QUANTITIES itself states
15. The order-by date is the LATEST day to place the order. Write "order by <date>" or "on or before <date>";
    never "on or after", "from" or "starting" that date. After it, the order arrives too late for the event
16. Always state the caveats that change how far the numbers can be trusted, in the first paragraph of the
    answer: (a) if a stock figure is N days old, say so wherever you quote stock, and never call stale stock
    "current" or "now"; (b) if a window is beyond what was validated or has no event, say it was not validated.
    The system appends the full list of order quantities after your answer, so do not write that list yourself;
    refer to SKUs by id, explain, and never state a quantity that is not in ORDER QUANTITIES
17. Do NOT write a recommendation, an instruction to order, a quantity, or an action label (REORDER NOW /
    MONITOR / ...): the system writes those from its own rules, after your text. Describe the situation only.
    The days to the order-by date and the days to the event are different numbers: use each only for what it
    is named
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

ORDER QUANTITIES (computed by the deterministic engine; state these exactly, never compute your own):
{quantity_text}

RETRIEVAL CONFIDENCE: {confidence:.2f} (threshold: 0.72)

Based on the data above, generate a retail merchandising recommendation in {language}.
Write what the data shows in 2-4 sentences, citing sources, and stop. Do not write a recommendation,
an instruction to order, any quantity, or an action label: the system adds the recommendation after your text.
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
        status = f"{days_left} days left to place the order"
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


_FLAG_SENTENCES = {
    "stock_snapshot_stale": "the stock figure is {age} days old",
    "stock_projected_without_replenishment": "stock at the window start is projected assuming no deliveries arrive",
    "open_orders_unknown": "open purchase orders are unknown and assumed to be zero",
    "high_masked_share": "many days of this SKU's history were stock-outs, so the forecast learned less",
    "beyond_validated_horizon": "the window starts {horizon} days after the last data, farther than the backtest validated",
    "lead_time_assumed": "the supplier lead time is assumed",
    "non_event_window": "this window has no event, and the backtest only validated event windows",
}


def _caveat_sentences(r: dict) -> list[str]:
    """Plain-language caveats, so the model never has to translate (or print) raw flag codes."""
    out = []
    for flag in r.get("flags") or []:
        if flag.startswith("events_without_history"):
            out.append("an overlapping event has no past occurrence to learn from (measured coverage is lower)")
        elif flag in _FLAG_SENTENCES:
            out.append(_FLAG_SENTENCES[flag].format(age=r.get("snapshot_age_days"), horizon=r.get("horizon_days")))
    return out


def _format_quantities(results: list) -> str:
    """The engine's numbers, one line per SKU, with every input and caveat the model must state."""
    if not results:
        return "No quantity computed (quantity engine produced no result)"
    lines = []
    for r in results:
        if r.get("order_qty") is None:
            reason = "; ".join(_caveat_sentences(r)) or ", ".join(r.get("flags") or ["unknown"]).replace("_", " ")
            lines.append(f"  • {r['sku_id']}: NO QUANTITY ({reason})")
            continue
        cov = (f"coverage estimate {r['coverage_estimate']:.0%} of past held-out windows ({r['coverage_basis']})"
               if r.get("coverage_estimate") is not None else f"coverage estimate: none ({r.get('coverage_basis', '')})")
        if r.get("indicative_only"):
            head = (f"INDICATIVE ONLY, about {r['order_qty']} units (planning estimate, NOT an order to place now; it is "
                    f"recomputed on the order-by date {r.get('order_by_date') or 'when the window comes within the validated horizon'})")
        else:
            head = f"ORDER {r['order_qty']} units"
        lines.append(
            f"  • {r['sku_id']}: {head} | window {r['window_start']} to {r['window_end']} "
            f"| forecast {r['forecast_units']} units x safety {r['safety_multiplier']} = target {r['target_units']} "
            f"| stock {r['on_hand_snapshot']} on {r['snapshot_date']}, projected {r['projected_on_hand']} at window start "
            f"| open orders {r['open_orders']} | {cov}"
            + (f" | caveats in plain words: {'; '.join(_caveat_sentences(r))}" if _caveat_sentences(r) else ""))
    return "\n".join(lines)


# A quantity with no measured coverage (outside the validated horizon) must not sit next to a high confidence.
MAX_QUANTITY_LINES = 12
UNVALIDATED_CONFIDENCE_CAP = 0.4        # below the 0.5 line the API documents as "indicative"


def _apply_quantity_confidence(confidence: float, results: list) -> float:
    """
    Cap the data-coverage score by what the backtest measured for the quantities in this answer:
    the lowest measured coverage among them, or UNVALIDATED_CONFIDENCE_CAP when any quantity has none.
    """
    made = [r for r in results if r.get("order_qty") is not None]
    if not made:
        return confidence
    if any(r.get("coverage_estimate") is None for r in made):
        return round(min(confidence, UNVALIDATED_CONFIDENCE_CAP), 2)
    return round(min(confidence, min(r["coverage_estimate"] for r in made)), 2)


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
    Data-coverage score for the recommendation (NOT a calibrated probability). Quantities then cap it
    (see _apply_quantity_confidence): the score never exceeds the measured coverage of the quantities given.

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


# ── Deterministic additions to the model's text ────────────────────────────────

# The model sometimes writes non-breaking hyphens and spaces (U+2011, U+202F), which break copy, search and date parsing.
_TEXT_FIXES = {"\u2010": "-", "\u2011": "-", "\u202f": " ", "\u00a0": " "}


def _normalize_text(text: str) -> str:
    """Replace look-alike hyphens and spaces with plain ones. En dashes are left alone: they are legitimate in ranges."""
    for bad, good in _TEXT_FIXES.items():
        text = text.replace(bad, good)
    return text


def _quantities_block(results: list, query_language: str = "en") -> str:
    """
    The engine's quantities, written by code, not by the model. The model explains; it cannot change these numbers,
    and it sometimes leaves them out. Same facts as the order_recommendations field of the API response.
    """
    if not results:
        return ""
    ar = query_language == "ar"
    units = "وحدة" if ar else "units"
    head = ("**كميات الطلب (من محرك التنبؤ، وليس من النموذج اللغوي)**" if ar else
            "**Order quantities (from the forecasting engine, not the language model)**")
    lines = []
    for r in results[:MAX_QUANTITY_LINES]:
        if r.get("order_qty") is None:
            reason = "; ".join(_caveat_sentences(r)) or "not enough data"
            lines.append(f"- {r['sku_id']}: no quantity ({reason})")
            continue
        qty = f"{r['order_qty']:,} {units}"
        if r.get("indicative_only"):
            when = f"indicative only; recomputed by {r['order_by_date']}" if r.get("order_by_date") else "indicative only"
        else:
            when = "order now" if not r.get("order_by_date") else f"order by {r['order_by_date']}"
        cov = (f"coverage {r['coverage_estimate']:.0%}" if r.get("coverage_estimate") is not None else "coverage not validated")
        lines.append(f"- {r['sku_id']}: **{qty}** ({when}; window {r['window_start']} to {r['window_end']}; {cov})")
    if len(results) > MAX_QUANTITY_LINES:
        lines.append(f"- … and {len(results) - MAX_QUANTITY_LINES} more in the order_recommendations field")
    return head + "\n" + "\n".join(lines)


ACTION_LABELS = ("REORDER NOW", "MONITOR", "INVESTIGATE", "NO ACTION NEEDED")
_LABEL_RE = "|".join(re.escape(x) for x in ACTION_LABELS)
# A trailing label the model wrote itself: optional heading line ("**Action label**"), then the label alone.
_TRAILING_LABEL = re.compile(
    rf"(?:\n+[ \t]*(?:#+\s*|\*\*)?(?:action(?: label)?|recommendation label)[ \t]*:?(?:\*\*)?[ \t]*)?"
    rf"\n+[ \t]*(?:\*\*)?(?:action(?: label)?:\s*)?(?:{_LABEL_RE})(?:\*\*)?[ \t.]*\s*$", re.IGNORECASE)


def action_label(state: AgentState) -> str:
    """
    The action label is DECIDED HERE from the engine's own signals, not by the language model (same reason as the
    quantities: the model contradicted the engine, saying REORDER NOW for an order-by date 291 days away).
      event order-by passed / close / event running  -> REORDER NOW
      stock critical today and the stock figure is fresh -> REORDER NOW
      stock critical today but the figure is stale   -> INVESTIGATE (an old snapshot cannot justify an order)
      event order-by still ahead (WATCH)             -> MONITOR
      nothing lifts these products and stock is fine -> NO ACTION NEEDED
      no usable data                                 -> INVESTIGATE
    """
    level = ((state.get("seasonality_signal") or {}).get("alert_level") or "none").lower()
    signals = [i for i in (state.get("inventory_signals") or []) if i.get("current_stock") != -1]
    critical = any(i.get("is_critical") for i in signals)
    stale = any("stock_snapshot_stale" in (r.get("flags") or []) for r in (state.get("quantity_results") or []))
    if level in ("critical", "act"):
        return "REORDER NOW"
    if critical:
        return "INVESTIGATE" if stale else "REORDER NOW"
    if level == "watch":
        return "MONITOR"
    if not signals and level == "none":
        return "INVESTIGATE"
    return "NO ACTION NEEDED"


def _strip_model_label(text: str) -> str:
    return _TRAILING_LABEL.sub("", text.rstrip()).rstrip()


# A section the model wrote anyway ("**Recommendation** ..."): cut from its heading to the end.
_MODEL_RECOMMENDATION = re.compile(
    r"\n[ \t]*(?:#+[ \t]*)?\*{0,2}(?:the[ \t]+)?(?:recommendation|action)\*{0,2}[ \t]*:?\*{0,2}[ \t]*(?:\n|$)",
    re.IGNORECASE)


def _cut_model_recommendation(text: str) -> str:
    m = _MODEL_RECOMMENDATION.search("\n" + text)
    return (("\n" + text)[:m.start()]).strip() if m else text


def _event_clause(sig: dict, ar: bool) -> str:
    """The event timing, from the signals. Order-by days and event days are stated separately, on purpose."""
    event, order_by = sig.get("upcoming_event"), sig.get("order_by_date")
    if not event:
        return ""
    until, left = sig.get("days_until_event"), sig.get("days_to_order_by")
    if until is not None and until <= 0:
        return (f"{event} جارٍ الآن." if ar else f"{event} is running now.")
    start = (f"يبدأ {event} بعد {until} يومًا" if ar else f"{event} starts in {until} days") if until is not None else event
    if not order_by:
        return start + "."
    if left is None:
        by = f"آخر موعد للطلب {order_by}" if ar else f"order by {order_by}"
    elif left < 0:
        by = (f"فات آخر موعد للطلب ({order_by}) منذ {-left} يومًا" if ar
              else f"the order-by date ({order_by}) passed {-left} days ago")
    elif left == 0:
        by = f"آخر موعد للطلب اليوم ({order_by})" if ar else f"the order-by date is today ({order_by})"
    else:
        by = (f"آخر موعد للطلب {order_by} (بعد {left} يومًا)" if ar
              else f"last day to place the order: {order_by} ({left} days from now)")
    return f"{start}؛ {by}." if ar else f"{start}; {by}."


def recommendation_sentence(state: AgentState, label: str, query_language: str = "en") -> str:
    """
    The instruction to the reader, written by code from the same signals as the label, so the label, the dates and
    the sentence cannot disagree. Every number and date here comes from the engine, never from the language model.
    """
    ar = query_language == "ar"
    sig = state.get("seasonality_signal") or {}
    known = [i for i in (state.get("inventory_signals") or []) if i.get("current_stock") != -1]
    critical = [i["sku_id"] for i in known if i.get("is_critical")]
    ages = [r.get("snapshot_age_days") for r in (state.get("quantity_results") or [])
            if "stock_snapshot_stale" in (r.get("flags") or []) and r.get("snapshot_age_days") is not None]
    age = max(ages) if ages else None
    event = _event_clause(sig, ar)
    ids = ", ".join(critical[:6]) + (f" +{len(critical) - 6}" if len(critical) > 6 else "")
    quant = (" الكميات أدناه إرشادية وتُحسب من جديد في موعد الطلب." if ar
             else " The quantities below are indicative and are recomputed on the order-by date.")
    level = (sig.get("alert_level") or "none").lower()

    if label == "REORDER NOW":
        if level in ("critical", "act"):
            head = (f"اطلب الآن: {event}" if ar else f"Place the order now: {event}")
        else:
            head = (f"المخزون حرج الآن في {len(critical)} من {len(known)} منتجًا ({ids}): اطلب تجديد المخزون الآن."
                    if ar else f"Stock is critical on {len(critical)} of {len(known)} listed SKUs ({ids}): "
                               f"place a replenishment order now.")
            if event:
                head += " " + event
        return head
    if label == "INVESTIGATE":
        if critical:
            old = f" عمره {age} يومًا" if ar and age else (f" is {age} days old" if age else " may be out of date")
            head = (f"يظهر المخزون حرجًا في {len(critical)} من {len(known)} منتجًا ({ids})، لكن الرقم{old}: "
                    f"تحقق من المخزون الحالي قبل أي طلب." if ar else
                    f"Stock reads critical on {len(critical)} of {len(known)} listed SKUs ({ids}), but the figure{old}: "
                    f"confirm current stock before ordering.")
            return head + (" " + event if event else "") + (quant if event else "")
        return ("لا توجد بيانات كافية للتوصية بإجراء: راجع بيانات المخزون لهذه المنتجات." if ar else
                "There is not enough data to recommend an action: check the inventory data for these products.")
    if label == "MONITOR":
        return ((f"لا حاجة لطلب الآن. {event}" if ar else f"No order is needed yet. {event}") + quant)
    return ("لا إجراء مطلوب: المخزون غير حرج ولا توجد مناسبة ترفع الطلب على هذه المنتجات." if ar else
            "No action needed: stock is not critical and no event lifts these products.")


def _finalize_text(recommendation: str, results: list, query_language: str, label: str | None = None,
                   state: AgentState | None = None) -> str:
    """The model's explanation, then everything that moves money, written by code: recommendation, label, quantities."""
    text = _normalize_text(recommendation)
    arabic = query_language == "ar"
    if label:
        text = _strip_model_label(_cut_model_recommendation(text))
        if state is not None:
            text += (f"\n\n**{'التوصية' if arabic else 'Recommendation'}**\n"
                     f"{recommendation_sentence(state, label, query_language)}")
        text += f"\n\n**{'الإجراء' if arabic else 'Action'}:** {label}"
    block = _quantities_block(results, query_language)
    return f"{text}\n\n{block}" if block else text


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
    confidence = _apply_quantity_confidence(_compute_confidence(state), (state.get("quantity_results") or []))
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
        quantity_text=_format_quantities((state.get("quantity_results") or [])),
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
            extra_body={"reasoning_effort": _reasoning_effort()} if "gpt-oss" in model else None,
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

    recommendation = _finalize_text(recommendation, (state.get("quantity_results") or []),
                                    state.get("query_language", "en"), label=action_label(state), state=state)
    update = {
        "recommendation":            recommendation,
        "recommendation_confidence": confidence,
        "reasoning_trace":           state["reasoning_trace"] + [trace],
    }
    if error:
        update["error"] = error
    return update

