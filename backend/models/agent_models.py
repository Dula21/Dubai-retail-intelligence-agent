"""
models/agent_models.py
-----------------------
Pydantic request/response models for the /api/v1/agent/query endpoint.

Engineering decision: Pydantic at the API boundary, TypedDict inside the graph.
Pydantic validation (type coercion, field validation, JSON schema generation)
is worth the cost at the API boundary where untrusted input arrives.
Inside the LangGraph state machine, TypedDict avoids that cost on every
node transition. Two different tools for two different jobs.

These models also serve as the OpenAPI schema — FastAPI auto-generates
/docs from them. For a portfolio project targeting noon and G42, a clean
API schema with field-level descriptions is a signal of production thinking.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request ────────────────────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    """
    Incoming query request for the retail intelligence agent.

    Supports Arabic, English, and mixed queries. Language detection
    is handled server-side via Unicode block analysis — the caller
    does not need to specify the language.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Product or merchandising query in Arabic or English",
        examples=[
            "What should I reorder before DSF?",
            "ما هي المنتجات التي يجب أن أعيد طلبها قبل دي إس إف؟",
            "Why is SKU-4421 underperforming compared to similar products?",
        ],
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


# ── Sub-response models ────────────────────────────────────────────────────────

class RetrievedProductResponse(BaseModel):
    """Simplified product info returned in the API response."""
    sku_id: str
    product_name_en: str
    product_name_ar: str
    category: str
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Vector-similarity score. null when the product was chosen by the calendar "
                    "(selection_basis = calendar_lift): there is no similarity to report",
    )
    selection_basis: str = Field(
        "similarity",
        description="How this product was chosen: similarity (vector search) | calendar_lift "
                    "(event-planning route, from the synthetic calendar lift table)",
    )
    event_lift: Optional[float] = Field(
        None, description="Expected demand multiplier from the calendar lift table (calendar_lift only)",
    )


class SeasonalityResponse(BaseModel):
    """UAE retail seasonality signal for the response."""
    upcoming_event: Optional[str] = None
    days_until_event: Optional[int] = None
    demand_multiplier: float = 1.0
    alert_level: str = "none"


class OrderQuantityResponse(BaseModel):
    """One deterministic order quantity from the quantity engine (never produced by the LLM)."""
    sku_id: str
    order_qty: Optional[int] = Field(None, description="Units to order; null = insufficient data (see flags)")
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    forecast_units: Optional[float] = Field(None, description="Forecast demand over the cover window")
    safety_multiplier: Optional[float] = None
    target_units: Optional[float] = Field(None, description="forecast x safety multiplier")
    on_hand_snapshot: Optional[int] = None
    snapshot_date: Optional[str] = None
    snapshot_age_days: Optional[int] = None
    projected_on_hand: Optional[float] = Field(None, description="Stock projected to the window start, no replenishment assumed")
    open_orders: int = 0
    coverage_estimate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Share of past held-out backtest windows this kind of target covered. NOT a probability for this "
                    "SKU. null = outside what the backtest validated")
    coverage_basis: str = ""
    indicative_only: bool = Field(
        False, description="True = planning estimate for a window beyond what the backtest validated: do not order "
                           "this now; it is recomputed on order_by_date")
    order_by_date: Optional[str] = Field(None, description="Date the order should be placed (event route only)")
    flags: list[str] = Field(default_factory=list)


# ── Main response ──────────────────────────────────────────────────────────────

class AgentQueryResponse(BaseModel):
    """
    Full agent response including recommendation and reasoning trace.

    The reasoning_trace field is a first-class output, not a debug artifact.
    It is what makes the system explainable to a retailer asking
    "why did you tell me to reorder 200 units?"

    confidence: data-coverage score (0.0–1.0), not a calibrated probability. With order quantities it is also
    capped by their measured coverage, and at 0.4 when a quantity lies outside the validated horizon. Below 0.5 should be
    treated as indicative rather than definitive by the caller. On the event-planning route it
    never reaches 1.0 (SKUs are chosen from a synthetic lift table, not retrieved by similarity).
    """
    # Core outputs
    recommendation: Optional[str] = Field(
        None,
        description="Actionable recommendation in the query language",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite confidence score for this recommendation",
    )

    # Explainability
    reasoning_trace: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning trace from each agent node",
    )

    # Retrieval results
    retrieved_products: list[RetrievedProductResponse] = Field(
        default_factory=list,
        description="Top product catalogue matches for this query",
    )

    # Phase 3: deterministic quantities (the recommendation text only explains these)
    order_recommendations: list[OrderQuantityResponse] = Field(
        default_factory=list,
        description="Order quantity per candidate SKU, computed by the quantity engine with every input and caveat",
    )

    # Seasonality context
    seasonality: SeasonalityResponse = Field(
        default_factory=SeasonalityResponse,
        description="Upcoming UAE retail event context",
    )

    # State flags
    fallback_triggered: bool = Field(
        False,
        description="True if retrieval confidence was below threshold",
    )
    query_language: str = Field(
        "en",
        description="Detected query language: en | ar | mixed",
    )
    intent: str = Field(
        "search",
        description="Classified query intent: reorder | analysis | search | trend | event_planning",
    )


# ── Fallback response ──────────────────────────────────────────────────────────

class FallbackResponse(BaseModel):
    """
    Returned when retrieval confidence is below threshold.

    Honest fallback: tells the caller exactly why no recommendation
    was generated. Consistent with architecture principle 2.
    """
    message: str
    fallback_triggered: bool = True
    retrieval_confidence: float
    retrieval_margin: float
    reasoning_trace: list[str]
    suggestion: str = (
        "Try rephrasing your query with a specific product name, "
        "SKU number, or category (e.g. 'footwear', 'electronics')."
    )



# ── Autonomous reorder scan (Phase 3) ──────────────────────────────────────────

class ScanRequest(BaseModel):
    """Run the autonomous reorder scan now. Both fields are optional."""
    as_of: Optional[date] = Field(
        default=None,
        description="Scan as of this date (default today). A past date is a replay: the data ends at the last "
                    "sales date, so use that date to see what the agent would have raised then.",
    )
    limit: Optional[int] = Field(
        default=None, ge=1, le=500,
        description="Most triggered SKUs to size with the quantity engine in this scan (default 40, most urgent first).",
    )


class DecisionRequest(BaseModel):
    """A human decision on one recommendation. The system's own recommendation is kept as it was."""
    decision: Literal["approve", "reject", "modify"]
    actor: str = Field(min_length=1, max_length=80, description="Who is deciding (name or user id)")
    reason: Optional[str] = Field(default=None, max_length=1000, description="Required for reject and modify")
    quantity: Optional[int] = Field(default=None, ge=0, le=1_000_000, description="Required for modify: the quantity to order")
