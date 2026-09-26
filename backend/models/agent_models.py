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

from typing import Optional

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
    relevance_score: float = Field(ge=0.0, le=1.0)


class SeasonalityResponse(BaseModel):
    """UAE retail seasonality signal for the response."""
    upcoming_event: Optional[str] = None
    days_until_event: Optional[int] = None
    demand_multiplier: float = 1.0
    alert_level: str = "none"


# ── Main response ──────────────────────────────────────────────────────────────

class AgentQueryResponse(BaseModel):
    """
    Full agent response including recommendation and reasoning trace.

    The reasoning_trace field is a first-class output, not a debug artifact.
    It is what makes the system explainable to a retailer asking
    "why did you tell me to reorder 200 units?"

    confidence: composite score (0.0–1.0). Below 0.5 should be treated
    as indicative rather than definitive by the caller.
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
        description="Classified query intent: reorder | analysis | search | trend",
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
