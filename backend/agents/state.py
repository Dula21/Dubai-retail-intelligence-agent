from __future__ import annotations

from typing import NotRequired, Optional, TypedDict


class RetrievedChunk(TypedDict):
    sku_id: str
    product_name_en: str
    product_name_ar: str
    category: str
    similarity_score: float
    content: str
    # Set by the event-planning route (event_candidates_node); absent on retrieval chunks.
    event_name: NotRequired[str]
    event_lift: NotRequired[float]
    supplier_lead_days: NotRequired[Optional[int]]


class SalesSignal(TypedDict):
    sku_id: str
    avg_daily_units: float
    trend_direction: str
    peak_period: Optional[str]
    confidence: float
    # Planning route only: the event asked about and the lift observed in its past windows.
    event_name: NotRequired[str]
    observed_event_lift: NotRequired[Optional[float]]


class InventorySignal(TypedDict):
    sku_id: str
    current_stock: int
    reorder_point: int
    runway_days: float
    is_critical: bool


class SeasonalitySignal(TypedDict):
    upcoming_event: Optional[str]
    days_until_event: Optional[int]
    expected_demand_multiplier: float
    alert_level: str
    event_start: NotRequired[Optional[str]]
    event_end: NotRequired[Optional[str]]
    order_by_date: NotRequired[Optional[str]]
    days_to_order_by: NotRequired[Optional[int]]
    lead_time_days: NotRequired[Optional[int]]
    lead_time_assumed: NotRequired[bool]
    date_basis: NotRequired[Optional[str]]
    named_in_query: NotRequired[bool]
    other_upcoming: NotRequired[list]


class AgentState(TypedDict):
    query: str
    query_language: str
    intent: str  # reorder | analysis | search | trend | event_planning
    retrieved_chunks: list[RetrievedChunk]
    retrieval_confidence: float
    retrieval_margin: float  # top-1 minus the next DISTINCT product (not top-3)
    sales_signals: list[SalesSignal]
    inventory_signals: list[InventorySignal]
    seasonality_signal: SeasonalitySignal
    reasoning_trace: list[str]
    recommendation: Optional[str]
    recommendation_confidence: float
    fallback_triggered: bool
    error: Optional[str]