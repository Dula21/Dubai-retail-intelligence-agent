"""
agent_router.py
----------------
FastAPI router for the /api/v1/agent/ endpoints.

Included in main.py with:
    from .agent_router import router as agent_router
    app.include_router(agent_router, prefix="/api/v1/agent")

Endpoints:
    POST /api/v1/agent/query  — Main agent query
    GET  /api/v1/agent/health — Agent graph health check

Redis caching:
    Query results cached by (query_hash, date) with 1-hour TTL.
    Date component ensures yesterday's reorder recommendations
    don't serve for today's query (inventory changes daily).
    Responses produced after a technical error are NOT cached, so a one-off
    failure (e.g. the vector store being briefly unavailable) is not served
    from Redis for the whole TTL.

Fallback behaviour (architecture principle 10):
    The graph's fallback_node always fills `recommendation`, so a fallback is a
    normal 200 response with fallback_triggered=true:
      - factual intents (reorder, analysis): "Insufficient data" and NO candidate
        products in retrieved_products (a client must not show them as answers)
      - narrative intents (search, trend): partial answer with candidates and a
        LOW confidence label
    The 422 branch below is a safety net for the case where no recommendation
    was produced at all.

FIX vs earlier version: Redis client now comes from app.state.redis
which is set in main.py lifespan. Previous version read from
request.app.state.redis but main.py never set it — Redis was
silently None on every request.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .agents.retail_agent import run_agent
from .agents.state import SeasonalitySignal
from .models.agent_models import (
    AgentQueryRequest,
    AgentQueryResponse,
    FallbackResponse,
    RetrievedProductResponse,
    SeasonalityResponse,
)
from .services.intent_classifier import classify_intent, detect_language, is_factual_intent

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["agent"])

CACHE_TTL_SECONDS = int(os.getenv("AGENT_CACHE_TTL", "3600"))


def _cache_key(query: str) -> str:
    """
    Cache key: hash of (normalised query, today's date).
    Date component ensures stale inventory data doesn't persist overnight.
    """
    normalised = query.lower().strip()
    today = date.today().isoformat()
    raw = f"{normalised}:{today}"
    return f"agent:query:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


async def _get_redis(request: Request):
    """
    Get Redis client from app.state (set in main.py lifespan).
    Returns None if Redis unavailable — all callers handle None gracefully.
    """
    return getattr(request.app.state, "redis", None)


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(
    body: AgentQueryRequest,
    request: Request,
) -> AgentQueryResponse | JSONResponse:
    """
    Main agent endpoint.

    Flow:
    1. Language detection + intent classification (rule-based, instant)
    2. Redis cache check (1-hour TTL, date-keyed)
    3. LangGraph agent invocation (6 nodes, conditional routing)
    4. Response formatting + cache write (skipped after a technical error)
    5. Fallback answer (from fallback_node) if retrieval was not confident enough
    """
    query = body.query
    query_language = detect_language(query)
    intent = classify_intent(query)

    logger.info(
        "agent_query_received",
        query=query[:60],
        language=query_language,
        intent=intent,
    )

    # ── Cache check ────────────────────────────────────────────────────────
    redis = await _get_redis(request)
    cache_key = _cache_key(query)

    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                logger.info("agent_cache_hit", key=cache_key[:20])
                return JSONResponse(content=json.loads(cached))
        except Exception as e:
            logger.warning("redis_read_failed", error=str(e))

    # ── Agent invocation ───────────────────────────────────────────────────
    try:
        state = await run_agent(
            query=query,
            query_language=query_language,
            intent=intent,
        )
    except Exception as exc:
        logger.error("agent_invocation_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Agent invocation failed: {exc}",
        )

    # ── Safety net: fallback with no answer text at all ────────────────────
    # fallback_node always sets `recommendation`, so this only fires if the
    # graph ended on a fallback without producing one.
    if state.get("fallback_triggered") and not state.get("recommendation"):
        fallback = FallbackResponse(
            message=(
                "Retrieval confidence below threshold — "
                "unable to generate a reliable recommendation for this query."
            ),
            fallback_triggered=True,
            retrieval_confidence=state.get("retrieval_confidence", 0.0),
            retrieval_margin=state.get("retrieval_margin", 0.0),
            reasoning_trace=state.get("reasoning_trace", []),
        )
        logger.info("fallback_response_returned", query=query[:60])
        return JSONResponse(status_code=422, content=fallback.model_dump())

    # ── Build response ─────────────────────────────────────────────────────
    # Hard fallback (factual intent, uncertain retrieval): do not expose the
    # candidate products either. The text says "insufficient data"; the JSON
    # must not hand the client a list that looks like an answer.
    hard_fallback = bool(state.get("fallback_triggered")) and is_factual_intent(intent)

    retrieved_products = [] if hard_fallback else [
        RetrievedProductResponse(
            sku_id=c["sku_id"],
            product_name_en=c["product_name_en"],
            product_name_ar=c["product_name_ar"],
            category=c["category"],
            relevance_score=c["similarity_score"],
        )
        for c in state.get("retrieved_chunks", [])
    ]

    raw_season: SeasonalitySignal = state.get("seasonality_signal") or {}
    seasonality = SeasonalityResponse(
        upcoming_event=raw_season.get("upcoming_event"),
        days_until_event=raw_season.get("days_until_event"),
        demand_multiplier=raw_season.get("expected_demand_multiplier", 1.0),
        alert_level=raw_season.get("alert_level", "none"),
    )

    response = AgentQueryResponse(
        recommendation=state.get("recommendation"),
        confidence=state.get("recommendation_confidence", 0.0),
        reasoning_trace=state.get("reasoning_trace", []),
        retrieved_products=retrieved_products,
        seasonality=seasonality,
        fallback_triggered=state.get("fallback_triggered", False),
        query_language=query_language,
        intent=intent,
    )

    # ── Cache write ────────────────────────────────────────────────────────
    # Never cache an answer produced after a technical error: it would be
    # served for the whole TTL even though the failure may have been momentary.
    if redis and not state.get("error"):
        try:
            await redis.setex(
                cache_key,
                CACHE_TTL_SECONDS,
                response.model_dump_json(),
            )
            logger.info("agent_response_cached", ttl=CACHE_TTL_SECONDS)
        except Exception as e:
            logger.warning("redis_write_failed", error=str(e))

    logger.info(
        "agent_query_complete",
        intent=intent,
        confidence=response.confidence,
        fallback=response.fallback_triggered,
        products_returned=len(retrieved_products),
    )

    return response


@router.get("/health")
async def agent_health() -> dict:
    """Agent health check — confirms LangGraph graph is compiled."""
    from .agents.retail_agent import retail_agent

    return {
        "status":         "healthy",
        "graph_compiled": retail_agent is not None,
        "nodes": [
            "rag_retrieval_node",
            "sales_analysis_node",
            "inventory_check_node",
            "seasonality_check_node",
            "recommendation_node",
            "fallback_node",
        ],
        "phase": 2,
    }