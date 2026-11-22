"""
Dubai Retail Intelligence Agent — FastAPI Backend
==================================================
Phase 1: Bilingual RAG retrieval endpoint
Phase 2: LangGraph agentic loop (agent_router)

Run:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /health              — liveness check
    POST /api/v1/search       — bilingual product search (Phase 1)
    POST /api/v1/agent/query  — full agentic loop (Phase 2)
    GET  /api/v1/agent/health — agent graph health check (Phase 2)
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .rag.retriever import RetailRetriever
from .agents.nodes.rag_node import set_retriever

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------

DATA_DIR       = os.getenv("DATA_DIR", "data/synthetic")
CATALOGUE_PATH = os.path.join(DATA_DIR, "product_catalogue.csv")
SALES_PATH     = os.path.join(DATA_DIR, "sales_history.csv")
CHROMA_DIR     = os.getenv("CHROMA_DIR", ".chroma_db")
FORCE_REINGEST = os.getenv("FORCE_REINGEST", "false").lower() == "true"
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379")

REORDER_SCAN_INTERVAL_MINUTES   = int(os.getenv("REORDER_SCAN_INTERVAL_MINUTES", "360"))
REORDER_SCAN_FIRST_DELAY_SECONDS = int(os.getenv("REORDER_SCAN_FIRST_DELAY_SECONDS", "180"))


async def _scan_loop() -> None:
    """
    Run the reorder scan forever: first after a delay (so startup and the first requests are not competing with
    Prophet fits), then every REORDER_SCAN_INTERVAL_MINUTES. A failed scan is logged and the loop carries on.
    DECISION: an in-process asyncio loop, not a cron service. Free tier has no scheduler, and a container that is
    asleep (Hugging Face Spaces) scans on wake-up. Next architectural step at scale: a separate worker or scheduled job.
    """
    from .agents import reorder_scan
    await asyncio.sleep(REORDER_SCAN_FIRST_DELAY_SECONDS)
    while True:
        try:
            result = await reorder_scan.run_scan()
            logger.info("reorder_scan_scheduled_done", recommendations=len(result["recommendations"]),
                        triggered=result["triggered"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("reorder_scan_scheduled_failed", error=str(exc))
        await asyncio.sleep(REORDER_SCAN_INTERVAL_MINUTES * 60)


# ---------------------------------------------------------------------------
# Global retriever instance (shared across requests)
# ---------------------------------------------------------------------------

retriever: Optional[RetailRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: initialise retriever and Redis once at startup.

    ENGINEERING DECISION: Startup initialisation over lazy init per-request.
    Loading a 300MB embedding model per request would destroy latency.
    We load once, keep in memory, serve all requests from the same model.

    FIX (vs earlier version): Redis client is now stored on app.state.redis
    so agent_router._get_redis() can access it. Previous version never set
    this — Redis caching was silently disabled on every request.
    """
    global retriever

    # ── Redis ──────────────────────────────────────────────────────────────
    try:
        app.state.redis = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        await app.state.redis.ping()
        logger.info("redis_connected", url=REDIS_URL)
    except Exception as e:
        logger.warning("redis_unavailable", error=str(e))
        app.state.redis = None   # agent_router degrades gracefully without Redis

    # ── Retriever ──────────────────────────────────────────────────────────
    logger.info("startup_initializing", catalogue=CATALOGUE_PATH)

    retriever = RetailRetriever(
        catalogue_path=CATALOGUE_PATH,
        sales_path=SALES_PATH,
        persist_directory=CHROMA_DIR,
    )

    init_result = retriever.initialize(force_reingest=FORCE_REINGEST)

    # Phase 2: make the retriever available to the LangGraph RAG node.
    # set_retriever() stores it as a module-level singleton in rag_node.py.
    # Without this call, Node 1 (rag_retrieval_node) has nothing to call.
    set_retriever(retriever)
    logger.info("rag_node_retriever_set")

    # Phase 3: load the sales history and the quantity calibration once, at startup, so the first
    # reorder question does not pay for a 365k-row CSV read.
    try:
        from .agents.nodes.quantity_node import load_calibration
        from .agents.nodes.sales_node import _load_sales_df
        _load_sales_df()
        logger.info("quantity_calibration_loaded", present=load_calibration() is not None)
    except Exception as e:  # never block startup on this; the nodes degrade and say so
        logger.warning("phase3_preload_failed", error=str(e))

    logger.info("startup_complete", **init_result)

    # Phase 3: the autonomous reorder scan runs on a timer with no request. 0 disables it.
    scan_task = None
    if REORDER_SCAN_INTERVAL_MINUTES > 0:
        scan_task = asyncio.create_task(_scan_loop())
        logger.info("reorder_scan_loop_started", every_minutes=REORDER_SCAN_INTERVAL_MINUTES,
                    first_run_in_seconds=REORDER_SCAN_FIRST_DELAY_SECONDS)

    yield  # Application runs here

    # ── Shutdown ───────────────────────────────────────────────────────────
    if scan_task:
        scan_task.cancel()
    if app.state.redis:
        await app.state.redis.aclose()
    logger.info("shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dubai Retail Intelligence Agent",
    description=(
        "Bilingual Arabic/English agentic RAG system for Dubai mid-market retail. "
        "Phase 1: Confidence-gated product catalogue retrieval. "
        "Phase 2: LangGraph agentic reorder recommendation loop."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Phase 2: Agent router
# Registered BEFORE inline endpoints so /api/v1/agent/* routes resolve here.
# ---------------------------------------------------------------------------

from .agent_router import router as agent_router  # noqa: E402
app.include_router(agent_router, prefix="/api/v1/agent")


# ---------------------------------------------------------------------------
# Request / Response Schemas — Phase 1 search (unchanged)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Product search query in Arabic or English",
        examples=[
            "show me summer dresses under AED 200",
            "أظهر لي الفساتين الصيفية تحت 200 درهم",
        ],
    )
    n_results:         int            = Field(default=5, ge=1, le=20)
    category:          Optional[str]  = Field(default=None)
    max_price_aed:     Optional[float] = Field(default=None, ge=0)
    min_price_aed:     Optional[float] = Field(default=None, ge=0)
    ramadan_hero_only: bool           = Field(default=False)
    dsf_hero_only:     bool           = Field(default=False)
    min_confidence:    Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        valid = {"fashion", "electronics", "home", "food", "beauty", None}
        if v not in valid:
            raise ValueError(f"category must be one of {valid - {None}}")
        return v

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class ProductResult(BaseModel):
    sku_id:           str
    name_en:          str
    name_ar:          str
    category:         str
    price_aed:        float
    similarity_score: float
    query_language:   str
    sales:            Optional[dict[str, Any]] = None
    is_ramadan_hero:  Optional[int] = None
    is_dsf_hero:      Optional[int] = None


class SearchResponse(BaseModel):
    query:           str
    query_language:  str
    result_count:    int
    confidence_gate: float
    results:         list[ProductResult]
    filters_applied: dict[str, Any]
    latency_ms:      float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness check. Returns 503 if retriever not yet ready."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return {
        "status":    "healthy",
        "version":   "0.2.0",
        "phase":     2,
        "retriever": "ready",
        "redis":     "connected" if app.state.redis else "unavailable",
    }


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_products(request: SearchRequest):
    """
    Bilingual product search endpoint (Phase 1).

    - Accepts Arabic and English queries
    - Applies confidence gating (default 0.72)
    - Returns products enriched with sales context
    - Target latency: < 500ms for warm queries
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    start = time.perf_counter()

    try:
        result = retriever.search(
            query=request.query,
            n_results=request.n_results,
            category=request.category,
            max_price_aed=request.max_price_aed,
            min_price_aed=request.min_price_aed,
            ramadan_hero_only=request.ramadan_hero_only,
            dsf_hero_only=request.dsf_hero_only,
        )
    except Exception as e:
        logger.error("search_error", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    logger.info(
        "search_request",
        query_preview=request.query[:50],
        result_count=result["result_count"],
        latency_ms=latency_ms,
    )

    return SearchResponse(**result, latency_ms=latency_ms)


