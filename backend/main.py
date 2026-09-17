"""
Dubai Retail Intelligence Agent — FastAPI Backend
==================================================
Phase 1 entrypoint: exposes the bilingual RAG retrieval endpoint.

Run:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /health              — liveness check
    POST /api/v1/search       — bilingual product search (Phase 1)
    POST /api/v1/agent/query  — full agentic loop (Phase 2, stubbed here)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .rag.retriever import RetailRetriever

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------

DATA_DIR        = os.getenv("DATA_DIR", "data/synthetic")
CATALOGUE_PATH  = os.path.join(DATA_DIR, "product_catalogue.csv")
SALES_PATH      = os.path.join(DATA_DIR, "sales_history.csv")
CHROMA_DIR      = os.getenv("CHROMA_DIR", ".chroma_db")
FORCE_REINGEST  = os.getenv("FORCE_REINGEST", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Global retriever instance (shared across requests)
# ---------------------------------------------------------------------------

retriever: Optional[RetailRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: initialize retriever once at startup.
    
    ENGINEERING DECISION: Startup initialization over lazy init per-request
    Loading a 300MB embedding model per request would destroy latency.
    We load once, keep in memory, serve all requests from the same model.
    Redis caching (Phase 2) sits in front of this for hot queries.
    """
    global retriever
    
    logger.info("startup_initializing", catalogue=CATALOGUE_PATH)
    
    retriever = RetailRetriever(
        catalogue_path=CATALOGUE_PATH,
        sales_path=SALES_PATH,
        persist_directory=CHROMA_DIR,
    )
    
    init_result = retriever.initialize(force_reingest=FORCE_REINGEST)
    logger.info("startup_complete", **init_result)
    
    yield  # Application runs here
    
    logger.info("shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dubai Retail Intelligence Agent",
    description=(
        "Bilingual Arabic/English agentic RAG system for Dubai mid-market retail. "
        "Phase 1: Confidence-gated product catalogue retrieval."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:7860"],  # Next.js + Gradio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas (Pydantic v2)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """
    Product search request. Supports Arabic and English queries.
    All fields validated before hitting the retrieval layer.
    """
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Product search query in Arabic or English",
        examples=["show me summer dresses under AED 200",
                  "أظهر لي الفساتين الصيفية تحت 200 درهم"],
    )
    n_results: int = Field(default=5, ge=1, le=20)
    category:   Optional[str]  = Field(default=None, description="Filter by category: fashion, electronics, home, food, beauty")
    max_price_aed: Optional[float] = Field(default=None, ge=0)
    min_price_aed: Optional[float] = Field(default=None, ge=0)
    ramadan_hero_only: bool = Field(default=False)
    dsf_hero_only:     bool = Field(default=False)
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
    query:             str
    query_language:    str
    result_count:      int
    confidence_gate:   float
    results:           list[ProductResult]
    filters_applied:   dict[str, Any]
    latency_ms:        float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness check. Returns 503 if retriever not yet ready."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return {
        "status":  "healthy",
        "version": "0.1.0",
        "retriever": "ready",
    }


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_products(request: SearchRequest):
    """
    Bilingual product search endpoint.
    
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

    return SearchResponse(
        **result,
        latency_ms=latency_ms,
    )


@app.post("/api/v1/agent/query", tags=["Agent"])
async def agent_query(body: dict):
    """
    Full agentic reasoning endpoint — stubbed for Phase 1.
    LangGraph agent loop will be wired here in Phase 2.
    
    Returns a clear "not yet implemented" response rather than 404,
    so frontend development can start against a real endpoint shape.
    """
    return {
        "status":  "phase_2_pending",
        "message": "LangGraph agentic loop not yet wired. Available in Phase 2.",
        "query":   body.get("query", ""),
        "estimated_availability": "Week 3–4",
    }
