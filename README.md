# Dubai Retail Intelligence Agent

> Bilingual Arabic/English agentic RAG system for Dubai mid-market retail.  
> Replaces a AED 15,000–25,000/month merchandising manager with an autonomous AI layer.

**Status:** Phase 2 complete — LangGraph agentic loop, 55 tests passing  
**Stack:** LangGraph · LangChain · ChromaDB · multilingual-e5-large · Prophet · Groq · FastAPI  
**Data:** Synthetic — 500 SKUs, 365,000 sales rows, UAE seasonality modeled

---

## The Problem

Dubai mid-market retailers selling on noon, their own website, and physical stores have no unified system connecting:

- What customers are searching for
- What is currently selling
- What inventory is running low
- What to stock for Ramadan, DSF, and National Day

Reorder decisions are made on gut feel. Demand surges during Dubai's major retail events are missed every year. Product catalogues exist in Arabic and English with inconsistent tagging and poor searchability.

This system solves that problem autonomously.

---

## Architecture

```
User Query (Arabic or English)
        ↓
Language Detection  [Unicode Arabic block ratio]
        ↓
Intent Classification  [rule-based: reorder / analysis / trend / search]
        ↓
Bilingual Embedding  [multilingual-e5-large]
        ↓
Confidence Gate  [floor: 0.72 cosine + margin-based routing]
        ↓
ChromaDB Vector Store  [metadata-filtered + cosine similarity]
        ↓
LangGraph Agent Loop  [conditional routing based on intent]
    ├── Node 1: RAG Retrieval      [bilingual, margin-gated, deduplicated]
    ├── Node 2: Sales Analysis     [30-day trend, UAE peak period detection]
    ├── Node 3: Inventory Check    [runway days, critical flag, hard fallback]
    ├── Node 4: Seasonality Check  [DSF / Ramadan / National Day / Eid]
    └── Node 5: Recommendation     [Groq LLM, grounded prompt, bilingual output]
        ↓
Explainable Reasoning Trace  [every node appends — returned to caller]
        ↓
Bilingual Response (Arabic / English)
        ↓
Audit Log  [PostgreSQL]
```

### Routing logic

Search and trend queries route directly to the recommendation node (2 nodes total).  
Reorder and analysis queries run the full 5-node path.  
Fallback triggers exit immediately at the RAG node — no LLM call made.

---

## Build Phases

| Phase | Scope | Status | Tests |
|-------|-------|--------|-------|
| 1 | Bilingual RAG over product catalogue | ✅ Complete | 25 |
| 2 | LangGraph agentic loop | ✅ Complete | 30 |
| 3 | Autonomous reorder recommendation engine | 📋 Planned | — |
| 4 | Arabic dialect polish + bilingual response generation | 📋 Planned | — |

**Total tests passing: 55**

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Dula21/dubai-retail-intelligence-agent
cd dubai-retail-intelligence-agent
pip install -r requirements.txt

# 2. Generate synthetic UAE retail dataset
python data/synthetic/generate_dataset.py
# → Creates product_catalogue.csv (500 SKUs)
# → Creates sales_history.csv (365,000 rows, 24 months)
# → Creates inventory_snapshot.csv

# 3. Run via Docker (recommended)
docker compose up --build
# API available at http://localhost:8000

# 4. Test a bilingual query
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what should I reorder before Ramadan?", "language": "en"}'

# Arabic query — same endpoint
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هي المنتجات التي يجب إعادة طلبها قبل رمضان؟", "language": "ar"}'

# 5. Run tests
docker compose exec api pytest tests/ -v
# 55 tests, all passing
```

---

## Key Engineering Decisions

### Why LangGraph over a LangChain chain?

A chain runs the same steps in the same order for every query. A reorder query and a search query have fundamentally different data requirements — running sales analysis and inventory checks on a simple search query wastes latency and compute. LangGraph's conditional edges let the agent take different paths based on live state. Search queries run 2 nodes. Reorder queries run 5. That routing decision is the difference between a pipeline and an agent.

### Margin-based confidence routing over absolute threshold

Absolute cosine thresholds drift with query length and phrasing. A 3-word Arabic query and a 15-word English query produce different score ranges for identical retrieval quality. The margin between top-1 and top-3 retrieved chunks is query-length invariant — a small margin means the retriever is uncertain regardless of absolute score level. Fallback requires both conditions: margin below threshold AND the queried entity absent from top-3 chunks. Either condition alone is insufficient.

Known limitation: chunks are deduplicated by content hash before margin calculation. Without deduplication, identical content in multiple source documents makes top-3 scores look artificially flat, triggering false fallbacks on correct retrievals.

### Field-type fallback split

Factual fields (SKU counts, stock levels, prices) use hard fallback — if retrieval is uncertain, return "insufficient data" and nothing else. A wrong inventory number is unrecoverable; a retailer acts on it immediately.

Narrative fields (trend summaries, reorder reasoning) allow partial synthesis with explicit provenance: *"Based on [source], [partial answer]. Note: this is partial — verify before acting."* A partial explanation is still useful. A wrong stock count is not.

### TypedDict over Pydantic for graph-internal state

LangGraph manages state transitions natively with TypedDict. Using Pydantic inside the graph adds serialization overhead on every node transition (5 transitions per query). Pydantic is used at the API boundary where validation matters. Inside the graph, TypedDict is the right tool. This distinction signals understanding of where validation cost is justified versus where it adds latency without benefit.

### Why multilingual-e5-large over separate Arabic/English models?

Cross-lingual retrieval: an Arabic query finding an English-tagged product works naturally in a shared embedding space. Two separate spaces cannot do this. The 300MB model size is the only trade-off, acceptable for a cloud-deployed Dubai retail system.

### Why ChromaDB over Pinecone?

Matches UAE data residency concerns (operational data stays local), runs without cloud dependency, and has built-in metadata filtering critical for `price < AED 200` type queries. At >1M vectors: Weaviate or managed Pinecone. Documented as a known scale boundary.

### Why daily sales granularity over weekly?

The UAE weekend is Friday–Saturday, not Saturday–Sunday. Weekly aggregation loses this signal and distorts DSF and Ramadan day-level spike detection. Daily granularity is a real production consideration — weekly aggregation is the common mistake.

### Caching scale ceiling

Current approach: TTL-based Redis invalidation tied to data upload events. Works at SME scale (hundreds to low thousands of SKUs).

Scale ceiling: breaks at millions of keys (noon-scale). Next architectural step: event-driven cache invalidation tied to inventory update events, LRU eviction policy, and partitioned Redis keys by product category.

---

## Production Bugs Fixed

Issues found during development that are not documented in tutorials or library docs:

**LangGraph node name clash (Phase 2)**  
LangGraph 0.2.x raises `ValueError` at startup if a node name matches a field name in the `AgentState` TypedDict. Fix: suffix all node names with `_node`. The state field `recommendation` and the node name `recommendation` cannot coexist.

**structlog reserved keyword (Phase 2)**  
`structlog` reserves `event` as its own parameter — it is the log message itself. Passing `event=` as a keyword argument to any `logger.*()` call raises `TypeError: got multiple values for argument 'event'`. Fix: rename to `upcoming_event=`.

**Relative import depth in pytest (Phase 2)**  
When pytest adds `backend/` to `sys.path` via `conftest.py`, modules resolve as `agents.*` not `backend.agents.*`. Three-dot relative imports (`from ...services import`) climb above the package root and fail with `ImportError: attempted relative import beyond top-level package`. Fix: use absolute imports from the `sys.path` root.

**PYTHONPATH mismatch between pytest and uvicorn (Phase 3 prep)**

Pytest used `conftest.py` to add `/app/backend` to `sys.path`, allowing bare `services.*`imports. uvicorn starts without that manipulation, so the same imports failed at runtime. Fix: `ENV PYTHONPATH=/app/backend` in the Dockerfile.

---

## Dataset

All data is **synthetic** — generated by `data/synthetic/generate_dataset.py`.

| File | Rows | Description |
|------|------|-------------|
| `product_catalogue.csv` | 500 | SKUs across 5 categories, bilingual names, AED pricing |
| `sales_history.csv` | 365,000 | Daily sales per SKU, 24 months, UAE event flags |
| `inventory_snapshot.csv` | 500 | Current stock levels, 15% in critical status |

UAE events modeled: Ramadan 2025/2026, Eid Al-Fitr/Al-Adha, DSF 2025/2026, National Day 2024/2025, Back-to-School.

**Known limitation:** Hijri calendar dates are approximate. Production fix: `hijri-converter` library. Documented as a known limitation — signals engineering maturity.

---

## Tests

```
tests/test_phase1.py — 25 tests
├── TestDataGeneration (8)      — dataset integrity, seasonality validation
├── TestLanguageDetection (5)   — Arabic/English/mixed detection
├── TestDocumentPreparation (4) — E5 prefix, bilingual document construction
├── TestConfidenceGate (3)      — threshold logic
└── TestCatalogueLoader (5)     — CSV loading, type casting, sales indexing

tests/test_phase2_agent.py — 30 tests
├── TestIntentClassifierEnglish (4)  — reorder / analysis / search / trend
├── TestIntentClassifierArabic (2)   — Arabic reorder and analysis
├── TestLanguageDetection (6)        — EN / AR / mixed / empty / SKU extraction
├── TestFactualIntentFlag (4)        — hard vs partial fallback routing
├── TestSeasonalityNode (4)          — DSF / Ramadan / National Day alert levels
├── TestInventoryNode (3)            — runway, critical flag, missing SKU sentinel
├── TestAgentRouting (5)             — conditional edge routing by intent
└── TestAgentEndToEnd (2)            — full trace, Groq rate limit degradation
```

---

## Project Context

This is the third project in a production AI engineering portfolio targeting a full-time AI Software Engineer role in Dubai by December 2026.

| Project | Core Tech | Market Focus |
|---------|-----------|--------------|
| Logistics Oracle | Dual-LLM routing, Prophet, Redis | JAFZA logistics operators |
| Dubai Property Intelligence | RAG, hallucination prevention, DLD data | Dubai real estate |
| **Dubai Retail Intelligence Agent** | **LangGraph, bilingual RAG, Arabic NLP** | **noon sellers, mid-market retail** |

Three projects. One coherent story. One market. One engineer.

---

*Started: August 2026 · Target completion: November 2026*  
*GitHub: github.com/Dula21*
