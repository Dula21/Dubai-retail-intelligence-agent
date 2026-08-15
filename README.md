# Dubai Retail Intelligence Agent
## Project Brief & Claude Project Instructions

---

## What This Project Is

An agentic RAG system that acts as an autonomous retail merchandising intelligence layer for Dubai mid-market retailers. It connects product catalogue data, sales history, and market trend signals — then autonomously runs a multi-step reasoning loop to answer questions like "what should I reorder this week before DSF" or "why is this product underperforming compared to similar SKUs."

The system replaces the role of a human merchandising manager — which costs AED 15,000–25,000/month — that mid-market Dubai retailers cannot afford.

---

## The Problem It Solves

Dubai mid-market retailers selling on noon, their own website, and physical stores simultaneously have no unified system that connects:
- What customers are searching for
- What is currently selling
- What inventory is running low
- What to stock next

They make reorder decisions based on gut feel. They miss demand surges during DSF, Ramadan, and National Day every year. Their product catalogues exist in Arabic and English with inconsistent tagging and poor searchability.

This project solves that problem autonomously.

---

## Full Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Agentic Framework | LangGraph | Multi-step autonomous reasoning loop |
| Chain Orchestration | LangChain | Document loaders, retrievers, prompt chains |
| RAG | Custom RAG pipeline | Product catalogue + sales data retrieval |
| Forecasting | Prophet | Demand signals, seasonality modeling |
| LLM — Cloud | Groq (Llama 3.1) | Fast inference for English queries |
| LLM — Arabic | Jais or bilingual embeddings | Arabic query handling |
| Backend | FastAPI | API layer |
| Database | PostgreSQL | Persistent storage |
| Cache | Redis | Performance layer |
| Frontend | Next.js | Dashboard UI |
| Embeddings | HuggingFace bilingual model | Arabic + English retrieval |

---

## Build Phases

### Phase 1 — Bilingual RAG over Product Catalogue (2 weeks)
- Ingest product catalogue data (Arabic + English)
- Build bilingual embedding layer
- Implement confidence-gated retrieval
- Answer queries like: "show me all underperforming SKUs in the footwear category"

### Phase 2 — LangGraph Agentic Loop (2 weeks)
- Build multi-step reasoning loop connecting RAG to forecasting
- Agent decides what data to fetch next autonomously
- Flow: check sales → check inventory → check trends → generate insight
- Implement state management across reasoning steps

### Phase 3 — Autonomous Reorder Recommendation Engine (2 weeks)
- Generate reorder recommendations without human approval at each step
- Explainable reasoning trace — agent shows its work
- DSF, Ramadan, National Day seasonality detection built in
- Critical runway override (same pattern as Logistics Oracle)

### Phase 4 — Arabic Query Handling (1 week)
- Full Arabic query support via bilingual embeddings
- Gulf Arabic dialect handling
- Bilingual response generation
- Mixed Arabic/English product name handling

---

## Data Strategy

### No proprietary data available — use these public sources:

1. **Kaggle retail datasets** — search "retail sales dataset Arabic" or "e-commerce product catalogue"
2. **UCI ML Repository** — Online Retail dataset (real transaction data, adaptable to UAE context)
3. **noon public product listings** — scrape publicly available product catalogue structure
4. **Synthetic data generation** — use Claude to generate realistic UAE retail product catalogue with Arabic/English fields, Ramadan/DSF seasonality patterns baked in

### Recommended starting point:
Generate a synthetic dataset of 500 SKUs across 5 categories (fashion, electronics, home, food, beauty) with:
- Arabic and English product names
- 24 months of sales history with Ramadan/DSF spikes
- Current inventory levels
- Supplier lead times
- Price points in AED

---

## Architecture Overview

```
User Query (Arabic or English)
        ↓
Language Detection Layer
        ↓
Bilingual Embedding + RAG Retrieval
        ↓
Confidence Gate (threshold: 0.72)
        ↓
LangGraph Agent Loop
    ├── Node 1: Sales Analysis
    ├── Node 2: Inventory Check  
    ├── Node 3: Demand Forecast (Prophet)
    ├── Node 4: Market Trend Signal
    └── Node 5: Recommendation Generation
        ↓
Explainable Reasoning Trace
        ↓
Bilingual Response (Arabic/English)
        ↓
Audit Log (PostgreSQL)
```

---

## Key Differentiators From Logistics Oracle

| Feature | Logistics Oracle | Dubai Retail Intelligence Agent |
|---------|-----------------|--------------------------------|
| Architecture | Dual-LLM routing | Full agentic loop (LangGraph) |
| Language | English only | Arabic + English bilingual |
| Autonomy | Single query response | Multi-step autonomous reasoning |
| Target market | JAFZA logistics operators | Dubai mid-market retailers / noon sellers |
| Reasoning trace | No | Yes — explainable decisions |

---

## Target Company Relevance

| Company | Why This Project Is Relevant |
|---------|----------------------------|
| noon | Directly solves noon seller ecosystem pain point |
| Talabat | Inventory and demand forecasting for F&B retail |
| Careem | Last-mile retail delivery demand signals |
| G42 | Agentic AI + Arabic NLP + sovereign data patterns |
| Presight | Predictive retail analytics architecture |

---

## Newsletter Arc (Arc 3) — Editions 19–24

| Edition | Title | Core Engineering Story |
|---------|-------|----------------------|
| 19 | Why I Am Building a Third Project | The retail problem I keep watching Dubai operators get wrong |
| 20 | LangGraph vs LangChain | Why the agent loop changes everything — not just chains |
| 21 | Building Bilingual RAG | The Arabic tokenization problem in actual code |
| 22 | Connecting Forecasting to Agentic Reasoning | The architecture decision that ties Prophet to LangGraph |
| 23 | The Reorder Recommendation Engine | What autonomous actually means in production |
| 24 | What Three Production AI Systems Taught Me | Synthesis — the Dubai AI engineer's real curriculum |

---

## LinkedIn Positioning

Every edition and every GitHub commit on this project should reinforce this narrative:

**"I am an AI engineer who builds production systems for the Dubai market — bilingual, agentic, and grounded in the specific operational rhythms of this market."**

That narrative connects:
- Logistics Oracle (JAFZA seasonality, dual-LLM, forecasting)
- Dubai Property Intelligence (RAG, hallucination prevention, DLD data)
- Dubai Retail Intelligence Agent (agentic AI, Arabic NLP, noon seller market)

Three projects. One coherent story. One market. One engineer.

---

## GitHub Repository Structure

```
dubai-retail-intelligence-agent/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── agents/
│   │   ├── retail_agent.py      # LangGraph agent definition
│   │   └── nodes/               # Individual agent nodes
│   ├── rag/
│   │   ├── retriever.py         # Bilingual retrieval layer
│   │   └── embeddings.py        # Arabic/English embedding model
│   ├── forecasting/
│   │   └── demand_forecast.py   # Prophet integration
│   ├── models/                  # Pydantic schemas
│   └── database/                # PostgreSQL + SQLAlchemy
├── frontend/
│   └── (Next.js dashboard)
├── data/
│   └── synthetic/               # Generated UAE retail dataset
├── tests/                       # Automated test suite
└── README.md
```

---

## Instructions for Claude When Working on This Project

### Context
You are working with Dulasi Nethma — an AI/ML Engineer and Full-Stack Developer based in Dubai. This is the third production AI project in a portfolio targeting a full-time AI Software Engineer role by December 2026, with a Tech Lead trajectory by 2031.

### How to help
- Always write production-grade code, not prototype code
- Every architectural decision should be explainable — document the why, not just the what
- Keep UAE market context in mind — Ramadan/DSF seasonality, Arabic/English bilingual requirements, data residency considerations
- Reference Logistics Oracle patterns where relevant (dual-LLM routing, Redis caching, Prophet forecasting, Pydantic validation)
- Reference DPI patterns where relevant (RAG pipeline, confidence gating, hallucination prevention, grounding prompts)
- When suggesting new approaches, explain the trade-off clearly

### Code standards
- FastAPI with async/await throughout
- Pydantic for all data validation
- Type hints on every function
- Docstrings on every class and public method
- Tests written alongside implementation, not after
- Redis caching on all expensive operations
- Structured logging via structlog
- Environment variables for all secrets and config

### Priority order when making decisions
1. Does it work correctly?
2. Is it production-safe (auth, validation, error handling)?
3. Is it fast enough (caching, async, efficient queries)?
4. Is it documented clearly enough that a hiring manager reading the GitHub understands every decision?

---

*Project started: August 2026*
*Target completion: November 2026*
*GitHub: github.com/Dula21*
