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

*Project started: August 2026*
*Target completion: November 2026*
*GitHub: github.com/Dula21*
