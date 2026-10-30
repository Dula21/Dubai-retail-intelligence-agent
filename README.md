# Dubai Retail Intelligence Agent

> Bilingual Arabic/English agentic RAG system for Dubai mid-market retail.
> Aims to give sellers who cannot afford a dedicated merchandising manager an autonomous
> merchandising layer: what to stock, when to order it, and why.

**Status:** Phase 2 complete (LangGraph agentic loop, calendar-driven seasonality, event-planning route) · 183 tests passing
**Stack:** LangGraph · LangChain · ChromaDB · multilingual-e5-small · Groq · FastAPI · Redis · Docker
**Wired, not yet used:** PostgreSQL (audit log, Phase 3) · **Planned:** Prophet forecasting (Phase 3)
**Data:** Synthetic. 500 SKUs, daily sales history, a UAE retail calendar for 2024–2027. See [Dataset](#dataset).

---

## The problem

Dubai mid-market retailers selling on noon, their own websites and physical stores have no unified system
connecting what is selling, what is running low, and what to stock for Ramadan, Eid, Diwali, DSF and National Day.
Reorder decisions are made on gut feel, event surges are missed every year, and Arabic/English catalogues are
inconsistently tagged and hard to search.

(The monthly cost of a merchandising manager is an assumption behind the project, not a measured figure.)

---

## Architecture

```
Query (Arabic or English)
   |
API layer: language detection (Unicode Arabic ratio) + rule-based intent classification
   |
   +-- event_planning  ("what should I stock more of before Diwali")
   |      event_candidates_node: event from the calendar -> SKUs the event lifts (no vector search)
   |         -> sales_analysis -> inventory_check -> seasonality_check -> recommendation
   |
   +-- everything else
          rag_retrieval_node: multilingual-e5-small + ChromaDB, margin-gated
             +-- low confidence -> fallback_node (no LLM) -> END
             +-- search / trend -> recommendation
             +-- reorder / analysis -> sales_analysis -> inventory_check -> seasonality_check -> recommendation

Every node appends to a human-readable reasoning trace that is returned to the caller.
Redis caches answers for 1 hour; the key is (code version, normalised query, date).
```

### Routing

- **Search and trend** queries: retrieval, then recommendation (2 nodes).
- **Reorder and analysis** queries: retrieval, sales, inventory, seasonality, recommendation (5 nodes).
- **Event-planning** queries: event candidates, sales, inventory, seasonality, recommendation (5 nodes, no vector search).
- **Fallback** exits at the retrieval node. No LLM call is made.

### Try it

```
what should I stock more of before Diwali        -> event planning, order-by date, real SKUs
what should I stock before Ramadan                -> same route, expected vs observed lift
how many units left of FSH-0014                   -> exact SKU lookup, factual, hard fallback if unknown
ما هي المنتجات التي يجب أن أخزنها قبل رمضان؟         -> Arabic planning question
```

---

## Quick start

```bash
# 1. Install
git clone https://github.com/Dula21/dubai-retail-intelligence-agent
cd dubai-retail-intelligence-agent
pip install -r requirements.txt

# 2. Generate the synthetic dataset (CSVs are not committed; generators are)
#    Exact commands are documented at the top of .gitignore.

# 3. Configure: .env needs GROQ_API_KEY and POSTGRES_PASSWORD

# 4. Run with Docker (code is baked into the image: rebuild after every change)
docker compose up -d --build
# first start downloads the embedding model and indexes the catalogue (a few minutes)

# 5. Ask a question
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what should I stock more of before Diwali"}'

# 6. Tests (the container runs Python 3.12 and langgraph 0.2.4: the reference environment)
docker compose exec api python -m pytest tests -q
```

The response contains the recommendation, a confidence score, the reasoning trace, the selected products
(`relevance_score` is `null` for calendar-selected SKUs, with `selection_basis` and `event_lift` instead) and the
seasonality signal.

---

## Key engineering decisions

### Event planning is its own intent, not a flavour of "reorder"
A reorder question is a lookup about specific products and is a factual field, so it goes through retrieval and the
hard fallback. "What should I stock before Diwali" names no product: near-identical variants tie within about 0.003,
so gating it on retrieval margin returned "insufficient data" to a question the calendar can answer. Planning now
starts from the calendar: event, then the SKUs it lifts, then the normal signal chain.
*Trade-off:* the lift table is synthetic, and equal-lift SKUs are ranked by shortest stock runway.

### One calendar drives data, seasonality and (next) forecasting
`uae_retail_calendar.csv` plus `event_lifts.csv` feed the data generator and the seasonality node. Hijri dates use
Umm al-Qura; Dubai Shopping Festival is marked `estimate_verify` because its dates are announced yearly.
The order-by date is `event start - lead-in days - supplier lead days - 5 buffer days`, and the alert level is keyed to
that date, not to the event date.
*Trade-off:* the file must be regenerated each year; the old hard-coded table went stale silently.

### Expected and observed lift are never mixed
The calendar says what to expect (for example 3.5x). The sales node measures what the history shows for the event that
was asked about. Both are shown, and the prompt forbids calling the expectation a past result. Past sales are censored
by stockouts, so observed lift can read low.

### Honest data handling
Missing stock is `-1`, never `0`, and an unknown SKU is never "critical". Supplier lead time that had to be assumed is
flagged in the trace and the prompt. The recommendation states no order quantities: a reorder point is a trigger level,
not an order size (a deterministic quantity engine is Phase 3). Confidence is a data-coverage score, not a calibrated
probability, and never reaches 1.0 on the planning route.

### Margin-based confidence routing over an absolute threshold
The 0.72 cosine floor does not discriminate on this catalogue: 16 of 16 not-in-catalogue queries passed it
(see `eval/RESULTS.md`). Routing uses the margin between the best product and the next *different* product (variants
collapsed). Fallback needs both a small margin and the queried entity absent from the top results.
The threshold is 0.015 for e5-small (it was 0.01 for e5-large; the smaller models give larger margins, so the same number catches fewer wrong results).
*Limit:* the threshold was read off the queries it was evaluated on (59 Gulf queries plus 16 not-in-catalogue); a held-out set is needed.

### Field-type fallback split
Factual fields (SKU counts, stock, prices) use a hard fallback: uncertain retrieval returns "insufficient data" and
nothing else. Narrative fields (trend summaries, reorder reasoning) allow a partial answer with provenance.

### LangGraph over a chain
Different questions need different data. Search runs 2 nodes; reorder and planning run 5. Conditional edges, including
two entry routes, are what make it an agent and not a fixed pipeline.

### TypedDict inside the graph, Pydantic at the API boundary
Validation cost is justified where untrusted input arrives, not on every node transition.

### One multilingual embedding model, sized by measurement (e5-small, not e5-large)
One shared embedding space lets an Arabic query find an English-tagged product, so there is no per-language routing. Three sizes of the same family were measured on a 4-CPU Docker host:

| Model | Gulf set top-1 (59) | Right product in top 3 | Embedding p50 | `/search` p95 |
|---|---|---|---|---|
| e5-large | 0.932 | 100% | 465 ms | 872 ms (misses the 500 ms target) |
| e5-base | 0.864 | 96.6% | 177 ms | not measured end to end |
| e5-small (chosen) | 0.864 | 96.6% | 35 ms | 76 ms |

*Why small:* it meets both project targets (retrieval accuracy above 85%, catalogue query under 500 ms). *Trade-off:* about 7 points of top-1 accuracy on the Gulf set, mostly near-ties between neighbouring products, which the margin check and the partial-answer fallback are designed to catch. The 0.864 margin over the 85% target is thin on 59 synthetic queries. *Alternative kept in mind:* two models (small for search, large for the agent), rejected for now as more RAM, two indexes and more code for a small gain. Details in `eval/RESULTS.md`.

### ChromaDB over Pinecone
Runs locally (relevant to UAE data-residency concerns) with built-in metadata filtering. Documented boundary: above
~1M vectors, use a distributed store.

### Daily sales granularity
The UAE weekend has been Saturday–Sunday since 2022; weekly aggregation hides day-level spikes around Ramadan and DSF.

### Caching scale ceiling
Current approach: 1-hour TTL, keyed on query, date and code version. Works at SME scale (hundreds to low thousands of
SKUs). It breaks at millions of keys (noon scale). Next architectural step: event-driven invalidation tied to inventory
updates, an LRU eviction policy and Redis keys partitioned by product category.

---

## Production bugs found and fixed

- **LangGraph node name clash.** LangGraph raises `ValueError` if a node name equals an `AgentState` field name. All
  nodes carry a `_node` suffix.
- **structlog reserved keyword.** `event=` collides with structlog's own parameter (`TypeError: got multiple values`).
  It came back in a later node and was renamed (`upcoming_event=`, `event_name=`).
- **Relative imports under pytest vs uvicorn.** Tests resolve `agents.*`, uvicorn resolves `backend.agents.*`. The
  Dockerfile sets `PYTHONPATH=/app/backend` so both styles work.
- **A mocked retriever hid a real integration bug.** Unit tests mocked a list shape; the real retriever returns a
  dictionary, so every query failed on real data while 59 tests were green. Integration tests now use the real shape.
- **Stock silently read as zero.** The node looked for `current_stock`; the data has `current_stock_units`. Fix: read the
  real columns, use `-1` for missing, and never flag unknown stock as critical.
- **Code baked into the Docker image.** A plain `docker compose up` ran old code. Rebuild after every change.
- **Silent stale vector index (fixed).** The Docker volume kept a 300-document index while the catalogue had 500 SKUs, because
  startup skipped re-ingestion whenever documents existed. Found when container tests diverged from local ones
  (top-1 scores matched to 0.001; the margin differed 6x). Fix: the collection name carries a content fingerprint, a
  changed catalogue builds a new collection, old ones are pruned. Covered by 10 tests; 178 pass locally and in Docker
  on a fresh volume.
- **`.env` risk in the image.** `.dockerignore` keeps secrets, history and generated data out of the build context.
- **Empty LLM answers (found by per-node timing, fixed).** The first per-node profile showed 4 of 6 agent questions ending
  in the trace-only fallback with confidence 0.0: `openai/gpt-oss-20b` is a reasoning model, and hidden reasoning tokens
  used up the completion budget before any visible answer was written. The unit tests could not see this because they use
  a fake model client. Fix: `reasoning_effort` is sent as `low` for gpt-oss models (`GROQ_REASONING_EFFORT` to change it).
  After the fix, 24 of 24 questions returned real answers (see Performance). Not yet compared: answer quality at `low`
  versus `medium` reasoning.

### Known issues (open)
- Lift multipliers are synthetic, not learned from history. Equal-lift SKUs are ranked by runway.
- No order-quantity engine yet; confidence is not calibrated (Phase 3).
- Agent latency is usually about 1.3 s but has occasional spikes of 5 to 20 s. Nearly all of that time is the single Groq call; the cause of the spikes is not established (free-tier queueing is a guess, not a finding).
- The first agent request after a container start takes about 8 s because the 365,000-row sales file is loaded lazily. Preloading it at startup is not done yet.
- Search latency on a laptop is not perfectly stable: in 7 runs the `/search` p95 was 71 to 99 ms six times and 694 ms once (cause not established).
- The smaller embedding model costs about 7 points of top-1 accuracy on the Gulf set; the 85% target is met only narrowly.

---

## Performance

Measured with `eval/measure_latency.py`: one client, sequential requests, client-side round trip, Docker on a 4-CPU laptop (no GPU), synthetic 500-SKU catalogue, e5-small.

Search figures come from 7 separate runs of `eval/measure_latency.py` (50 requests each); agent figures from 2 runs of 8 distinct questions plus one direct profile run.

| Case | Result |
|---|---|
| `/search`, 50 different English and Arabic queries (cold embedding) | p50 41 to 44 ms in 6 of 7 runs (68 ms in the other). p95: 71, 76, 82, 86, 90, 99 ms in six runs and **694 ms in one run**. Target (p95 under 500 ms) met in 6 of 7 runs. |
| `/search`, same query repeated | p50 42 to 74 ms, p95 86 to 195 ms |
| Agent, distinct questions (calls Groq) | 16 questions in 2 runs, all real answers. p50 1.33 s in both runs; slowest request 19.5 s and 8.4 s. |
| Agent, same question repeated (Redis hit) | p50 9 to 12 ms, p95 14 to 22 ms (n=20 per run) |

Where the agent's time goes (direct profile, `eval/profile_agent.py`, 7 questions after the first): the Groq call is about two thirds of a typical run, sales analysis about 28% (0.37 s), everything else under 2%. Typical Groq call 0.7 to 0.9 s; in 3 of those 7 questions it took 4.8 to 11 s.
Limits: `/search` has no cache of its own; agent numbers come from small samples on the Groq free tier, so no p95 is quoted (with 8 requests it is just the slowest one); one client, no concurrency, so this is not a load test; laptop under Docker, not a server.

---

## Dataset

All data is **synthetic**, produced by `data/generate_synthetic.py` and `data/make_calendar.py`. CSVs are ignored by
git; regenerate them with the commands documented in `.gitignore`. Exact counts are in
`data/synthetic/dataset_metadata.json`.

| File | Description |
|------|-------------|
| `product_catalogue.csv` | 500 SKUs in five categories, bilingual names, AED prices, supplier lead times |
| `sales_history.csv` | Daily sales per SKU, generated from the calendar, censored by stockouts, with a `stockout_occurred` flag |
| `inventory_snapshot.csv` | Current stock per SKU: 324 healthy, 127 critical, 49 overstock |
| `uae_retail_calendar.csv` | 17 events across 2024–2027 (67 rows): Ramadan, both Eids, Diwali, Christmas, White Friday, DSF, National Day and more |
| `event_lifts.csv` | Expected demand multiplier per event, category and keyword |
| `ground_truth_demand.csv` | True uncensored demand. Evaluation only; the agent never reads it |

**Known limitations:** the lift table is synthetic and every number is hypothetical; DSF dates are estimates until
announced; the catalogue has no festival-specific products yet (diyas, sweet boxes, Christmas decor).

---

## Tests

183 tests across: Phase 1 retrieval and data checks, synthetic-data invariants, real-CSV inventory checks, intent and
routing, calendar seasonality, the recommendation node (fake model client), the event-planning route end to end on real
data, sales-event analysis, the API layer (null relevance for calendar-selected SKUs, cache version, health), the RAG
node with the real retriever, the real-retriever smoke cases, stale-index detection (fake embedder, real Chroma), and the per-node timing wrapper.

```
python -m pytest tests -q
```

The container run is the reference (Python 3.12, langgraph 0.2.4). Model-loading tests are slow on CPU.

---

## Project context

Third project in a production AI engineering portfolio targeting a full-time AI Software Engineer role in Dubai.

| Project | Core tech | Market focus |
|---------|-----------|--------------|
| Logistics Oracle | Dual-LLM routing, Prophet, Redis | JAFZA logistics operators |
| Dubai Property Intelligence | RAG, hallucination prevention, DLD data | Dubai real estate |
| **Dubai Retail Intelligence Agent** | **LangGraph, bilingual RAG, Arabic NLP** | **noon sellers, mid-market retail** |

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Bilingual RAG over the product catalogue | Complete |
| 2 | LangGraph agentic loop, event planning | Complete |
| 3 | Autonomous reorder engine: quantities, calibrated confidence, human override, audit log | Planned |
| 4 | Arabic dialect polish and bilingual response generation | Planned |

*Started: August 2026 · GitHub: github.com/Dula21*