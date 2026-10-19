# Bilingual retrieval evaluation (Dubai Retail Intelligence Agent)

Date: 2026-10-05 (Phase 1) · Phase 2 section added 2026-10-14 · Embedding-model section added 2026-10-19 · Baseline commit: `6fdc697` · Model for the Phase 1 results below: `intfloat/multilingual-e5-large` (production model is now e5-small, see the last results section) · Vector store: ChromaDB (cosine)

> **Data transparency:** the catalogue and all queries are **synthetic**. No real retailer data or real user logs were used.
> Queries were drafted with an AI assistant and have not been validated by native Gulf Arabic speakers.

## Setup

- **Catalogue:** 500 SKUs, five categories (fashion 120, electronics 100, home 100, beauty 100, food 80).
  It contains **75 base products**, each repeated 5–8 times with suffix variants (e.g. "Summer Dress", "Summer Dress Pro").
- **Ground truth is built from the catalogue**, not hand-guessed: a query's correct answers are all SKUs of the target base product(s).
  (An earlier first attempt used guessed regexes and left 31 of 50 queries with no ground truth; it was discarded.)
- **Retrieval-only evaluation.** The confidence gate is switched off so raw scores are visible.
- Variants of one product are **collapsed to a single base product** before scoring, so "top-1" means best product, not best SKU.

| Test | Script | Queries |
|---|---|---|
| Exact catalogue names, English vs Arabic | `run_baseline_v2.py` | 75 products × 2 languages, plus 8 not-in-catalogue products × 2 languages |
| Gulf dialect, code-switched, spelling variants | `run_gulf_test.py` | 59 (39 Gulf, 15 code-switched, 5 spelling) |
| Dense vs BM25 vs BM25+RRF | `run_rrf_test.py` | same 59 |

## Results

### 1. Exact catalogue names: no English/Arabic gap

| Query language | hit@1 | hit@5 | precision@5 |
|---|---|---|---|
| English | 75/75 (1.00) | 1.00 | 1.00 |
| Arabic | 75/75 (1.00) | 1.00 | 0.995 |

All five categories scored 1.00 in both languages. This is an upper bound (the query is the catalogue name itself) and a pipeline sanity check.

### 2. Gulf-style queries: top-1 accuracy by configuration

| Configuration | Gulf (n=39) | Code-switched (n=15) | Spelling (n=5) | **All (n=59)** | Right product in top 3 |
|---|---|---|---|---|---|
| Baseline (documents embedded **without** the E5 `passage:` prefix) | 0.897 | 0.867 | 1.00 | **0.898** (53/59) | 100% |
| + `passage:` prefix fix | 0.923 | 0.933 | 1.00 | **0.932** (55/59) | 100% |
| BM25 only | 0.769 | 1.00 | 0.60 | 0.814 (48/59) | 81% |
| Dense + BM25 fused with RRF | 0.949 | 0.933 | 1.00 | **0.949** (56/59) | 100% |

- The prefix fix flipped 2 queries (`مصلى`, `oud perfume رجالي`). RRF flipped 1 more (`سماعات بلوتوث`), a case where the ground truth is debatable.
- **Small differences (1–2 queries of 59) are within noise on this sample.** The prefix fix is justified by the model documentation, not by this margin of improvement.
- BM25 alone was perfect on code-switched queries containing an English word, and weak on Arabic dialect words and alternate spellings.

Remaining misses after the prefix fix:

| Query | Wanted | Top-1 returned | Likely cause |
|---|---|---|---|
| `جاكيت ثقيل للبرد` | Winter Coat | Denim Jacket (gap 0.002) | near-tie; "heavy" is not captured |
| `سبيكر` | Bluetooth Speaker | Sumac Spice | catalogue has no Arabic word for "speaker"; needs an alias/synonym layer |
| `dates فاخر` | Medjool Dates 1kg | Phone Case Premium / Saffron Premium | "فاخر" matches "Premium" in other products |
| `سماعات بلوتوث` | Earbuds or Gaming Headset | Bluetooth Speaker | arguably acceptable; ground truth is debatable |

### 3. The 0.72 cosine gate does not discriminate

| Signal | Value |
|---|---|
| Not-in-catalogue queries that **passed** the 0.72 gate | **16 of 16** (8 products × English/Arabic) |
| Top-1 score, not-in-catalogue products | 0.77–0.83 (mean 0.79) |
| Top-1 score, correct exact-name matches | mean 0.88 (EN) / 0.86 (AR) |
| Top-1 score, Gulf set: correct vs wrong+absent | mean 0.833 vs 0.794 |

E5 cosine scores occupy a narrow high band, so a fixed floor of 0.72 does not separate good from bad retrieval.
Arabic exact-name queries scored about 2.5% lower on average than English (0.857 vs 0.879).

**Margin to the next *different* product** (variants collapsed), Gulf set after the prefix fix. Correct n=55, wrong+absent n=20:

| Rule | Correct answers kept | Wrong/absent caught |
|---|---|---|
| margin ≥ 0.005 | 0.89 | 0.55 |
| margin ≥ 0.01 | 0.78 | 0.90 |
| margin ≥ 0.015 | 0.67 | 0.90 |
| margin ≥ 0.02 | 0.62 | 0.95 |
| score ≥ 0.80 | 0.80 | 0.70 |
| score ≥ 0.84 | 0.42 | 0.95 |

Without collapsing variants, the top-1 vs top-3 margin is about 0.005 for correct and wrong results alike, because the top 3 are variants of the same product.
**Caveat:** thresholds were read off the same queries they are evaluated on. One wrong/absent query is worth 5 points in the right-hand column. Treat this as direction, not a calibrated setting. A held-out set is needed.

## Findings and decisions

| Finding | Decision |
|---|---|
| Documents were embedded without the E5 `passage:` prefix (a bug in `prepare_document_text`) | Fixed; regression test added |
| Absolute 0.72 floor never fires | Keep as minimum floor only; add margin-based routing with variants collapsed |
| BM25 + RRF is neutral at this scale | Not shipped. Revisit with real SKU/product-name traffic |
| SKU codes are not part of the embedded text, so vector search cannot find them | Route SKU-pattern queries to an exact catalogue/SQL lookup; factual fields use hard fallback |
| Right product is always in the top 3 | On low margin, show the top distinct products with a confidence label instead of a single answer (narrative fields only); implemented in Phase 2 in `fallback_node` (hard "insufficient data" for factual intents, partial answer with provenance for narrative ones) |

## Phase 2 findings (agent loop, event planning, Docker)

Phase 1 numbers above are retrieval-only and unchanged. Phase 2 findings come from running the agent end to end.

| Finding | Evidence | Decision |
|---|---|---|
| A planning question ("what should I stock more of before Diwali") hit the hard "insufficient data" fallback | Near-identical variants tie within about 0.003, so the retrieval margin gate fires on a question the calendar can answer | New `event_planning` intent, routed calendar-first: event, then SKUs the event lifts, then sales, inventory, seasonality, recommendation. Deterministic until the final explanation. Treated as a narrative intent |
| The LLM presented the calendar's 3.5x expected lift as a past lift | Seen in a live Ramadan answer | Expected lift (calendar table) and observed lift (sales history in event windows vs baseline outside all events) are reported separately; prompt rule forbids mixing them |
| Observed Ramadan lift across 8 SKUs: 1.26x to 3.38x; calendar table says 3.5x | Sales node trace | Reported as a range. Observed history is censored by stockouts, so observed lift can read low (suspected, not yet tested against `ground_truth_demand.csv`) |
| Recommendation answered stock-today and the event deadline as one question | Review of live answers | Timing is computed in code (order-by = start - lead-in - max supplier lead - 5 day buffer); a WATCH deadline is never described as urgent; no order quantities are stated (a reorder point is not an order quantity) |
| Confidence of 1.00 on event answers | Live JSON | Confidence is a data-coverage score, not a calibrated probability. Event path: 0.2 selection + 0.3 sales + 0.2 inventory + 0.1 no-error, minus 0.1 if the lead time is assumed. Maximum 0.8 |
| API returned a similarity score for SKUs that were chosen by the calendar | Router JSON | `relevance_score` is null for calendar-selected SKUs; `selection_basis` (`calendar_lift` or `similarity`) and `event_lift` added |
| **Silent stale vector index in Docker** | The container held 300 documents while the catalogue had 500 SKUs; startup skipped ingestion whenever documents existed. Container and local top-1 scores agreed to within 0.001 (0.7847 vs 0.7836) but the margin differed (0.0011 vs 0.0065) | Collection name now carries a content fingerprint; a changed catalogue builds a new collection and old ones are pruned. 10 regression tests. I first blamed library version drift; the logs showed that was wrong |

Test status: 178 passed locally (Python 3.14) and 178 passed in the Docker image on a fresh Chroma volume (Python 3.12, chromadb 0.5.3).

Not yet done: the lift table is synthetic; observed lift has not been compared with `ground_truth_demand.csv`; margin thresholds are still not validated on held-out data; no deterministic order-quantity engine; confidence is not calibrated.

## Embedding model choice and latency (2026-10-19)

**Question:** can `/search` meet the 500 ms target, and at what accuracy cost?
**Method:** `eval/measure_latency.py` (end-to-end `/search` and agent latency), `eval/profile_embedding.py` (embedding step alone, inside the container), `eval/run_candidate_eval.py` (the Gulf and exact-name evals for another model, in its own index folder). Docker on a 4-CPU laptop, single client.

### Latency: the embedding step is the bottleneck
| Model | Embedding p50 / p95 | `/search` p50 / p95 (50 distinct queries) |
|---|---|---|
| e5-large | 465 / 1091 ms (n=20, 2 threads) | 502 / 872 ms (**target missed**) |
| e5-base | 177 / 815 ms (n=40; noisy run) | not measured |
| e5-small | 35 / 94 ms (n=40) | 44 / 76 ms (**target met**) |

The server-reported time was 490 ms of a 502 ms round trip for e5-large, so almost all of the time was inside the search call. More CPU threads did not help on this host (1 thread: 1202 ms, 2: 465 ms, 4: 771 ms; noisy). A repeated query cost the same as a new one: `/search` has no cache.

### Accuracy on the 59 Gulf-style queries (same queries, same method)
| Model | top-1 all | Gulf (39) | code-switched (15) | spelling (5) | right product in top 3 | exact names, Arabic top-1 |
|---|---|---|---|---|---|---|
| e5-large | 0.932 (55/59) | 0.923 | 0.933 | 1.00 | 100% | 75/75 |
| e5-base | 0.864 (51/59) | 0.872 | 0.800 | 1.00 | 96.6% | 74/75 |
| e5-small | 0.864 (51/59) | 0.846 | 0.867 | 1.00 | 96.6% | not recorded |

Base and small tie overall; differences between them are one query per group. The extra misses (versus large) are mostly near-ties between neighbouring products, for example `ملابس سباحة للمحجبات` (right product second by 0.005), plus cases where the right product fell out of the top 3 (`مخدات للمجلس` for both smaller models, `dates فاخر` for small, `سبيكر` for base).

### Margin threshold must change with the model
Margin to the next different product (variants collapsed). Correct n=55 for large, 51 for base and small; wrong or absent n=20 for large, 24 for the others, so the columns are not exactly comparable.

| Rule | large: kept / caught | base: kept / caught | small: kept / caught |
|---|---|---|---|
| margin >= 0.01 | 0.78 / 0.90 | 0.82 / 0.71 | 0.82 / 0.67 |
| margin >= 0.015 | 0.67 / 0.90 | 0.75 / 0.88 | 0.77 / 0.88 |

Production threshold changed from 0.01 (large) to 0.015 (small). Same caveat as before: thresholds were read off the same queries; a held-out set is still needed. The 0.72 score floor still does not discriminate: all 16 not-in-catalogue queries passed it with e5-base as well.

### Decision
Switched production to **e5-small**: 13x faster at the median, meets both project targets (accuracy above 85%, catalogue query under 500 ms). Cost: about 7 points of top-1 on this sample; the 85% margin is thin on 59 synthetic queries. Rejected for now: two models (small for search, large for the agent) because of the extra RAM, two indexes and code; ONNX/int8 quantisation of e5-large (not tested); caching query embeddings (does not help the cold path).
A bug found on the way: the index fingerprint did not include the model name, so switching models would have reused the old index. Fixed, with a test.

### Agent latency (not improved by this change, and expected not to be)
The agent path takes 10 to 20 seconds when it calls Groq (n=8 per run, free tier, large variation: median 9.8 s with e5-large, 14.5 s with e5-small; with 8 requests this is most likely run-to-run noise, since the agent does not spend its time in the embedding step). The embedding step was about 5% of it. Where the rest goes has not been profiled per node. A cached agent answer takes about 10 ms.

Test status after the switch: 179 passed locally (Python 3.14).

## Limitations

- Synthetic catalogue with 75 base products; no footwear, phones, TVs or laptops. Results do not transfer to a real, more diverse catalogue.
- 59 queries is a small sample; queries are synthetic and not native-validated.
- Phase 1 numbers are retrieval only and were measured with e5-large. Latency is now measured (see the embedding-model section); it is a single-client test on a 4-CPU laptop, not a load test.
- Not tested: BGE-M3, a cross-encoder reranker, translate-query-to-English, Arabic normalisation of documents (only light normalisation inside BM25).
- Confidence thresholds are not validated on held-out data.

## Reproduce

```
python -m eval.run_baseline_v2
python -m eval.run_gulf_test
python -m eval.run_rrf_test
```

`PERSIST_DIR` in `run_baseline_v2.py` selects the Chroma index; the Gulf and RRF scripts import it.
After the final re-ingest it should point back to `.chroma_db`.

## Scale note

BM25 here is an in-process implementation over 500 documents. At catalogue sizes in the hundreds of thousands, the next step is a proper search engine with sparse + dense retrieval; that is documented as a next architectural step, not implemented.