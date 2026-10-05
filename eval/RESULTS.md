# Bilingual retrieval evaluation (Dubai Retail Intelligence Agent)

Dates: 2026-10-05 / 2026-10-06 · Baseline commit: `6fdc697` · Model: `intfloat/multilingual-e5-large` · Vector store: ChromaDB (cosine)
Final index: `.chroma_db`, rebuilt with the E5 `passage:` document prefix. Sections 1–3 numbers were re-run on it and match the experiment index.

> **Data transparency:** the catalogue and all queries are **synthetic**. No real retailer data or real user logs were used.
> Queries were drafted with an AI assistant and have not been validated by native Gulf Arabic speakers.

## Setup

- **Catalogue:** 500 SKUs, five categories (fashion 120, electronics 100, home 100, beauty 100, food 80).
  It contains **75 base products**, each repeated 5–8 times with suffix variants (e.g. "Summer Dress", "Summer Dress Pro", "Summer Dress Lite", "Summer Dress 6").
- **Ground truth is built from the catalogue**, not hand-guessed: a query's correct answers are all SKUs of the target base product(s).
  (An earlier first attempt used guessed regexes and left 31 of 50 queries with no ground truth; it was discarded.)
- **Retrieval-only evaluation** (sections 1–3). The confidence gate is switched off so raw scores are visible.
- Variants of one product are **collapsed to a single base product** before scoring, so "top-1" means best product, not best SKU.

| Test | Script | Queries |
|---|---|---|
| Exact catalogue names, English vs Arabic | `run_baseline_v2.py` | 75 products × 2 languages, plus 8 not-in-catalogue products × 2 languages |
| Gulf dialect, code-switched, spelling variants | `run_gulf_test.py` | 59 (39 Gulf, 15 code-switched, 5 spelling) |
| Dense vs BM25 vs BM25+RRF | `run_rrf_test.py` | same 59 |
| Agent node on the real retriever | `smoke_agent.py`, `tests/test_rag_node_integration.py` | 8 hand-picked queries, 17 automated tests |

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
| + `passage:` prefix fix (**final index**) | 0.923 | 0.933 | 1.00 | **0.932** (55/59) | 100% |
| BM25 only | 0.769 | 1.00 | 0.60 | 0.814 (48/59) | 81% |
| Dense + BM25 fused with RRF | 0.949 | 0.933 | 1.00 | **0.949** (56/59) | 100% |

- The prefix fix flipped 2 queries (`مصلى`, `oud perfume رجالي`). RRF flipped 1 more (`سماعات بلوتوث`), a case where the ground truth is debatable.
- **Small differences (1–2 queries of 59) are within noise on this sample.** The prefix fix is justified by the model documentation, not by this margin of improvement.
- BM25 alone was perfect on code-switched queries containing an English word, and weak on Arabic dialect words and alternate spellings.
- The final index (rebuilt from scratch into `.chroma_db`) reproduced the 0.932 / 0.949 results exactly.

Remaining misses after the prefix fix:

| Query | Wanted | Top-1 returned | Likely cause |
|---|---|---|---|
| `جاكيت ثقيل للبرد` | Winter Coat | Denim Jacket (gap 0.004) | near-tie; "heavy" is not captured |
| `سبيكر` | Bluetooth Speaker | Sumac Spice | catalogue has no Arabic word for "speaker"; needs an alias/synonym layer |
| `dates فاخر` | Medjool Dates 1kg | Phone Case Premium / Saffron Premium | "فاخر" matches "Premium" in other products |
| `سماعات بلوتوث` | Earbuds or Gaming Headset | Bluetooth Speaker | arguably acceptable; ground truth is debatable |

### 3. The 0.72 cosine gate does not discriminate

| Signal | Value |
|---|---|
| Not-in-catalogue queries that **passed** the 0.72 gate | **16 of 16** (8 products × English/Arabic) |
| Top-1 score, not-in-catalogue products | 0.76–0.83 (mean 0.784 EN, 0.793 AR) |
| Top-1 score, correct exact-name matches | mean 0.878 (EN) / 0.860 (AR) |
| Top-1 score, Gulf set: correct vs wrong+absent | mean 0.833 vs 0.794 |

E5 cosine scores occupy a narrow high band, so a fixed floor of 0.72 does not separate good from bad retrieval.
Arabic exact-name queries scored about 2% lower on average than English (0.860 vs 0.878).

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

### 4. Agent integration check: the real retriever through the real RAG node

**What happened.** The Phase 2 unit tests mocked the retriever with a bare list (`product_name_en`, `content`). The real
`RetailRetriever.search()` returns a dictionary (`{"results": [...]}`) with `name_en` / `document_text_preview`.
Run against real data, **all 8 smoke-test queries failed** with `'str' object has no attribute 'get'` and fell back, including
`summer dress`. The 59 existing tests were green throughout, because the mock matched the node, not the retriever.

**What changed in `rag_node.py`:**

| Change | Reason |
|---|---|
| Accept both the real dictionary shape and the old list shape | The bug itself |
| Margin computed on **distinct products** (variants collapsed), best product vs next different product | Per-SKU margin was ≈0.005 for right and wrong answers alike |
| `MARGIN_THRESHOLD` 0.05 → **0.01** | 0.05 would flag nearly every query on this catalogue. 0.01 is a hypothesis (section 3 caveat applies) |
| Exact SKU lookup (`FSH-0014` → confidence 1.0, no vector search). Unknown SKU → hard fallback, only when the catalogue is loaded | SKU codes are not in the embedded text |
| Entity check counts 3-letter terms (`تمر`, `دلة`, `oud`) and normalises Arabic hamza/alef/ya/ta marbuta | `عباية` (customer spelling) vs `عباءة` (catalogue) was not matched, so a near-tie between two real abayas became a fallback |
| `search()` runs in a worker thread | It is synchronous and runs the embedding model |

**Smoke test on the final index** (fallback requires margin < 0.01 **and** entity absent, or top-1 < 0.72):

| Query | Top-1 / margin | Fallback | Correct? |
|---|---|---|---|
| `summer dress` | 0.861 / 0.074 | no | yes |
| `فستان صيفي` | 0.874 / 0.055 | no | yes |
| `kandura أبيض` | 0.872 / 0.073 | no | yes |
| `عباية` | 0.792 / 0.003 (two real abayas tie) | no (entity present) | yes, returns both abayas |
| `Nike shoes` (not in catalogue) | 0.760 / 0.000 | **yes** | yes |
| `ايفون 15` (not in catalogue) | 0.785 / 0.001 | **yes** | yes |
| `FSH-0014` | exact lookup, 1.000 | no | yes (Printed T-Shirt) |
| `how many units left of FSH-0014` | exact lookup, 1.000 | no | yes |

**Tests:** ests: 88 pass (59 before this check, plus 17 in tests/test_rag_node_integration.py and 12 in tests/test_agent_real_retriever.py, which use the real retriever). Run against the old node, 8 of the first 15 of the first set fail with the exact error above.

**Caveats:**
- The smoke queries are covered by assertions in tests/test_agent_real_retriever.py, and keep the point that 8 queries is an observation, not a benchmark.
- The node now returns the top 3 **distinct products**, not three variants of one. Weak neighbours appear (`summer dress` returns
  Modest Swimwear Lite and Beach Kaftan Lite; `عباية` returns Storage Baskets third). Decide in Phase 3 whether to filter by score distance or label them as alternatives.
- The margin is top-1 vs the next different product (top-2 at product level). The project's principle text says top-1 vs top-3; the evidence
  here is for the product-level top-2 version, and the threshold would need re-measuring if switched.
- Near-ties between valid products (`عباية`) return results. Fine for search; risky for a reorder question. Routing still sends every fallback straight to the end; the
  partial-answer path for narrative intents is not wired yet.
- Indicative speed only: about 75 queries in 30 s on the dev machine (≈0.4 s each, after a ≈10 s model load), including the Chroma query. Not a controlled benchmark, and Hugging Face free CPU Spaces were not measured.

## Findings and decisions

| Finding | Decision |
|---|---|
| Documents were embedded without the E5 `passage:` prefix (a bug in `prepare_document_text`) | Fixed; regression test added; main index rebuilt |
| Absolute 0.72 floor never fires | Kept as minimum floor only; margin-based routing with variants collapsed added |
| BM25 + RRF is neutral at this scale | Not shipped. Revisit with real SKU/product-name traffic |
| SKU codes are not part of the embedded text, so vector search cannot find them | Exact catalogue lookup for SKU-pattern queries; unknown SKU is a hard fallback |
| Right product is always in the top 3 | On low margin, show the top distinct products with a confidence label (narrative fields only). Not yet wired |
| Mocked retriever hid a real integration bug: every query failed on real data | Integration tests now use the real output shape; keep a real-retriever smoke check in `eval/` |
| Phase 3 demo text uses `SKU-4421`, which is not in this catalogue (IDs look like `FSH-0014`) | Use real SKUs in demos; an unknown SKU correctly returns "insufficient data" |

## Limitations

- Synthetic catalogue with 75 base products; no footwear, phones, TVs or laptops. Results do not transfer to a real, more diverse catalogue.
- 59 queries is a small sample; queries are synthetic and not native-validated.
- Latency (target under 500 ms on e5-large, CPU) is only indicated, not benchmarked (see section 4).
- Not tested: BGE-M3, a cross-encoder reranker, translate-query-to-English, Arabic normalisation of documents (only light normalisation inside BM25 and the entity check).
- Confidence thresholds (including the 0.01 margin) are not validated on held-out data.
- Variant grouping assumes a variant's name is its base name plus extra words, within a category. A real catalogue should have an explicit product-family column.

## Reproduce

```
python -m eval.run_baseline_v2
python -m eval.run_gulf_test
python -m eval.run_rrf_test
python -m eval.smoke_agent
python -m pytest tests -q
```

`PERSIST_DIR` in `run_baseline_v2.py` selects the Chroma index (now `.chroma_db`); the Gulf and RRF scripts import it.

## Scale note

BM25 here is an in-process implementation over 500 documents. At catalogue sizes in the hundreds of thousands, the next step is a proper search engine with sparse + dense retrieval; that is documented as a next architectural step, not implemented.
The product-family map and SKU index are built in memory once at startup; at that scale they belong in the database.