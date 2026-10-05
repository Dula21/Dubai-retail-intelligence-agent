# Bilingual retrieval evaluation (Dubai Retail Intelligence Agent)

Date: 2026-10-05 · Baseline commit: `6fdc697` · Model: `intfloat/multilingual-e5-large` · Vector store: ChromaDB (cosine)

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
| Right product is always in the top 3 | On low margin, show the top distinct products with a confidence label instead of a single answer (narrative fields only) |

## Limitations

- Synthetic catalogue with 75 base products; no footwear, phones, TVs or laptops. Results do not transfer to a real, more diverse catalogue.
- 59 queries is a small sample; queries are synthetic and not native-validated.
- Retrieval only. Latency (target under 500 ms on e5-large, CPU) has **not** been measured.
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