"""Baseline v2: ground truth is built FROM your catalogue (no guessed regexes).

Run from the repo root:   python -m eval.run_baseline_v2

Idea: your catalogue repeats 15 base products per category ("Summer Dress", "Summer Dress Pro", ...).
Each base product's English name and Arabic name become the English / Arabic query, and every SKU
that belongs to that base product counts as correct. Also tests queries for products that are NOT
in the catalogue (the system should be unsure about those).
The confidence gate is switched OFF so raw scores are visible.
"""
import json
from collections import defaultdict
from pathlib import Path

from backend.rag.retriever import RetailRetriever

# ---------- CONFIG: copy CATALOGUE_CSV from your run_baseline.py ----------
CATALOGUE_CSV = "data/synthetic/product_catalogue.csv"
PERSIST_DIR = ".chroma_db"
OUT_FILE = "eval/baseline_v2_results.json"
LABEL = "baseline_6fdc697_v2"      # change per experiment, e.g. "after_passage_prefix_fix"
K = 5
GATE = 0.72
CYCLE = 15                          # base products per category (seen in the SKU numbering)
# ---------------------------------------------------------------------------

# Products that are clearly NOT in this catalogue. A good system should be unsure about these.
ABSENT = [
    ("sports shoes", "جزمة رياضية"), ("Nike shoes", "حذاء نايكي"), ("iPhone 15", "ايفون 15"),
    ("Samsung TV", "تلفزيون سامسونج"), ("football boots", "حذاء كرة قدم"),
    ("electric guitar", "جيتار كهربائي"), ("car tyres", "إطارات سيارة"), ("PlayStation 5", "بلايستيشن 5"),
]


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def build_groups(catalogue):
    """Return (bases, groups): bases = [(name_en, name_ar, category)], groups = {(category, name_en): {sku_ids}}."""
    by_cat = defaultdict(list)
    for row in catalogue:
        by_cat[row["category"]].append(row)
    bases = [(r["name_en"], r["name_ar"], cat) for cat, rows in by_cat.items() for r in rows[:CYCLE]]
    groups = {(b[2], b[0]): set() for b in bases}
    for row in catalogue:
        cands = [b for b in bases if b[2] == row["category"] and row["name_en"].startswith(b[0])]
        if cands:
            best = max(cands, key=lambda b: len(b[0]))   # longest matching base name wins
            groups[(best[2], best[0])].add(row["sku_id"])
    return bases, groups


def margin(hits):
    return round(hits[0][1] - hits[2][1], 4) if len(hits) >= 3 else None


def main() -> None:
    rt = RetailRetriever(CATALOGUE_CSV, None, PERSIST_DIR)
    info = rt.initialize()   # does NOT re-embed if the collection already exists
    model = rt.embedding_mgr.config.model_name
    print(f"[{LABEL}] {info} | model: {model}")

    def run(text):
        res = rt.vector_store.query(text, n_results=K, min_confidence=-1.0)   # gate OFF
        return [(r["sku_id"], r["similarity_score"], r["name_en"]) for r in res]

    catalogue = rt.loader.load_catalogue()
    bases, groups = build_groups(catalogue)
    sizes = sorted({len(v) for v in groups.values()})
    print(f"base products: {len(bases)} | SKUs per base product: {sizes} | total grouped: {sum(len(v) for v in groups.values())}")

    rows = []
    for name_en, name_ar, cat in bases:
        rel = groups[(cat, name_en)]
        rows.append({"category": cat, "en": name_en, "ar": name_ar, "relevant": sorted(rel),
                     "hits": {"en": run(name_en), "ar": run(name_ar)}})

    def stats(rs, v):
        h1, h5, p5 = [], [], []
        for r in rs:
            h, rel = r["hits"][v], set(r["relevant"])
            h1.append(float(bool(h) and h[0][0] in rel))
            h5.append(float(any(s in rel for s, _, _ in h)))
            p5.append(sum(s in rel for s, _, _ in h) / K)
        return {"hit@1": mean(h1), "hit@5": mean(h5), "precision@5": mean(p5)}

    print("\n=== 1. ACCURACY: catalogue name used as the query (EN vs AR) ===")
    for v in ("en", "ar"):
        print(f"{v}: {stats(rows, v)}")
    print("by category (hit@1 en / ar):")
    for cat in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == cat]
        print(f"  {cat:<12} n={len(rs):<3} en={stats(rs, 'en')['hit@1']}  ar={stats(rs, 'ar')['hit@1']}")

    print(f"\n=== 2. IS THE {GATE} GATE USEFUL? (present products) ===")
    for v in ("en", "ar"):
        ok, bad, m_ok, m_bad = [], [], [], []
        for r in rows:
            h = r["hits"][v]
            if not h:
                continue
            good = h[0][0] in set(r["relevant"])
            (ok if good else bad).append(h[0][1])
            m = margin(h)
            if m is not None:
                (m_ok if good else m_bad).append(m)
        print(f"{v}: correct top-1 n={len(ok)} mean score={mean(ok)} below gate={mean([float(s < GATE) for s in ok])}"
              f" | wrong top-1 n={len(bad)} mean score={mean(bad)} pass gate={mean([float(s >= GATE) for s in bad])}"
              f" | margin correct={mean(m_ok)} wrong={mean(m_bad)}")

    print("\n=== 3. PRODUCTS NOT IN THE CATALOGUE (should be unsure) ===")
    absent_rows = []
    for en, ar in ABSENT:
        absent_rows.append({"en": en, "ar": ar, "hits": {"en": run(en), "ar": run(ar)}})
    for v in ("en", "ar"):
        tops = [r["hits"][v][0][1] for r in absent_rows]
        ms = [margin(r["hits"][v]) for r in absent_rows]
        print(f"{v}: top-1 score mean={mean(tops)} min={min(tops)} max={max(tops)} | pass gate={mean([float(s >= GATE) for s in tops])}"
              f" | margin mean={mean([m for m in ms if m is not None])}")
    print("(compare with correct top-1 scores in section 2: if they overlap, no score threshold can separate them)")

    print("\n=== 4. ARABIC MISSES (top-1 not the right product), max 15 ===")
    shown = 0
    for r in rows:
        h = r["hits"]["ar"]
        if h and h[0][0] not in set(r["relevant"]) and shown < 15:
            print(f"  '{r['ar']}' (want: {r['en']}) -> got {h[0][0]} {h[0][2]!r} score {h[0][1]}")
            shown += 1

    Path(OUT_FILE).parent.mkdir(exist_ok=True)
    Path(OUT_FILE).write_text(json.dumps({"label": LABEL, "model": model, "rows": rows, "absent": absent_rows},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()