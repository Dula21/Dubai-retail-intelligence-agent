"""Baseline parallel-query test against the REAL retriever (ChromaDB + the model in embeddings.py).

Run from the repo root:   python -m eval.run_baseline
Reads the 50 queries from eval/parallel_query_test.py (no faiss needed).
The confidence gate is switched OFF for this test so we can see raw scores.
"""
import ast
import json
import re
from pathlib import Path

from backend.rag.retriever import RetailRetriever

# ---------- CONFIG: set the two paths, leave the rest ----------
CATALOGUE_CSV = "data/synthetic/product_catalogue.csv"   # <- your real catalogue CSV path
SALES_CSV = None                                  # not needed for this test
PERSIST_DIR = ".chroma_db"
QUERY_FILE = "eval/parallel_query_test.py"
OUT_FILE = "eval/baseline_results.json"
LABEL = "baseline_6fdc697"                        # change per experiment, e.g. "after_passage_prefix_fix"
K = 5
GATE = 0.72
# ---------------------------------------------------------------


def load_queries(path: str) -> list:
    """Pull QUERIES_RAW out of the old script without importing it (it needs faiss)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "QUERIES_RAW" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"QUERIES_RAW not found in {path}")


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def main() -> None:
    retriever = RetailRetriever(CATALOGUE_CSV, SALES_CSV, PERSIST_DIR)
    info = retriever.initialize()  # does NOT re-embed if the collection already exists
    model_name = retriever.embedding_mgr.config.model_name
    print(f"[{LABEL}] retriever: {info} | model: {model_name}")

    catalogue = retriever.loader.load_catalogue()
    queries = load_queries(QUERY_FILE)

    rows = []
    for i, (intent, en, ar, cs, pattern) in enumerate(queries, 1):
        relevant = {c["sku_id"] for c in catalogue if re.search(pattern, c["name_en"], re.I)}
        hits = {}
        for variant, text in (("en", en), ("ar", ar), ("cs", cs)):
            res = retriever.vector_store.query(text, n_results=K, min_confidence=-1.0)  # gate OFF
            hits[variant] = [(r["sku_id"], r["similarity_score"]) for r in res]
        rows.append({"id": f"q{i:02d}", "intent": intent, "queries": {"en": en, "ar": ar, "cs": cs},
                     "relevant": sorted(relevant), "hits": hits})

    scored = [r for r in rows if r["relevant"]]
    groups = {"all": rows, "name": [r for r in rows if r["intent"] == "name"],
              "desc": [r for r in rows if r["intent"] == "desc"]}
    scored_groups = {g: [r for r in rs if r["relevant"]] for g, rs in groups.items()}

    print("\n=== 0. GROUND-TRUTH SANITY (regex matches on name_en) ===")
    print("no matching SKU in catalogue:", [r["id"] for r in rows if not r["relevant"]])
    print("very loose (>40 matches):   ", [(r["id"], len(r["relevant"])) for r in rows if len(r["relevant"]) > 40])

    print("\n=== 1. TOP-5 OVERLAP vs the English query ===")
    for v in ("ar", "cs"):
        out = {g: mean([len({s for s, _ in r["hits"][v]} & {s for s, _ in r["hits"]["en"]}) / K for r in rs])
               for g, rs in groups.items()}
        print(f"{v:>2} vs en: {out}")

    print("\n=== 2. HIT@5 (any relevant SKU in top 5) ===")
    for v in ("en", "ar", "cs"):
        out = {g: mean([1.0 if {s for s, _ in r["hits"][v]} & set(r["relevant"]) else 0.0 for r in rs])
               for g, rs in scored_groups.items()}
        print(f"{v:>2} query: {out}")

    print(f"\n=== 3. HOW USEFUL IS THE {GATE} GATE? (top-1 correct vs wrong) ===")
    for v in ("en", "ar", "cs"):
        correct, wrong, m_c, m_w = [], [], [], []
        for r in scored:
            h = r["hits"][v]
            if not h:
                continue
            margin = round(h[0][1] - h[2][1], 4) if len(h) >= 3 else None
            if h[0][0] in r["relevant"]:
                correct.append(h[0][1])
                if margin is not None:
                    m_c.append(margin)
            else:
                wrong.append(h[0][1])
                if margin is not None:
                    m_w.append(margin)
        print(f"{v:>2}: correct top-1 n={len(correct)} mean score={mean(correct)} below gate={mean([float(s < GATE) for s in correct])}"
              f" | wrong top-1 n={len(wrong)} mean score={mean(wrong)} PASS the gate={mean([float(s >= GATE) for s in wrong])}"
              f" | mean margin top1-top3: correct={mean(m_c)} wrong={mean(m_w)}")

    Path(OUT_FILE).parent.mkdir(exist_ok=True)
    Path(OUT_FILE).write_text(json.dumps({"label": LABEL, "model": model_name, "rows": rows},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()