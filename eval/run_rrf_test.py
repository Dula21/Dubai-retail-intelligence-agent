"""Experiment 2: dense vs BM25 vs dense+BM25 fused with Reciprocal Rank Fusion (RRF).

Run from the repo root:   python -m eval.run_rrf_test
Uses the same 59 queries as eval/run_gulf_test.py and the index folder set by PERSIST_DIR in
eval/run_baseline_v2.py (keep it on the exp1 index). No new packages: BM25 is implemented below.

RRF merges RANKS, not scores, so the dense/BM25 score-scale mismatch does not matter.
Results are collapsed to base products (variants of the same product count once).
"""
import math
import re
import sys
from collections import Counter

from backend.rag.retriever import RetailRetriever
from eval.run_baseline_v2 import CATALOGUE_CSV, PERSIST_DIR, build_groups, mean
from eval.run_gulf_test import QUERIES

sys.stdout.reconfigure(encoding="utf-8")

N = 50        # SKU candidates taken from each retriever
RRF_K = 60    # standard RRF constant

_DIAC = re.compile(r"[\u064B-\u0652\u0640]")   # Arabic diacritics + tatweel


def norm(text: str) -> str:
    """Light Arabic normalisation + lowercase, applied to documents AND queries for BM25."""
    t = _DIAC.sub("", text.lower())
    t = re.sub("[أإآ]", "ا", t)
    return t.replace("ى", "ي").replace("ة", "ه")


def tokens(text: str) -> list:
    return re.findall(r"[a-z0-9\u0621-\u064A]+", norm(text))


class BM25:
    """Minimal Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(len(d) for d in docs) / len(docs)
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens):
        out = [0.0] * len(self.docs)
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    norm_len = 1 - self.b + self.b * len(self.docs[i]) / self.avgdl
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm_len)
        return out


def fuse(rank_lists):
    s = {}
    for lst in rank_lists:
        for r, sku in enumerate(lst, 1):
            s[sku] = s.get(sku, 0.0) + 1.0 / (RRF_K + r)
    return sorted(s, key=s.get, reverse=True)


def main() -> None:
    rt = RetailRetriever(CATALOGUE_CSV, None, PERSIST_DIR)
    rt.initialize()
    catalogue = rt.loader.load_catalogue()
    bases, groups = build_groups(catalogue)
    sku2base = {sku: key[1] for key, skus in groups.items() for sku in skus}
    skus = [c["sku_id"] for c in catalogue]
    bm25 = BM25([tokens(" ".join(str(c.get(f, "")) for f in ("name_en", "name_ar", "tags_en", "tags_ar", "category")))
                 for c in catalogue])

    def dense_list(q):
        return [r["sku_id"] for r in rt.vector_store.query(q, n_results=N, min_confidence=-1.0)]

    def bm25_list(q):
        sc = bm25.scores(tokens(q))
        order = sorted(range(len(skus)), key=lambda i: sc[i], reverse=True)
        return [skus[i] for i in order[:N] if sc[i] > 0]

    def bases_of(sku_list):
        seen = []
        for s in sku_list:
            b = sku2base.get(s)
            if b and b not in seen:
                seen.append(b)
        return seen

    systems = {"dense": [], "bm25": [], "rrf": []}
    detail = []
    for kind, q, targets in QUERIES:
        d, b = dense_list(q), bm25_list(q)
        ranks = {"dense": bases_of(d), "bm25": bases_of(b), "rrf": bases_of(fuse([d, b]))}
        row = {"kind": kind, "q": q, "targets": targets}
        for name, rk in ranks.items():
            row[name] = (bool(rk) and rk[0] in targets, any(x in targets for x in rk[:3]), rk[:3])
        detail.append(row)

    print(f"queries={len(detail)} | dense index folder: {PERSIST_DIR}")
    print("\n=== ACCURACY (top-1 product / any of top-3 products) ===")
    for kind in ("gulf", "cs", "spell", None):
        rs = [r for r in detail if kind is None or r["kind"] == kind]
        line = f"{(kind or 'ALL'):<6} n={len(rs):<3}"
        for name in ("dense", "bm25", "rrf"):
            line += f" | {name}: hit@1={mean([float(r[name][0]) for r in rs])} hit@3={mean([float(r[name][1]) for r in rs])}"
        print(line)

    print("\n=== QUERIES WHERE RRF CHANGED THE TOP-1 RESULT vs dense alone ===")
    for r in detail:
        if r["dense"][0] != r["rrf"][0]:
            print(f"  {'FIXED ' if r['rrf'][0] else 'BROKE '} [{r['kind']}] {r['q']}  want {r['targets']}  rrf top-1: {r['rrf'][2][:1]}")

    print("\n=== STILL WRONG WITH RRF ===")
    for r in detail:
        if not r["rrf"][0]:
            print(f"  [{r['kind']}] {r['q']}  want {r['targets']}  got {r['rrf'][2]}")


if __name__ == "__main__":
    main()