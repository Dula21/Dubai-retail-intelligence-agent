"""
eval/profile_embedding.py
--------------------------
Where does a /search request spend its time? The /search path is: embed the query (multilingual-e5-large
on CPU) -> ChromaDB lookup over 500 documents -> attach cached sales summary. This script times the
embedding step alone, with the same model, the same "query: " prefix and the same encode() call the
retriever uses. Pass a model name to time a candidate model the same way.

Run it INSIDE the API container (same CPU, same Python 3.12, same torch build as the measured API):

    docker compose cp eval/profile_embedding.py api:/app/profile_embedding.py
    docker compose exec api python profile_embedding.py                                   # the current model
    docker compose exec api python profile_embedding.py intfloat/multilingual-e5-base     # a candidate
    docker compose exec api python profile_embedding.py intfloat/multilingual-e5-small    # a candidate

A candidate is downloaded into the container's .model_cache the first time (one-off, a few hundred MB).
This only measures SPEED. Accuracy has to be re-measured with the eval scripts before any model is switched.

It loads a second copy of the model (about 2 GB RAM), so make sure Docker Desktop has memory to spare.
"""

from __future__ import annotations

import os
import sys
import time

import torch
from sentence_transformers import SentenceTransformer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "intfloat/multilingual-e5-large"
QUERIES = [
    "summer dress under AED 200", "فستان صيفي تحت 200 درهم", "winter coat under AED 300",
    "معطف شتوي تحت 300 درهم", "bluetooth speaker under AED 100", "سماعة بلوتوث تحت 100 درهم",
    "perfume under AED 150", "عطر تحت 150 درهم", "sneakers under AED 250", "حذاء رياضي تحت 250 درهم",
    "laptop bag under AED 100", "حقيبة لابتوب تحت 100 درهم", "scented candle under AED 50",
    "شمعة معطرة تحت 50 درهم", "dates gift box under AED 150", "علبة تمور تحت 150 درهم",
    "phone case under AED 50", "جراب هاتف تحت 50 درهم", "face cream under AED 200", "كريم وجه تحت 200 درهم",
]


def pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    k = max(1, int(round(p / 100.0 * len(s) + 0.4999999)))
    return s[min(k, len(s)) - 1]


def main() -> None:
    print(f"cpu_count={os.cpu_count()}  torch={torch.__version__}  default torch threads={torch.get_num_threads()}")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL, cache_folder=".model_cache", device="cpu")
    print(f"model load: {time.perf_counter() - t0:.1f} s (one-off, not per request)")

    model.encode(["query: warm up"], normalize_embeddings=True)   # first call is slower, not counted

    # Default thread count only: on this 4-CPU Docker setup, 1 and 4 threads were slower and noisier than 2.
    print(f"\nmodel: {MODEL}  (threads={torch.get_num_threads()})")
    ms = []
    for _round in range(2):                       # 2 passes over 20 queries = 40 samples
        for q in QUERIES:
            t = time.perf_counter()
            model.encode([f"query: {q}"], normalize_embeddings=True)
            ms.append((time.perf_counter() - t) * 1000.0)
    print(f"{'n':>4}{'p50 ms':>10}{'p95 ms':>10}{'min':>9}{'max':>9}")
    print(f"{len(ms):>4}{pct(ms, 50):>10.1f}{pct(ms, 95):>10.1f}{min(ms):>9.1f}{max(ms):>9.1f}")
    print(f"embedding dimension: {model.get_sentence_embedding_dimension()}")


if __name__ == "__main__":
    main()