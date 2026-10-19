"""
eval/run_candidate_eval.py
---------------------------
Run the two existing accuracy evals (exact catalogue names EN/AR, and the 59 Gulf / code-switched / spelling
queries) for ANY embedding model, without touching the production index.

    python -m eval.run_candidate_eval intfloat/multilingual-e5-small
    python -m eval.run_candidate_eval intfloat/multilingual-e5-base
    python -m eval.run_candidate_eval intfloat/multilingual-e5-large     # reference, same method

What it does:
  * sets the EMBEDDING_MODEL environment variable BEFORE the backend is imported, so the retriever uses it
  * builds the vector index in its own folder (.chroma_db_eval_<model>), never in .chroma_db, so the
    production index is not replaced or pruned
  * runs run_baseline_v2.main() and run_gulf_test.main() unchanged, and saves their output to
    eval/results_<model>_baseline_v2.json and eval/results_<model>_gulf.json

Run it from the repo root on your machine (not in Docker). The first run per model downloads the model and
embeds the 500-SKU catalogue (a few minutes for e5-large, much less for small).
Requires the one-line change in backend/rag/embeddings.py that reads EMBEDDING_MODEL from the environment.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m eval.run_candidate_eval <huggingface-model-name>")
    model = sys.argv[1]
    slug = model.split("/")[-1]

    os.environ["EMBEDDING_MODEL"] = model          # must happen before backend.rag.embeddings is imported

    from eval import run_baseline_v2 as base       # noqa: E402  (imports the backend, so after the env var)
    from eval import run_gulf_test as gulf         # noqa: E402

    persist = f".chroma_db_eval_{slug}"
    base.PERSIST_DIR = persist
    base.OUT_FILE = f"eval/results_{slug}_baseline_v2.json"
    base.LABEL = f"{slug}_baseline_v2"
    gulf.PERSIST_DIR = persist                      # run_gulf_test copied this name at import time
    gulf.OUT_FILE = f"eval/results_{slug}_gulf.json"
    gulf.LABEL = f"{slug}_gulf"

    print(f"=== model: {model} | index folder: {persist} ===\n")
    base.main()
    print("\n" + "=" * 70 + "\n")
    gulf.main()


if __name__ == "__main__":
    main()