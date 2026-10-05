"""Smoke test: the REAL RetailRetriever feeding the REAL rag_retrieval_node (no mocks).

Run from the repo root:   python -m eval.smoke_agent

Why: the unit tests replace the retriever with a mock that returns a list in the shape the node
expects. This script checks whether the real retriever's output actually works with the node.
It changes nothing; it only prints.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path("backend").resolve()))   # the tests import agents.* / services.* this way

from backend.rag.retriever import RetailRetriever
from eval.run_baseline_v2 import CATALOGUE_CSV, PERSIST_DIR
from agents.nodes import rag_node

sys.stdout.reconfigure(encoding="utf-8")

CASES = [
    ("search", "summer dress"),
    ("search", "فستان صيفي"),
    ("search", "عباية"),
    ("search", "kandura أبيض"),
    ("search", "Nike shoes"),          # not in the catalogue: should be unsure
    ("search", "ايفون 15"),            # not in the catalogue: should be unsure
    ("search", "FSH-0014"),            # an exact SKU
    ("reorder", "how many units left of FSH-0014"),
]


async def main() -> None:
    rt = RetailRetriever(CATALOGUE_CSV, None, PERSIST_DIR)
    rt.initialize()
    rag_node.set_retriever(rt)

    sample = rt.search("summer dress", n_results=3)
    print("retriever.search() returns a:", type(sample).__name__)
    if isinstance(sample, dict):
        print("  keys:", list(sample.keys()))
        first = sample["results"][0] if sample.get("results") else {}
        print("  first result keys:", list(first.keys()))

    for intent, q in CASES:
        state = {"query": q, "intent": intent, "reasoning_trace": []}
        out = await rag_node.rag_retrieval_node(state)
        names = [c.get("product_name_en") or "<empty name>" for c in out["retrieved_chunks"]]
        print(f"\n[{intent}] {q}")
        print(f"  fallback={out['fallback_triggered']} confidence={out['retrieval_confidence']:.3f} "
              f"margin={out['retrieval_margin']:.3f} chunks={len(names)} names={names}")
        if out.get("error"):
            print("  ERROR:", out["error"])
        print("  trace:", out["reasoning_trace"][-1])


if __name__ == "__main__":
    asyncio.run(main())