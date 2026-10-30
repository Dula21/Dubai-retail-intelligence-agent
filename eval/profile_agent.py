"""
eval/profile_agent.py
---------------------
Where do the agent's 10-20 seconds go? Runs the agent directly (no HTTP, no Redis cache) on several
event-planning questions and prints the time spent in each node.

Run inside the API container (it needs the data files, the models and GROQ_API_KEY already set there):

    docker compose cp eval/profile_agent.py api:/app/profile_agent.py
    docker compose exec api python profile_agent.py            # 6 questions, 4 s apart
    docker compose exec api python profile_agent.py --n 3 --pause 6

The first question is reported separately (WARM-UP): it pays the one-off cost of loading the sales CSV
into memory, which later questions do not. Percentiles use nearest rank. With n this small, read the
medians and the largest-node ranking, not the p95.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import time

QUESTIONS = [
    "what should I stock more of before Ramadan",
    "what should I stock more of before Eid al-Fitr",
    "what should I stock more of before Dubai Shopping Festival",
    "what should I stock more of before National Day",
    "what should I stock more of before Back to School",
    "what should I stock more of before Diwali",
    "what should I stock more of before White Friday",
    "what should I stock more of before Eid al-Adha",
]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


async def main(n: int, pause: float) -> None:
    from backend.agents.retail_agent import run_agent

    rows: list[dict] = []
    for i, question in enumerate(QUESTIONS[:n]):
        start = time.perf_counter()
        state = await run_agent(question, "en", "event_planning")
        total_ms = (time.perf_counter() - start) * 1000
        timings = dict(state.get("node_timings_ms") or {})
        accounted = sum(timings.values())
        rows.append({"q": question, "total": total_ms, "nodes": timings, "other": total_ms - accounted,
                     "chunks": len(state.get("retrieved_chunks", [])),
                     "error": state.get("error")})
        short = ", ".join(f"{k.replace('_node', '')}={v:.0f}" for k, v in timings.items())
        flag = f" | ERROR: {state['error']}" if state.get("error") else ""
        print(f"[{i + 1}/{n}] {total_ms:8.0f} ms | {rows[-1]['chunks']} SKUs | {short}{flag}")
        if i < n - 1:
            await asyncio.sleep(pause)  # stay under the Groq free-tier rate limit

    if not rows:
        return
    warm, rest = rows[0], rows[1:]
    print("\nWARM-UP (question 1, includes one-off data loading):")
    print(f"  total {warm['total']:.0f} ms | " + ", ".join(f"{k}={v:.0f}" for k, v in warm["nodes"].items()))

    if not rest:
        return
    names = sorted({k for r in rest for k in r["nodes"]})
    print(f"\nQUESTIONS 2..{n} (n={len(rest)}), milliseconds:")
    print(f"{'node':28}{'p50':>9}{'max':>9}{'share of total':>16}")
    total_p50 = statistics.median(r["total"] for r in rest)
    for name in names:
        vals = [r["nodes"][name] for r in rest if name in r["nodes"]]
        print(f"{name:28}{statistics.median(vals):9.0f}{max(vals):9.0f}"
              f"{statistics.median(vals) / total_p50:15.0%}")
    other = [r["other"] for r in rest]
    print(f"{'(outside nodes)':28}{statistics.median(other):9.0f}{max(other):9.0f}"
          f"{statistics.median(other) / total_p50:15.0%}")
    print(f"{'TOTAL':28}{total_p50:9.0f}{max(r['total'] for r in rest):9.0f}")
    errors = [r for r in rest if r["error"]]
    if errors:
        print(f"\n{len(errors)} of {len(rest)} questions ended with an error/degraded answer; "
              "their timings may not represent a normal run (for example a Groq rate limit).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pause", type=float, default=4.0)
    args = ap.parse_args()
    asyncio.run(main(args.n, args.pause))