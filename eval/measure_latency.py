"""
eval/measure_latency.py
------------------------
Single-client latency measurement for the Dubai Retail Intelligence Agent API.
Standard library only (no extra installs).

What it measures (client-side round trip, in milliseconds, one request at a time):
  1. search_repeat   POST /api/v1/search, the SAME query repeated (model and any cache are warm)
  2. search_distinct POST /api/v1/search, a DIFFERENT query every time (cold embedding each time).
                     This is the case the "under 500 ms for catalogue queries" target applies to.
  3. agent_cold      POST /api/v1/agent/query, distinct queries (calls Groq; free tier is rate limited,
                     so few requests with a pause between them)
  4. agent_cached    POST /api/v1/agent/query, the same query repeated (Redis hit)

For search it also reports the server-side `latency_ms` field from the response, so you can separate
network + FastAPI overhead from retrieval time.

Run (API must be up, with the embedding model already loaded):
    python eval/measure_latency.py
    python eval/measure_latency.py --skip-agent          # search only, no Groq calls
    python eval/measure_latency.py --base-url http://localhost:8000

For a clean agent_cold number, flush Redis first:
    docker compose exec redis redis-cli FLUSHDB

Results are printed and saved to eval/latency_results.json.
Limits (stated in the output file): one client, one machine, Docker on CPU, synthetic catalogue of 500 SKUs.
p99 over ~50 samples is effectively the slowest request; treat p50 and p95 as the headline numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

TARGET_MS = 500          # project target for catalogue queries (cold embedding path, p95)

# Distinct English and Arabic catalogue-style queries. Built from parts so there are 50 different strings.
EN_PRODUCTS = ["summer dress", "winter coat", "bluetooth speaker", "perfume", "sneakers",
               "laptop bag", "scented candle", "dates gift box", "phone case", "face cream"]
AR_PRODUCTS = ["فستان صيفي", "معطف شتوي", "سماعة بلوتوث", "عطر", "حذاء رياضي",
               "حقيبة لابتوب", "شمعة معطرة", "علبة تمور", "جراب هاتف", "كريم وجه"]
PRICES = [50, 100, 150, 200, 300]


def distinct_search_queries(n: int) -> list[str]:
    queries: list[str] = []
    i = 0
    while len(queries) < n:
        k = i // 2                                   # alternate English / Arabic
        product_i = k % len(EN_PRODUCTS)
        price = PRICES[(k // len(EN_PRODUCTS)) % len(PRICES)]   # 10 products x 5 prices = 50 per language
        if i % 2 == 0:
            queries.append(f"{EN_PRODUCTS[product_i]} under AED {price}")
        else:
            queries.append(f"{AR_PRODUCTS[product_i]} تحت {price} درهم")
        i += 1
    return queries


AGENT_QUERIES = [
    "what should I stock more of before Ramadan",
    "what should I stock more of before Diwali",
    "what should I stock before National Day",
    "what should I stock before Christmas",
    "what should I stock before White Friday",
    "what should I stock before Back to School",
    "what should I stock before Eid al-Adha",
    "what should I stock before Valentine's Day",
]


def post(url: str, body: dict, timeout: float = 120.0) -> tuple[float, int, dict | None]:
    data = json.dumps(body).encode("utf-8")   # ensure_ascii default: safe for Arabic
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        payload, status = e.read(), e.code
    except Exception as e:  # connection refused, timeout
        print(f"  request failed: {e}", file=sys.stderr)
        return (time.perf_counter() - t0) * 1000.0, 0, None
    ms = (time.perf_counter() - t0) * 1000.0
    try:
        return ms, status, json.loads(payload)
    except ValueError:
        return ms, status, None


def pct(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    if not sorted_vals:
        return float("nan")
    k = max(1, int(round(p / 100.0 * len(sorted_vals) + 0.4999999)))
    return sorted_vals[min(k, len(sorted_vals)) - 1]


def summarise(name: str, ms: list[float], errors: int, server_ms: list[float] | None = None) -> dict:
    s = sorted(ms)
    out = {
        "case": name, "ok": len(ms), "errors": errors,
        "p50_ms": round(pct(s, 50), 1) if s else None,
        "p95_ms": round(pct(s, 95), 1) if s else None,
        "p99_ms": round(pct(s, 99), 1) if s else None,
        "min_ms": round(s[0], 1) if s else None,
        "max_ms": round(s[-1], 1) if s else None,
        "mean_ms": round(statistics.fmean(s), 1) if s else None,
    }
    if server_ms:
        ss = sorted(server_ms)
        out["server_p50_ms"] = round(pct(ss, 50), 1)
        out["server_p95_ms"] = round(pct(ss, 95), 1)
    return out


def run_search(base: str, queries: list[str]) -> tuple[list[float], list[float], int]:
    ms, server, errors = [], [], 0
    for q in queries:
        t, status, body = post(f"{base}/api/v1/search", {"query": q, "n_results": 5})
        if status == 200 and body is not None:
            ms.append(t)
            if isinstance(body.get("latency_ms"), (int, float)):
                server.append(float(body["latency_ms"]))
        else:
            errors += 1
    return ms, server, errors


def run_agent(base: str, queries: list[str], pause_s: float) -> tuple[list[float], int, bool]:
    ms, errors, rate_limited = [], 0, False
    for q in queries:
        t, status, body = post(f"{base}/api/v1/agent/query", {"query": q})
        if status == 200 and body is not None:
            ms.append(t)
        else:
            errors += 1
            if status == 429:
                rate_limited = True
                print("  got HTTP 429 (rate limit); stopping this case", file=sys.stderr)
                break
        time.sleep(pause_s)
    return ms, errors, rate_limited


def print_table(rows: list[dict]) -> None:
    head = f"{'case':<18}{'n':>4}{'err':>5}{'p50':>9}{'p95':>9}{'p99':>9}{'min':>9}{'max':>9}"
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        if r["p50_ms"] is None:
            print(f"{r['case']:<18}{r['ok']:>4}{r['errors']:>5}   no successful requests")
            continue
        print(f"{r['case']:<18}{r['ok']:>4}{r['errors']:>5}{r['p50_ms']:>9}{r['p95_ms']:>9}"
              f"{r['p99_ms']:>9}{r['min_ms']:>9}{r['max_ms']:>9}")
        if "server_p50_ms" in r:
            print(f"{'  (server latency_ms)':<18}{'':>9}{r['server_p50_ms']:>9}{r['server_p95_ms']:>9}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=50, help="requests per search case")
    ap.add_argument("--agent-n", type=int, default=8, help="distinct agent queries (cold case)")
    ap.add_argument("--agent-pause", type=float, default=4.0, help="seconds between agent calls (Groq free tier)")
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_results.json"))
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    try:
        urllib.request.urlopen(f"{base}/health", timeout=10).read()
    except Exception as e:
        print(f"API not reachable at {base}: {e}\nStart it with: docker compose up -d", file=sys.stderr)
        return 1

    rows: list[dict] = []

    print("Warm-up (3 search requests, not counted)...")
    run_search(base, distinct_search_queries(3))

    print(f"search_repeat: same query x{args.n}")
    repeat_q = "summer dress under AED 200"
    run_search(base, [repeat_q])                       # first call primes any cache, not counted
    ms, server, err = run_search(base, [repeat_q] * args.n)
    rows.append(summarise("search_repeat", ms, err, server))

    print(f"search_distinct: {args.n} different queries (EN and AR)")
    queries = distinct_search_queries(args.n + 3)[3:]  # skip the 3 used for warm-up
    ms, server, err = run_search(base, queries)
    rows.append(summarise("search_distinct", ms, err, server))

    rate_limited = False
    if not args.skip_agent:
        print(f"agent_cold: {args.agent_n} distinct queries, {args.agent_pause}s apart")
        aq = AGENT_QUERIES[: args.agent_n]
        ms, err, rate_limited = run_agent(base, aq, args.agent_pause)
        rows.append(summarise("agent_cold", ms, err))
        if not rate_limited:
            print("agent_cached: same query x20 (Redis hit)")
            ms, err, _ = run_agent(base, [aq[0]] * 20, 0.0)
            rows.append(summarise("agent_cached", ms, err))

    print_table(rows)

    sd = next((r for r in rows if r["case"] == "search_distinct"), None)
    verdict = None
    if sd and sd["p95_ms"] is not None:
        verdict = "PASS" if sd["p95_ms"] < TARGET_MS else "FAIL"
        print(f"\nTarget: catalogue query p95 < {TARGET_MS} ms (cold embedding path) -> "
              f"p95 = {sd['p95_ms']} ms -> {verdict}")

    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": base,
        "environment": {"os": platform.platform(), "cpu_count": os.cpu_count(),
                        "python": platform.python_version(), "runs_in": "Docker on CPU (no GPU)"},
        "method": "single client, sequential requests, client-side round trip in ms",
        "target_ms_catalogue_p95": TARGET_MS,
        "target_verdict": verdict,
        "agent_rate_limited": rate_limited,
        "results": rows,
        "limits": ["one client, no concurrency (see load test for that)",
                   "synthetic catalogue of 500 SKUs",
                   "p99 over ~50 samples is effectively the slowest request",
                   "agent_cold depends on Groq free-tier speed and rate limits"],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())