"""
agents/timing.py
----------------
Per-node wall-clock timing for the LangGraph agent.

Why this exists: end-to-end agent latency was measured at p50 14.5 s / p95 20.6 s (uncached), but the
embedding step is only ~35 ms of that. Before changing anything (model, prompt, parallelism) we need to
know WHICH node the time goes to. This wrapper measures each node without touching node code.

How it works: each node is wrapped at registration time. The wrapper runs the node, then adds
`{node_name: elapsed_ms}` to the state key `node_timings_ms`. Nodes that run twice would overwrite
themselves, which cannot happen in this graph (no loops); if loops are added, store a list instead.

The wrapper never changes what a node returns except for adding that one key, and it re-raises any
exception unchanged (the elapsed time is still logged).
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

NodeFn = Callable[[dict], Awaitable[dict]]


def timed_node(name: str, fn: NodeFn) -> NodeFn:
    """Wrap an async LangGraph node so its wall-clock time is recorded in state['node_timings_ms']."""

    async def wrapper(state: dict) -> dict:
        start = time.perf_counter()
        try:
            result = await fn(state)
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info("node_timing", node=name, elapsed_ms=elapsed_ms)

        result = dict(result) if result else {}
        timings = dict(state.get("node_timings_ms") or {})
        timings[name] = elapsed_ms
        result["node_timings_ms"] = timings
        return result

    wrapper.__name__ = f"timed_{name}"
    return wrapper