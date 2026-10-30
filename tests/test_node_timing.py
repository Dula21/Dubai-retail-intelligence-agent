"""Tests for the per-node timing wrapper (backend/agents/timing.py)."""

import asyncio

import pytest

from backend.agents.timing import timed_node


def run(coro):
    return asyncio.run(coro)


def test_adds_elapsed_ms_for_the_node_and_keeps_the_node_output():
    async def node(state):
        await asyncio.sleep(0.05)
        return {"recommendation": "ok"}

    out = run(timed_node("slow_node", node)({"node_timings_ms": {}}))

    assert out["recommendation"] == "ok"
    assert out["node_timings_ms"]["slow_node"] >= 45


def test_keeps_timings_from_earlier_nodes():
    async def node(state):
        return {}

    out = run(timed_node("second", node)({"node_timings_ms": {"first": 12.0}}))

    assert out["node_timings_ms"]["first"] == 12.0
    assert "second" in out["node_timings_ms"]


def test_works_when_state_has_no_timings_key_yet():
    async def node(state):
        return {"x": 1}

    out = run(timed_node("only", node)({}))

    assert set(out["node_timings_ms"]) == {"only"}


def test_exception_from_the_node_is_not_swallowed():
    async def node(state):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run(timed_node("bad", node)({}))