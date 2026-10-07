"""
tests/test_recommendation_node.py
----------------------------------
Recommendation node with a fake Groq client (no network, no key needed).
Guards: the model comes from env, empty/failed LLM replies degrade cleanly, and no raw
provider error reaches the caller or the reasoning trace.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.nodes import recommendation_node as rn  # noqa: E402


def make_state():
    return {
        "query": "how many units left of FSH-0014", "query_language": "en", "intent": "reorder",
        "retrieved_chunks": [], "sales_signals": [], "inventory_signals": [],
        "seasonality_signal": {}, "retrieval_confidence": 0.8,
        "reasoning_trace": ["Agent started"],
    }


class FakeClient:
    def __init__(self, content="MONITOR", exc=None):
        self.kwargs, self._content, self._exc = None, content, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        if self._exc:
            raise self._exc
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def run(monkeypatch, client):
    monkeypatch.setattr(rn, "get_groq_client", lambda: client)
    return asyncio.run(rn.recommendation_node(make_state()))


def test_success_uses_model_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some/test-model")
    client = FakeClient(content="Stock is fine.\nMONITOR")
    out = run(monkeypatch, client)
    assert client.kwargs["model"] == "some/test-model"
    assert out["recommendation"].endswith("MONITOR")
    assert "some/test-model" in out["reasoning_trace"][-1]
    assert "error" not in out


def test_provider_error_never_reaches_caller(monkeypatch):
    client = FakeClient(exc=RuntimeError("Error code: 404 model_not_found secret-detail"))
    out = run(monkeypatch, client)
    assert "secret-detail" not in out["recommendation"]
    assert "secret-detail" not in " ".join(out["reasoning_trace"])
    assert out["recommendation_confidence"] == 0.0
    assert out["error"]                      # set, so the API does not cache it


def test_empty_model_reply_is_a_degraded_answer_not_a_blank(monkeypatch):
    out = run(monkeypatch, FakeClient(content=""))
    assert out["recommendation"].strip()
    assert out["error"]