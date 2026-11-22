"""
tests/test_reorder_override.py
-------------------------------
Phase 3 human override + audit log: pure decision rules, the PostgreSQL layer (exercised on SQLite through the REAL
Alembic migration), the memory/database/down routing, and the HTTP endpoints. No network, no model.
"""
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

pytest.importorskip("aiosqlite")

from services import reorder_db as db  # noqa: E402
from services import reorder_override as ov  # noqa: E402


def rec(rec_id="rec_a", sku="SKU-1", qty=100, status="pending", **kw):
    base = {"rec_id": rec_id, "sku_id": sku, "product_name": "Thing", "category": "home", "trigger": "event",
            "event": "DSF", "urgency": "now", "order_qty": qty, "order_by_date": "2026-11-20",
            "window_start": "2026-12-01", "window_end": "2026-12-31", "confidence": 0.94, "confidence_basis": "x",
            "flags": [], "headline": "Reorder 100 units", "reasoning_trace": ["Trigger: event"], "status": status,
            "as_of": "2026-11-18", "replay": False, "created_at": "2026-11-18T10:00:00+00:00"}
    base.update(kw)
    return base


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.reset_engine()
    yield
    db.reset_engine()


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    """A SQLite database built by running the actual Alembic migration."""
    from alembic import command
    from alembic.config import Config
    url = f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    db.reset_engine()
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "migrations"))
    command.upgrade(cfg, "head")
    return url


# ── pure decision rules ────────────────────────────────────────────────────────

def test_approve_keeps_the_system_quantity():
    new, ev = ov.apply_decision(rec(), "approve", "Dulasi")
    assert new["status"] == "approved" and new["final_qty"] == 100 and new["order_qty"] == 100
    assert ev["event_type"] == "approved" and ev["actor"] == "Dulasi" and ev["qty_before"] == ev["qty_after"] == 100


def test_modify_stores_the_human_number_beside_the_original():
    new, ev = ov.apply_decision(rec(), "modify", "Dulasi", "Supplier MOQ is 150", qty=150)
    assert new["status"] == "modified" and new["final_qty"] == 150 and new["order_qty"] == 100   # original untouched
    assert new["modification_delta_pct"] == 50.0 and new["decision_reason"] == "Supplier MOQ is 150"
    assert (ev["qty_before"], ev["qty_after"]) == (100, 150)


def test_reject_needs_a_reason_and_orders_nothing():
    new, _ = ov.apply_decision(rec(), "reject", "Dulasi", "Overstocked at the warehouse")
    assert new["status"] == "rejected" and new["final_qty"] is None
    for bad in (None, "", "  ", "no"):
        with pytest.raises(ov.OverrideError):
            ov.apply_decision(rec(), "reject", "Dulasi", bad)


def test_modify_validation():
    with pytest.raises(ov.OverrideError, match="reason"):
        ov.apply_decision(rec(), "modify", "D", None, qty=120)
    with pytest.raises(ov.OverrideError, match="integer"):
        ov.apply_decision(rec(), "modify", "D", "because", qty=None)
    with pytest.raises(ov.OverrideError, match="integer"):
        ov.apply_decision(rec(), "modify", "D", "because", qty=True)
    with pytest.raises(ov.OverrideError, match="unchanged"):
        ov.apply_decision(rec(), "modify", "D", "because", qty=100)
    with pytest.raises(ov.OverrideError, match="between"):
        ov.apply_decision(rec(), "modify", "D", "because", qty=-1)
    assert ov.apply_decision(rec(), "modify", "D", "stock is covered elsewhere", qty=0)[0]["final_qty"] == 0


def test_a_decision_needs_a_known_kind_and_a_named_actor():
    with pytest.raises(ov.OverrideError):
        ov.apply_decision(rec(), "maybe", "D")
    with pytest.raises(ov.OverrideError, match="actor"):
        ov.apply_decision(rec(), "approve", "  ")


def test_a_decided_recommendation_cannot_be_decided_again():
    with pytest.raises(ov.OverrideError) as e:
        ov.apply_decision(rec(status="approved"), "reject", "D", "changed my mind")
    assert e.value.code == "conflict"


def test_apply_decision_does_not_mutate_its_input():
    original = rec()
    snapshot = dict(original)
    ov.apply_decision(original, "approve", "D")
    assert original == snapshot


# ── database layer (real migration on SQLite) ──────────────────────────────────

def run(coro):
    return asyncio.run(coro)


def test_the_migration_creates_both_tables(sqlite_db):
    assert run(db.ping()) is True
    assert run(db.fetch("nope")) is None and run(db.fetch_list()) == []


def test_save_new_inserts_with_a_created_event(sqlite_db):
    out = run(db.save_new([rec()]))
    assert out[0]["rec_id"] == "rec_a"
    stored = run(db.fetch("rec_a"))
    assert stored["status"] == "pending" and stored["order_qty"] == 100 and stored["reasoning_trace"] == ["Trigger: event"]
    ev = run(db.fetch_events("rec_a"))
    assert [e["event_type"] for e in ev] == ["created"] and ev[0]["actor"] == db.SYSTEM_ACTOR


def test_rescanning_refreshes_a_pending_row_without_a_second_created_event(sqlite_db):
    run(db.save_new([rec(qty=100)]))
    run(db.save_new([rec(qty=120)]))
    assert run(db.fetch("rec_a"))["order_qty"] == 120
    assert len(run(db.fetch_events("rec_a"))) == 1


def test_a_decided_row_is_never_overwritten_by_a_rescan(sqlite_db):
    run(db.save_new([rec(qty=100)]))
    new, ev = ov.apply_decision(rec(qty=100), "modify", "Dulasi", "MOQ", qty=150)
    assert run(db.record_decision(new, ev)) is True
    back = run(db.save_new([rec(qty=999)]))                       # a later scan proposes a different number
    assert back[0]["status"] == "modified" and back[0]["final_qty"] == 150
    stored = run(db.fetch("rec_a"))
    assert stored["order_qty"] == 100 and stored["final_qty"] == 150 and stored["decided_by"] == "Dulasi"
    assert stored["modification_delta_pct"] == 50.0               # same value after a reload as right after the decision


def test_only_one_of_two_simultaneous_decisions_wins(sqlite_db):
    run(db.save_new([rec()]))
    a, ea = ov.apply_decision(rec(), "approve", "alice")
    b, eb = ov.apply_decision(rec(), "reject", "bob", "not needed")
    assert run(db.record_decision(a, ea)) is True
    assert run(db.record_decision(b, eb)) is False                # the row is no longer pending
    assert run(db.fetch("rec_a"))["status"] == "approved"
    assert [e["event_type"] for e in run(db.fetch_events("rec_a"))] == ["created", "approved"]


def test_listing_filters_by_status_and_sku(sqlite_db):
    run(db.save_new([rec("r1", "A"), rec("r2", "B"), rec("r3", "A")]))
    new, ev = ov.apply_decision(rec("r2", "B"), "approve", "D")
    run(db.record_decision(new, ev))
    assert {r["rec_id"] for r in run(db.fetch_list(status="pending"))} == {"r1", "r3"}
    assert {r["rec_id"] for r in run(db.fetch_list(sku_id="A"))} == {"r1", "r3"}
    assert [r["rec_id"] for r in run(db.fetch_list(status="approved"))] == ["r2"]


def test_the_database_layer_reports_unavailable_instead_of_crashing(tmp_path, monkeypatch):
    with pytest.raises(db.DatabaseUnavailable):                   # no DATABASE_URL
        run(db.fetch("x"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}")   # a database nobody migrated
    db.reset_engine()
    with pytest.raises(db.DatabaseUnavailable):
        run(db.save_new([rec()]))
    assert run(db.ping()) is True                                 # reachable, just not migrated


# ── repo routing ───────────────────────────────────────────────────────────────

@pytest.fixture
def repo():
    from backend.agents import reorder_repo
    reorder_repo.clear_memory()
    yield reorder_repo
    reorder_repo.clear_memory()


def test_memory_mode_persist_decide_and_history(repo):
    cur, persisted, warning = run(repo.persist_new([rec()]))
    assert persisted is False and warning is None and cur[0]["status"] == "pending"
    updated, persisted = run(repo.decide("rec_a", "approve", "Dulasi", None, None))
    assert updated["status"] == "approved" and persisted is False
    r, history, source = run(repo.get_one("rec_a"))
    assert source == "memory" and [h["event_type"] for h in history] == ["created", "approved"]
    with pytest.raises(ov.OverrideError):
        run(repo.decide("rec_a", "reject", "D", "second thoughts"))
    with pytest.raises(LookupError):
        run(repo.decide("missing", "approve", "D", None, None))


def test_database_mode_round_trip(repo, sqlite_db):
    cur, persisted, warning = run(repo.persist_new([rec()]))
    assert persisted is True and warning is None
    updated, persisted = run(repo.decide("rec_a", "modify", "Dulasi", "MOQ", 150))
    assert persisted is True and updated["final_qty"] == 150
    rows, source = run(repo.list_recs("modified", None))
    assert source == "postgres" and rows[0]["rec_id"] == "rec_a"
    r, history, source = run(repo.get_one("rec_a"))
    assert [h["event_type"] for h in history] == ["created", "modified"]
    with pytest.raises(ov.OverrideError) as e:
        run(repo.decide("rec_a", "approve", "someone else"))
    assert e.value.code == "conflict"


def test_when_the_database_is_down_scans_survive_but_decisions_are_refused(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'not_migrated.db'}")
    db.reset_engine()
    cur, persisted, warning = run(repo.persist_new([rec()]))
    assert persisted is False and "alembic upgrade head" in warning and cur[0]["rec_id"] == "rec_a"
    rows, source = run(repo.list_recs(None, None))
    assert source == "memory" and len(rows) == 1                  # reads fall back, and say so
    with pytest.raises(db.DatabaseUnavailable):                   # a decision is never applied to memory only
        run(repo.decide("rec_a", "approve", "Dulasi", None, None))
    assert repo.rs._store["rec_a"]["status"] == "pending"


# ── HTTP ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(repo):
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend import agent_router as ar
    app = FastAPI()
    app.include_router(ar.router, prefix="/api/v1/agent")
    run(ar.reorder_repo.persist_new([rec("rec_a", "A"), rec("rec_b", "B")]))
    return TestClient(app)


def test_approve_then_history_over_http(client):
    r = client.post("/api/v1/agent/recommendations/rec_a/decision", json={"decision": "approve", "actor": "Dulasi"})
    assert r.status_code == 200 and r.json()["recommendation"]["status"] == "approved"
    got = client.get("/api/v1/agent/recommendations/rec_a").json()
    assert [h["event_type"] for h in got["history"]] == ["created", "approved"]
    assert got["recommendation"]["order_qty"] == 100 and got["recommendation"]["final_qty"] == 100


def test_modify_over_http_requires_reason_and_quantity(client):
    url = "/api/v1/agent/recommendations/rec_a/decision"
    assert client.post(url, json={"decision": "modify", "actor": "D", "quantity": 150}).status_code == 422
    assert client.post(url, json={"decision": "modify", "actor": "D", "reason": "MOQ"}).status_code == 422
    ok = client.post(url, json={"decision": "modify", "actor": "D", "reason": "MOQ", "quantity": 150})
    assert ok.status_code == 200 and ok.json()["recommendation"]["final_qty"] == 150


def test_http_errors_are_precise(client):
    url = "/api/v1/agent/recommendations/{}/decision"
    body = {"decision": "approve", "actor": "D"}
    assert client.post(url.format("nope"), json=body).status_code == 404
    assert client.get("/api/v1/agent/recommendations/nope").status_code == 404
    assert client.post(url.format("rec_a"), json=body).status_code == 200
    assert client.post(url.format("rec_a"), json=body).status_code == 409            # a decision is final
    assert client.post(url.format("rec_b"), json={"decision": "maybe", "actor": "D"}).status_code == 422
    assert client.post(url.format("rec_b"), json={"decision": "approve", "actor": ""}).status_code == 422


def test_listing_shows_the_decision_state_and_source(client):
    client.post("/api/v1/agent/recommendations/rec_a/decision", json={"decision": "reject", "actor": "D", "reason": "no need"})
    listed = client.get("/api/v1/agent/recommendations?status=rejected").json()
    assert listed["count"] == 1 and listed["source"] == "memory" and listed["recommendations"][0]["final_qty"] is None
    assert client.get("/api/v1/agent/recommendations?status=pending").json()["count"] == 1


def test_a_decision_is_refused_with_503_when_the_database_is_down(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'not_migrated.db'}")
    db.reset_engine()
    r = client.post("/api/v1/agent/recommendations/rec_a/decision", json={"decision": "approve", "actor": "D"})
    assert r.status_code == 503 and "NOT recorded" in r.json()["detail"]
