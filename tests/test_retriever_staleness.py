"""
tests/test_retriever_staleness.py
----------------------------------
A persisted vector index must match the catalogue it was built from. Found when the container
kept a 300-doc Phase 1 index for a 500-SKU catalogue and silently reported "already populated".
No embedding model is loaded here: a fake embedder and a fake chunker keep these tests fast.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.rag import retriever as r  # noqa: E402


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[1.0, float(len(t) % 7 + 1), float(i + 1), 0.5] for i, t in enumerate(texts)]


def fake_chunker(rows):
    return [{"doc_id": f"doc-{row['sku_id']}", "text": f"{row['sku_id']} {row['name_en']}",
             "metadata": {"sku_id": row["sku_id"], "category": row.get("category", "home")}}
            for row in rows]


def catalogue(n, name="Lantern"):
    return [{"sku_id": f"HOM-{i:04d}", "name_en": f"{name} {i}", "category": "home"} for i in range(n)]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "chunk_catalogue_for_embedding", fake_chunker)
    return r.RetailVectorStore(persist_directory=str(tmp_path / "chroma"), embedding_manager=FakeEmbedder())


def names(store):
    return sorted(getattr(c, "name", c) for c in store._get_chroma_client().list_collections())


def test_first_ingest_creates_a_fingerprinted_collection(store):
    assert store.ingest_catalogue(catalogue(5)) == 5
    assert len(names(store)) == 1 and names(store)[0].startswith(r.RetailVectorStore.COLLECTION_NAME + "_")


def test_same_catalogue_is_not_re_embedded(store):
    store.ingest_catalogue(catalogue(5))
    calls = store.embedding_manager.calls
    assert store.ingest_catalogue(catalogue(5)) == 5
    assert store.embedding_manager.calls == calls


def test_old_unfingerprinted_index_is_replaced_not_trusted(store):
    # What the container had: the base-named collection holding 3 old docs.
    old = store._get_chroma_client().get_or_create_collection(r.RetailVectorStore.COLLECTION_NAME)
    old.add(ids=["a", "b", "c"], embeddings=[[1.0, 1.0, 1.0, 1.0]] * 3, documents=["x"] * 3)
    assert store.ingest_catalogue(catalogue(5)) == 5          # 5 expected, not the 3 found
    assert r.RetailVectorStore.COLLECTION_NAME not in names(store)
    assert len(names(store)) == 1


def test_changed_catalogue_content_triggers_reindex_and_prunes_the_old_one(store):
    store.ingest_catalogue(catalogue(5, name="Lantern"))
    first = names(store)
    store.ingest_catalogue(catalogue(5, name="Prayer Mat"))     # same size, different content
    second = names(store)
    assert len(second) == 1 and second != first


def test_grown_catalogue_is_re_ingested(store):
    store.ingest_catalogue(catalogue(3))
    assert store.ingest_catalogue(catalogue(5)) == 5


def test_interrupted_ingest_is_rebuilt(store):
    store.ingest_catalogue(catalogue(5))
    store.collection.delete(ids=["doc-HOM-0000", "doc-HOM-0001"])   # simulate a half-finished index
    assert store.collection.count() == 3
    assert store.ingest_catalogue(catalogue(5)) == 5


def test_force_reingest_rebuilds_even_when_matching(store):
    store.ingest_catalogue(catalogue(5))
    calls = store.embedding_manager.calls
    assert store.ingest_catalogue(catalogue(5), force_reingest=True) == 5
    assert store.embedding_manager.calls > calls


def test_fingerprint_is_stable_and_content_sensitive():
    a = r.RetailVectorStore._fingerprint(fake_chunker(catalogue(3)))
    assert a == r.RetailVectorStore._fingerprint(fake_chunker(catalogue(3)))
    assert a != r.RetailVectorStore._fingerprint(fake_chunker(catalogue(3, name="Mat")))


# ── sales summary: "last 30 days" follows the data, not a hard-coded date ─────────────────────

def write_sales(path, rows):
    path.write_text("date,sku_id,units_sold,unit_price_aed,revenue_aed,stockout_occurred\n" +
                    "\n".join(f"{d},{s},{u},10.0,{u * 10.0},{o}" for d, s, u, o in rows), encoding="utf-8")


def test_last_30d_units_is_relative_to_the_newest_date(tmp_path):
    f = tmp_path / "sales.csv"
    write_sales(f, [("2026-09-01", "A", 5, 0),     # older than 30 days before the newest date
                    ("2026-09-20", "A", 3, 1),     # inside the window
                    ("2026-10-08", "A", 2, 0)])    # newest date
    loader = r.RetailCatalogueLoader("unused.csv", sales_path=str(f))
    summary = loader.load_sales_summary()["A"]
    assert summary["last_30d_units"] == 5          # 3 + 2, not 10
    assert summary["stockout_days"] == 1


def test_no_sales_path_is_not_an_error(tmp_path):
    assert r.RetailCatalogueLoader("unused.csv").load_sales_summary() == {}