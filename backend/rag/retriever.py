"""
Bilingual RAG Retriever — Confidence-Gated
==========================================
Loads product catalogue and sales data, builds a ChromaDB vector store,
and exposes a confidence-gated retrieval interface.

ENGINEERING DECISION: ChromaDB over FAISS or Pinecone
------------------------------------------------------
Alternatives considered:
  A) FAISS (Meta) — fast, in-memory, no persistence
  B) Pinecone   — managed, excellent scale, costs $$$
  C) ChromaDB   — embedded, persistent, open source, SQLite-backed
Chosen: ChromaDB

Why for this stage:
  1. Runs locally without cloud dependency — matches data residency concerns
     relevant for UAE enterprise (G42, Presight).
  2. Persistent across restarts — no re-embedding on every launch.
  3. Metadata filtering built in — critical for "show me home products under
     AED 100" queries (price + category filter before vector search).
  4. Upgrades to distributed Chroma in production without API change.

Trade-off: At >1M vectors, FAISS or Weaviate would outperform. 
500 SKUs → ChromaDB is the right call.

HIRING NOTE: Being able to explain WHY you didn't use Pinecone is
often more impressive than saying you used it.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .embeddings import (
    CONFIDENCE_THRESHOLD,
    EmbeddingConfig,
    EmbeddingManager,
    chunk_catalogue_for_embedding,
    detect_language,
)


# ---------------------------------------------------------------------------
# LangChain Document Loader (wraps CSV → LangChain Document objects)
# ---------------------------------------------------------------------------


def load_csv(path: str) -> list[dict]:
    """Load a CSV file and return a list of dicts. Pure stdlib."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class RetailCatalogueLoader:
    """
    LangChain-compatible document loader for the product catalogue CSV.
    
    ENGINEERING DECISION: Custom loader over CSVLoader
    LangChain's built-in CSVLoader treats each row as raw text.
    We need structured metadata extraction (price, category, bilingual names)
    for ChromaDB metadata filtering. A custom loader gives us that.
    
    This also mirrors the pattern used in DPI (Dubai Property Intelligence)
    for DLD transaction data — consistent loader pattern across all three
    portfolio projects.
    """

    def __init__(self, catalogue_path: str, sales_path: Optional[str] = None):
        self.catalogue_path = catalogue_path
        self.sales_path = sales_path
        self._catalogue_cache: Optional[list[dict]] = None
        self._sales_index:     Optional[dict[str, dict]] = None

    def load_catalogue(self) -> list[dict]:
        """Load and cache the product catalogue."""
        if self._catalogue_cache is None:
            rows = load_csv(self.catalogue_path)
            # Cast numeric fields
            for row in rows:
                row["price_aed"]            = float(row["price_aed"])
                row["supplier_lead_days"]   = int(row["supplier_lead_days"])
                row["supplier_reliability"] = float(row["supplier_reliability"])
                row["reorder_point_units"]  = int(row["reorder_point_units"])
                row["reorder_qty_units"]    = int(row["reorder_qty_units"])
                row["is_ramadan_hero"]      = int(row["is_ramadan_hero"])
                row["is_dsf_hero"]          = int(row["is_dsf_hero"])
            self._catalogue_cache = rows
            logger.info("catalogue_loaded", sku_count=len(rows))
        return self._catalogue_cache

    def load_sales_summary(self) -> dict[str, dict]:
        """
        Build a per-SKU sales summary indexed by sku_id.
        Used to enrich retrieval results with performance context.
        
        Returns dict: sku_id → {total_units, total_revenue, avg_daily_units,
                                  stockout_days, last_30d_units}
        """
        if self._sales_index is not None:
            return self._sales_index

        if not self.sales_path or not os.path.exists(self.sales_path):
            logger.warning("sales_file_not_found", path=self.sales_path)
            return {}

        rows = load_csv(self.sales_path)
        index: dict[str, dict] = {}

        for row in rows:
            sku_id = row["sku_id"]
            units  = int(row["units_sold"])
            rev    = float(row["revenue_aed"])
            date_s = row["date"]

            if sku_id not in index:
                index[sku_id] = {
                    "total_units":    0,
                    "total_revenue":  0.0,
                    "stockout_days":  0,
                    "all_dates":      [],
                    "last_30d_units": 0,
                }

            idx = index[sku_id]
            idx["total_units"]   += units
            idx["total_revenue"] += rev
            idx["stockout_days"] += int(row.get("stockout_occurred", 0))
            idx["all_dates"].append(date_s)

            # Approximate last-30-day: dates after 2026-07-15
            if date_s >= "2026-07-15":
                idx["last_30d_units"] += units

        # Compute averages, clean up helper fields
        for sku_id, idx in index.items():
            day_count = max(1, len(idx["all_dates"]))
            idx["avg_daily_units"] = round(idx["total_units"] / day_count, 2)
            idx["total_revenue"]   = round(idx["total_revenue"], 2)
            del idx["all_dates"]

        self._sales_index = index
        logger.info("sales_summary_built", sku_count=len(index))
        return index


# ---------------------------------------------------------------------------
# Vector Store Manager
# ---------------------------------------------------------------------------


class RetailVectorStore:
    """
    Manages the ChromaDB vector store for product catalogue retrieval.
    Handles: ingestion, persistence, and confidence-gated query.
    """

    COLLECTION_NAME = "dubai_retail_catalogue_v1"

    def __init__(
        self,
        persist_directory: str = ".chroma_db",
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        self.persist_directory = persist_directory
        self.embedding_manager = embedding_manager or EmbeddingManager(config)
        self._collection = None

    def _get_chroma_client(self):
        """Lazy-init ChromaDB client."""
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
        
        return chromadb.PersistentClient(path=self.persist_directory)

    @property
    def collection(self):
        if self._collection is None:
            client = self._get_chroma_client()
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                # ENGINEERING NOTE: cosine distance for normalised embeddings.
                # ChromaDB stores distance (1 - cosine), so we convert back
                # in the retrieval step. Alternatively use "ip" (inner product)
                # which equals cosine for normalised vectors — but cosine is
                # more explicit and readable.
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def ingest_catalogue(
        self,
        catalogue_rows: list[dict],
        force_reingest: bool = False,
    ) -> int:
        """
        Embed and store the product catalogue in ChromaDB.
        
        Args:
            catalogue_rows:  List of product dicts from RetailCatalogueLoader
            force_reingest:  If True, drop existing collection and re-embed.
                             Use when catalogue data changes significantly.
        
        Returns: Number of documents ingested.
        
        ENGINEERING NOTE: force_reingest=False is the default so that a
        server restart doesn't trigger a ~2 minute re-embedding of 500 SKUs.
        Redis will cache embeddings in Phase 2 for sub-second warm starts.
        """
        if force_reingest:
            client = self._get_chroma_client()
            try:
                client.delete_collection(self.COLLECTION_NAME)
            except Exception:
                pass
            self._collection = None  # Reset so it gets recreated

        # Check if already ingested
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reingest:
            logger.info(
                "vector_store_already_populated",
                existing_docs=existing_count,
                message="Skipping re-ingestion. Use force_reingest=True to rebuild.",
            )
            return existing_count

        chunks = chunk_catalogue_for_embedding(catalogue_rows)
        
        # Batch embed
        logger.info("embedding_catalogue", chunk_count=len(chunks))
        start = time.perf_counter()
        
        texts    = [c["text"] for c in chunks]
        doc_ids  = [c["doc_id"] for c in chunks]
        metadata = [c["metadata"] for c in chunks]
        
        # ChromaDB batches internally but we control batch size for memory
        BATCH = 50
        for i in range(0, len(chunks), BATCH):
            batch_texts = texts[i : i + BATCH]
            batch_ids   = doc_ids[i : i + BATCH]
            batch_meta  = metadata[i : i + BATCH]
            
            embeddings = self.embedding_manager.embed_documents(batch_texts)
            
            self.collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )
            logger.debug("batch_ingested", batch_start=i, batch_end=i + BATCH)

        elapsed = time.perf_counter() - start
        ingested = self.collection.count()
        logger.info(
            "catalogue_ingested",
            total_docs=ingested,
            elapsed_seconds=round(elapsed, 2),
            docs_per_second=round(ingested / elapsed, 1),
        )
        return ingested

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        metadata_filter: Optional[dict] = None,
        min_confidence: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant products with confidence gating.
        
        CONFIDENCE GATE: Results below min_confidence are excluded entirely.
        This is the core hallucination-prevention mechanism from DPI, applied
        to retrieval. The agent in Phase 2 can request a broader search if
        no results pass the gate, rather than returning low-confidence noise.
        
        Args:
            query_text:      User query in Arabic or English
            n_results:       Number of candidates to retrieve before filtering
            metadata_filter: ChromaDB where-clause for pre-filtering
                             e.g. {"category": "fashion", "price_aed": {"$lt": 200}}
            min_confidence:  Override default threshold (0.72)
        
        Returns:
            List of results above confidence threshold, each with:
            - sku_id, name_en, name_ar, category, price_aed
            - similarity_score (0.0–1.0)
            - detected_query_language
        
        ENGINEERING NOTE: We retrieve n_results=10 candidates then filter.
        If all 10 fail the confidence gate, the agent gets an empty list and
        can widen the search or acknowledge uncertainty. This is intentional —
        returning garbage at 0.4 similarity is worse than returning nothing.
        """
        threshold = min_confidence if min_confidence is not None else CONFIDENCE_THRESHOLD
        lang      = detect_language(query_text)

        logger.info(
            "retrieval_query",
            query_preview=query_text[:60],
            detected_language=lang,
            confidence_threshold=threshold,
        )

        query_embedding = self.embedding_manager.embed_query(query_text)

        # Build ChromaDB query kwargs
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            query_kwargs["where"] = metadata_filter

        raw = self.collection.query(**query_kwargs)

        # Parse and gate results
        results = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # ChromaDB cosine returns distance (0=identical, 2=opposite)
            # Convert to similarity: similarity = 1 - (distance / 2)
            # For cosine in ChromaDB: distance = 1 - cosine_similarity
            similarity = 1.0 - dist  # ChromaDB cosine distance is already 1 - cosine

            if similarity < threshold:
                logger.debug(
                    "result_below_threshold",
                    sku_id=meta.get("sku_id"),
                    similarity=round(similarity, 4),
                    threshold=threshold,
                )
                continue  # GATE: drop this result

            results.append({
                **meta,
                "similarity_score":     round(similarity, 4),
                "query_language":       lang,
                "document_text_preview": doc[:120],
            })

        logger.info(
            "retrieval_complete",
            candidates_retrieved=len(raw["documents"][0]),
            results_above_threshold=len(results),
            threshold=threshold,
        )

        # Sort by similarity descending
        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# High-level Retrieval Interface (used by LangGraph agent in Phase 2)
# ---------------------------------------------------------------------------


class RetailRetriever:
    """
    Unified retrieval interface combining vector search + sales context.
    This is the object the LangGraph agent nodes will call.
    
    Design principle: The retriever returns rich context — not just a
    product list. Each result includes sales performance data so the agent
    can reason about "underperforming" vs "trending" without an extra lookup.
    """

    def __init__(
        self,
        catalogue_path: str,
        sales_path: Optional[str] = None,
        persist_directory: str    = ".chroma_db",
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.loader        = RetailCatalogueLoader(catalogue_path, sales_path)
        self.embedding_mgr = EmbeddingManager(embedding_config)
        self.vector_store  = RetailVectorStore(persist_directory, self.embedding_mgr)
        self._ingested     = False

    def initialize(self, force_reingest: bool = False) -> dict:
        """
        Load catalogue and build (or load) vector store.
        Call once at startup. Idempotent unless force_reingest=True.
        """
        catalogue = self.loader.load_catalogue()
        n = self.vector_store.ingest_catalogue(catalogue, force_reingest=force_reingest)
        self._ingested = True
        
        return {
            "status":      "ready",
            "sku_count":   len(catalogue),
            "vector_docs": n,
        }

    def search(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str]  = None,
        max_price_aed: Optional[float] = None,
        min_price_aed: Optional[float] = None,
        ramadan_hero_only: bool = False,
        dsf_hero_only:     bool = False,
    ) -> dict[str, Any]:
        """
        Main search endpoint called by the LangGraph agent.
        
        Combines:
          1. Metadata pre-filtering (category, price range, seasonal flags)
          2. Vector similarity search with confidence gating
          3. Sales context enrichment
        
        Returns structured response ready for the agent to reason over.
        """
        if not self._ingested:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")

        # Build metadata filter
        meta_filter: dict[str, Any] = {}
        conditions: list[dict] = []

        if category:
            conditions.append({"category": {"$eq": category}})
        if max_price_aed is not None:
            conditions.append({"price_aed": {"$lte": max_price_aed}})
        if min_price_aed is not None:
            conditions.append({"price_aed": {"$gte": min_price_aed}})
        if ramadan_hero_only:
            conditions.append({"is_ramadan_hero": {"$eq": 1}})
        if dsf_hero_only:
            conditions.append({"is_dsf_hero": {"$eq": 1}})

        if len(conditions) == 1:
            meta_filter = conditions[0]
        elif len(conditions) > 1:
            meta_filter = {"$and": conditions}

        # Vector search
        results = self.vector_store.query(
            query_text=query,
            n_results=n_results * 2,  # Over-fetch then filter
            metadata_filter=meta_filter if meta_filter else None,
        )[:n_results]

        # Enrich with sales context
        sales_summary = self.loader.load_sales_summary()
        for r in results:
            sku_id = r.get("sku_id")
            if sku_id and sku_id in sales_summary:
                r["sales"] = sales_summary[sku_id]
            else:
                r["sales"] = None

        return {
            "query":          query,
            "query_language": results[0]["query_language"] if results else detect_language(query),
            "results":        results,
            "result_count":   len(results),
            "confidence_gate": CONFIDENCE_THRESHOLD,
            "filters_applied": {
                "category":         category,
                "max_price_aed":    max_price_aed,
                "min_price_aed":    min_price_aed,
                "ramadan_hero_only":ramadan_hero_only,
                "dsf_hero_only":    dsf_hero_only,
            },
        }
