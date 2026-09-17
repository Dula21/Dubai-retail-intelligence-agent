-- ============================================================
-- Dubai Retail Intelligence Agent — Database Init
-- ============================================================
-- This runs automatically when the postgres container starts for
-- the first time. Tables are ready for Phase 2 agent queries.

-- Query audit log — every agent query gets recorded
CREATE TABLE IF NOT EXISTS query_log (
    id              SERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    query_language  VARCHAR(10),
    result_count    INTEGER,
    latency_ms      FLOAT,
    confidence_gate FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Agent reasoning traces (Phase 2)
CREATE TABLE IF NOT EXISTS agent_traces (
    id              SERIAL PRIMARY KEY,
    query_id        INTEGER REFERENCES query_log(id),
    node_name       VARCHAR(100),
    node_input      JSONB,
    node_output     JSONB,
    duration_ms     FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Reorder recommendations (Phase 3)
CREATE TABLE IF NOT EXISTS reorder_recommendations (
    id                  SERIAL PRIMARY KEY,
    sku_id              VARCHAR(20) NOT NULL,
    recommended_qty     INTEGER,
    confidence_score    FLOAT,
    reasoning_trace     TEXT,
    trigger_event       VARCHAR(50),
    status              VARCHAR(20) DEFAULT 'pending',
    created_at          TIMESTAMP DEFAULT NOW(),
    actioned_at         TIMESTAMP
);

-- Index for fast SKU lookups
CREATE INDEX IF NOT EXISTS idx_reorder_sku ON reorder_recommendations(sku_id);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);
