-- index progress report from PGVECTOR
-- for hnsw indexes
-- Phase 1: initializing
-- Phase 2: loading tuples
SELECT phase,
    round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS "%"
FROM pg_stat_progress_create_index;