BEGIN;
SET ivfflat.probes = 10;
SET synchronous_commit = 'off';
SET maintenance_work_mem = '1GB';
SET max_parallel_workers = 10;
SET work_mem = '4GB';
SET max_parallel_workers_per_gather = 10;
SET effective_cache_size = '24GB';
SELECT pg_prewarm('idx_bge_ivfflat');
SElECT pg_prewarm('lib');
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) WITH random_vector AS (
    SELECT bge,
    FROM lib
    WHERE id = (
            SELECT FLOOR(RANDOM() * 2100000) + 1
        )::int
    LIMIT 1
)
SELECT title,
    bge <=> (
        SELECT bge
        from random_vector
    ) as distance
from lib
ORDER BY distance ASC
LIMIT 5000;
ROLLBACK;