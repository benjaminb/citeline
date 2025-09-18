BEGIN;
SET ivfflat.probes = 40;
SET synchronous_commit = 'off';
SET maintenance_work_mem = '1GB';
SET max_parallel_workers = 40;
SET work_mem = '4GB';
SET max_parallel_workers_per_gather = 10;
SET effective_cache_size = '24GB';
SELECT pg_prewarm('idx_bge_ivfflat');
SElECT pg_prewarm('lib');
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) WITH random_row AS (
    SELECT bge_norm,
        pubdate
    FROM lib
    WHERE id = (
            SELECT FLOOR(RANDOM() * 2100000) + 1
        )::int
    LIMIT 1
)
SELECT lib.doi,
    lib.title,
    lib.abstract,
    lib.chunk,
    lib.pubdate,
    lib.bge_norm <=> (
        SELECT bge_norm
        FROM random_row
    ) AS distance
FROM lib
WHERE lib.pubdate < (
        SELECT pubdate
        FROM random_row
    )
ORDER BY distance ASC
LIMIT 10000;
ROLLBACK;