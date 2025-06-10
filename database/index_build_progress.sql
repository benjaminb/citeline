SELECT
  pid,
  relid::regclass AS table_name,
  index_relid::regclass AS index_name,
  phase,
  ROUND(100.0 * blocks_done / NULLIF(blocks_total, 0), 2) AS percent_done
FROM
  pg_stat_progress_create_index;
