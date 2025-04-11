SELECT backend_type,
    context,
    object,
    reads,
    writes,
    extends,
    read_time,
    write_time
FROM pg_stat_io
WHERE reads > 0
    OR writes > 0
    OR extends > 0
ORDER BY backend_type,
    context,
    object;