singularity exec --env PGHOST=${DB_HOST}.rc.fas.harvard.edu,PGPORT=${DB_PORT},PGDATABASE=${DB_NAME},PGUSER=${DB_USER},PGPASSWORD=${DB_PASSWORD} /n/netscratch/informatics/Everyone/nweeks/ticket/INC05791924/pgvector_0.8.0-pg17.sif psql