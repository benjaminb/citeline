singularity run \
  --cleanenv \
  --env PGDATA="/n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citelinedb" \
  --env PGPORT=5432 \
  --scratch /run \
  /n/singularity_images/OOD/postgres/pgvector_0.8.0-pg17.sif
  