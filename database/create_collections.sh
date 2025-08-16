python milvusdb.py --create-collection \
--name astrollama_chunks \
--data-source ../data/research_chunks.jsonl \
--embedder UniverseTBD/astrollama \
# --normalize

python milvusdb.py --create-collection \
--name astrollama_contributions \
--data-source ../data/research_contributions.jsonl \
--embedder UniverseTBD/astrollama \
# --normalize

