python milvusdb.py --create-collection \
--name bge_chunks \
--data-source ../data/research_chunks.jsonl \
--embedder BAAI/bge-large-en-v1.5 \
--normalize

python milvusdb.py --create-collection \
--name bge_contributions \
--data-source ../data/research_contributions.jsonl \
--embedder BAAI/bge-large-en-v1.5 \
--normalize