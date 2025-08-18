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

python milvusdb.py --create-collection \
--name astrobert_chunks \
--data-source ../data/research_chunks.jsonl \
--embedder adsabs/astroBERT \
# --normalize

python milvusdb.py --create-collection \
--name astrobert_contributions \
--data-source ../data/research_contributions.jsonl \
--embedder adsabs/astroBERT \
# --normalize

python milvusdb.py --create-collection \
--name astrobert_chunks_norm \
--data-source ../data/research_chunks.jsonl \
--embedder adsabs/astroBERT \
--normalize

python milvusdb.py --create-collection \
--name astrobert_contributions_norm \
--data-source ../data/research_contributions.jsonl \
--embedder adsabs/astroBERT \
--normalize

