# Creates
# bge chunks (continues)
# astrobert 
# astrobert normalized
# qwen8b
# astrollama

python milvusdb.py --create-collection \
--name bge_chunks \
--data-source ../data/research_chunks.jsonl \
--embedder BAAI/bge-large-en-v1.5 \
--normalize \
--batch-size 32

python milvusdb.py --create-collection \
--name astrobert_chunks \
--data-source ../data/research_chunks.jsonl \
--embedder adsabs/astroBERT \
--batch-size 32
# --normalize

python milvusdb.py --create-collection \
--name astrobert_contributions \
--data-source ../data/research_contributions.jsonl \
--embedder adsabs/astroBERT \
--batch-size 64
# --normalize

python milvusdb.py --create-collection \
--name specter_contributions \
--data-source ../data/research_contributions.jsonl \
--embedder allenai/specter2 \
--batch-size 64
# --normalize

python milvusdb.py --create-collection \
--name specter_chunks \
--data-source ../data/research_chunks.jsonl \
--embedder allenai/specter2 \
--batch-size 32
# --normalize

python milvusdb.py --create-collection \
--name qwen8b_contributions \
--data-source ../data/research_contributions.jsonl \
--embedder Qwen/Qwen3-Embedding-8B \
--batch-size 32

python milvusdb.py --create-collection \
--name qwen8b_chunks \
--data-source ../data/research_chunks.jsonl \
--embedder Qwen/Qwen3-Embedding-8B \
--batch-size 16

python milvusdb.py --create-collection \
--name astrollama_contributions \
--data-source ../data/research_contributions.jsonl \
--embedder UniverseTBD/astrollama \
--batch-size 32

python milvusdb.py --create-collection \
--name astrollama_chunks \
--data-source ../data/research_chunks.jsonl \
--embedder UniverseTBD/astrollama \
--batch-size 16
