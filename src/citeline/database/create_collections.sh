python milvusdb.py --create-collection \
--name qwen06b_chunks \
--data-source ../../../data/research_chunks.jsonl \
--embedder Qwen/Qwen3-Embedding-0.6B \
--batch-size 16