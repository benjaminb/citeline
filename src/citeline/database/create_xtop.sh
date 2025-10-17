python milvusdb.py --create-xtop-collection \
--name xtop_5 \
--from-collection qwen06_chunks \
--pca-pickle ../../../xtop_pca_1000.pkl \
--mean-vector-pickle ../../../xtop_mean_vector_1000.pkl \
--remove-top-n 5 \
--embedder Qwen/Qwen3-Embedding-0.6B \
--normalize \
--batch-size 32