import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.decomposition import PCA
from citeline.embedders import Embedder

# Sample queries and chunks
NUM_SAMPLE = 1000
queries = (
    pd.read_json("data/dataset/nontrivial_checked.jsonl", lines=True)
    .sample(n=NUM_SAMPLE, random_state=42)
    .reset_index(drop=True)
)
chunks = (
    pd.read_json("data/research_chunks.jsonl", lines=True).sample(n=NUM_SAMPLE, random_state=42).reset_index(drop=True)
)

# Instantiate embedder
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
embedder = Embedder.create("Qwen/Qwen3-Embedding-0.6B", device=device, normalize=True)
print(f"Embedder: {embedder}")

query_vectors = embedder(queries["sent_no_cit"].tolist(), for_queries=True)
chunk_vectors = embedder(chunks["text"].tolist(), for_queries=False)
data_matrix = np.vstack([query_vectors, chunk_vectors])
print("Data matrix shape:", data_matrix.shape)

mean_vector = np.mean(data_matrix, axis=0)
centered_data = data_matrix - mean_vector
with open(f"xtop_mean_vector_{NUM_SAMPLE}.pkl", "wb") as f:
    pickle.dump(mean_vector, f)
print("Mean vector saved. Working on PCA...")

pca = PCA()
pca.fit(centered_data)

with open(f"xtop_pca_{NUM_SAMPLE}.pkl", "wb") as f:
    pickle.dump(pca, f)

print("PCA saved.")
