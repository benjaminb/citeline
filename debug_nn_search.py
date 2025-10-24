"""Debug script to test NN transformation and search"""
import torch
import pandas as pd
import numpy as np
from citeline.database.milvusdb import MilvusDB

# Load one example from validation set
df = pd.read_json("src/citeline/nn/np_vectors_val.jsonl", lines=True, nrows=1)
print("Loaded 1 validation example")
print(f"Keys: {df.columns.tolist()}")
print(f"Citation DOIs: {df.iloc[0]['citation_dois']}")
print(f"Pubdate: {df.iloc[0]['pubdate']}")
print(f"Vector shape: {len(df.iloc[0]['vector'])}")
print(f"Vector norm (before NN): {np.linalg.norm(df.iloc[0]['vector']):.6f}")

# Load NN model
model = torch.jit.load("notebooks/best_model_traced.pth", map_location="cpu")
model.eval()

# Transform vector with NN
original_vector = df.iloc[0]['vector']
with torch.no_grad():
    vectors_list = [original_vector]  # Simulate batch["vector"].tolist()
    input_vectors = torch.tensor(vectors_list, device="cpu", dtype=torch.float32)
    transformed = model(input_vectors).numpy()
    transformed_vector = transformed[0].tolist()

print(f"\nVector norm (after NN): {np.linalg.norm(transformed_vector):.6f}")
print(f"Cosine similarity (before vs after NN): {np.dot(original_vector, transformed_vector):.6f}")

# Now test search with both vectors
db = MilvusDB()

print("\n=== SEARCH WITH ORIGINAL VECTOR ===")
search_results_orig = db.search(
    collection_name="qwen06_chunks",
    query_records=[df.iloc[0].to_dict()],
    query_vectors=[original_vector],
    limit=10,
    output_fields=["text", "doi", "pubdate", "citation_count"],
)
print(f"Number of results: {len(search_results_orig[0])}")
print(f"Top 3 DOIs: {[r['doi'] for r in search_results_orig[0][:3]]}")
print(f"Target DOI(s): {df.iloc[0]['citation_dois']}")
target_in_results = any(r['doi'] in df.iloc[0]['citation_dois'] for r in search_results_orig[0])
print(f"Target found in top 10: {target_in_results}")

print("\n=== SEARCH WITH NN-TRANSFORMED VECTOR ===")
search_results_nn = db.search(
    collection_name="qwen06_chunks",
    query_records=[df.iloc[0].to_dict()],
    query_vectors=[transformed_vector],
    limit=10,
    output_fields=["text", "doi", "pubdate", "citation_count"],
)
print(f"Number of results: {len(search_results_nn[0])}")
print(f"Top 3 DOIs: {[r['doi'] for r in search_results_nn[0][:3]]}")
target_in_results_nn = any(r['doi'] in df.iloc[0]['citation_dois'] for r in search_results_nn[0])
print(f"Target found in top 10: {target_in_results_nn}")

print("\n=== METRICS COMPARISON ===")
print(f"Top 3 distances (original): {[r['metric'] for r in search_results_orig[0][:3]]}")
print(f"Top 3 distances (NN): {[r['metric'] for r in search_results_nn[0][:3]]}")
