"""Check if the stored positive vectors are actually similar to the query"""
import pandas as pd
import numpy as np

# Load one example
df = pd.read_json("src/citeline/nn/np_vectors_val.jsonl", lines=True, nrows=1)
row = df.iloc[0]

query_vec = np.array(row['vector'])
positive_vecs = np.array(row['positive_vectors'])
negative_vecs = np.array(row['negative_vectors'])

print(f"Query vector norm: {np.linalg.norm(query_vec):.6f}")
print(f"Number of positive vectors: {len(positive_vecs)}")
print(f"Number of negative vectors: {len(negative_vecs)}")

# Calculate similarities
pos_sims = [np.dot(query_vec, pos_vec) for pos_vec in positive_vecs]
neg_sims = [np.dot(query_vec, neg_vec) for neg_vec in negative_vecs]

print(f"\nPositive similarities: {pos_sims}")
print(f"Negative similarities: {neg_sims}")
print(f"\nMean positive sim: {np.mean(pos_sims):.4f}")
print(f"Mean negative sim: {np.mean(neg_sims):.4f}")
print(f"Margin (neg - pos): {np.mean(neg_sims) - np.mean(pos_sims):.4f}")

# This margin should be NEGATIVE (positives should be MORE similar than negatives)
if np.mean(pos_sims) > np.mean(neg_sims):
    print("\n✓ GOOD: Positives are more similar than negatives")
else:
    print("\n✗ BAD: Negatives are more similar than positives!")
