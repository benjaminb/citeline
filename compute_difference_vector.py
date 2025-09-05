import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from embedders import Embedder
from database.milvusdb import MilvusDB
from query_expander import get_expander

tqdm.pandas()

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME = "qwen06_chunks"
QUERY_EXPANSION = "add_prev_3"

# Globals
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
embedder = Embedder.create(model_name=EMBEDDING_MODEL_NAME, device=device, normalize=True, for_queries=True)
db = MilvusDB()


def get_sample_df(n=1000, embedder=None):
    """
    Subsamples n examples from the nontrivial_checked.jsonl dataset, embeds sent_no_cit column,
    then denormalizes on citation_dois to get one row per (example, target_doi) pair.
    Returns a DataFrame with columns: sent_no_cit, vector, target_doi
    """
    examples = pd.read_json("data/dataset/nontrivial_checked.jsonl", lines=True)
    examples = examples.sample(n=n, random_state=43)
    print(f"Loaded {len(examples)} examples")

    # Query Expansion on sent_no_cit
    expander = get_expander(QUERY_EXPANSION, path_to_data="data/preprocessed/reviews.jsonl")
    print(f"Using query expansion: {expander}")
    examples["sent_no_cit"] = expander(examples)

    # Add vector column to examples
    examples["vector"] = examples.progress_apply(lambda row: embedder([row["sent_no_cit"]])[0], axis=1)

    # Denormalize on citation_dois (targets)
    examples = examples.explode("citation_dois", ignore_index=True)
    print(f"Number of samples after denormalization: {examples.shape[0]}")
    examples.rename(columns={"citation_dois": "target_doi"}, inplace=True)
    return examples


def most_similar_to_query(example: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    """
    Takes in an example (with 'vector' column already set), and from the candidates
    (returned entities with that doi from the database), returns the vector most similar
    to the example's vector.

    """
    # Converts 'vector' column to rows * dim array, holding the candidate vectors
    candidate_vectors = np.stack(candidates["vector"])
    best_idx = np.argmax(np.dot(candidate_vectors, example["vector"]))
    best_vector = candidate_vectors[best_idx]
    return best_vector


def compute_difference_vector(example: pd.Series, doi: str) -> np.ndarray:
    """
    Computes the difference vector between the example's vector and the most similar vector
    from the candidates retrieved by doi
    """
    candidates = db.select_by_doi(doi, collection_name=COLLECTION_NAME)
    most_similar = most_similar_to_query(example, candidates)
    # NOTE: be sure to remain consistent that query vector is first, target vector is 2nd in diff
    return example["vector"] - most_similar


def compute_vector_stats(vectors: np.ndarray) -> dict:
    """
    Computes basic statistics for a set of vectors.

    Args:
        vectors (np.ndarray): An array of shape (n_samples, n_features) containing the vectors.

    Returns:
        dict: A dictionary containing the mean, std, min, and max for each feature.
    """
    mean_vector = np.mean(vectors, axis=0)
    cov_matrix = np.cov(vectors.T)
    trace = np.trace(cov_matrix)
    stats = {
        "mean_vector": mean_vector,
        "average_norm": np.linalg.norm(mean_vector),
        "std": np.std(vectors, axis=0),
        "trace": trace,
        "determinant": np.linalg.det(cov_matrix),
    }
    return stats


def main():
    sample_df = get_sample_df(n=100, embedder=embedder)
    diff_vectors = np.zeros((len(sample_df), embedder.dim))
    for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        diff_vectors[i] = compute_difference_vector(row, doi=row["target_doi"])

    stats = compute_vector_stats(diff_vectors)
    for key, value in stats.items():
        print(f"{key}: {value}")
    outfile_name = f"{COLLECTION_NAME}_{QUERY_EXPANSION}_diff_vector.npy"
    difference_vector = stats["mean_vector"]
    np.save(outfile_name, difference_vector)
    print(f"Saved difference vector to {outfile_name}")


if __name__ == "__main__":
    main()
