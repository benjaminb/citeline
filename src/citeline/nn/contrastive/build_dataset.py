"""
Build contrastive HDF5 datasets from nontrivial_llm.jsonl.

Each split (train/val/test) stores:
  /queries              float32 (N, dim)
  /positives            float32 (N, dim)
  /negatives            float32 (N, K, dim)
  /neg_cosine_sims      float32 (N, K)
  /neg_retrieval_ranks  int32   (N, K)

Usage:
  python -m citeline.nn.contrastive.build_dataset configs/contrastive/build.yaml
"""

import sys
import os
from datetime import date

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from citeline.database.milvusdb import MilvusDB
from citeline.embedders import Embedder
from citeline.nn.contrastive.config import BuildConfig


def pubdate_str_to_int(pubdate_str: str) -> int:
    """Convert 'YYYY-MM-DD' string to YYYYMMDD integer for Milvus filter."""
    return int(pubdate_str.replace("-", ""))


def embed_in_batches(texts: list[str], embedder: Embedder, batch_size: int = 32) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding queries"):
        batch = texts[i : i + batch_size]
        vecs = embedder(batch, for_queries=True)
        all_vecs.append(vecs)
    return np.concatenate(all_vecs, axis=0)


def get_best_positive(query_vec: np.ndarray, target_doi: str, collection: str, db: MilvusDB) -> np.ndarray | None:
    """Return the chunk vector from target_doi most similar to the query. Returns None if DOI not in DB."""
    records = db.select_by_doi(doi=target_doi, collection_name=collection)
    if records.empty:
        return None
    target_vecs = np.array(records["vector"].tolist())
    sims = query_vec @ target_vecs.T
    return target_vecs[np.argmax(sims)]


def get_negatives_with_metadata(
    query_vec: np.ndarray,
    pubdate_int: int,
    all_target_dois: set,
    collection: str,
    db: MilvusDB,
    K: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Retrieve top-k results from Milvus, filter out all target DOIs, return first K.

    Returns (neg_vectors, neg_cosine_sims, neg_retrieval_ranks) or None if < K found.
    """
    query_record = {"pubdate": pubdate_int}
    results = db.search(
        collection_name=collection,
        query_records=[query_record],
        query_vectors=[query_vec.tolist()],
        limit=top_k,
        output_fields=["doi", "vector"],
    )[0]

    non_target = [r for r in results if r["doi"] not in all_target_dois]
    if len(non_target) < K:
        return None

    non_target = non_target[:K]
    neg_vectors = np.array([r["vector"] for r in non_target], dtype=np.float32)
    neg_cosine_sims = np.array([r["metric"] for r in non_target], dtype=np.float32)
    neg_retrieval_ranks = np.arange(K, dtype=np.int32)
    return neg_vectors, neg_cosine_sims, neg_retrieval_ranks


def build_split(
    df: pd.DataFrame,
    split_name: str,
    config: BuildConfig,
    db: MilvusDB,
    output_dir: str,
) -> None:
    K = config.num_negatives
    queries, positives, negatives, neg_cosine_sims, neg_retrieval_ranks = [], [], [], [], []
    skipped = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building {split_name}"):
        query_vec = np.array(row["vector"], dtype=np.float32)
        target_doi = row["target_doi"]
        all_target_dois = set(row["all_target_dois"])
        pubdate_int = row["pubdate_int"]

        positive = get_best_positive(query_vec, target_doi, config.collection, db)
        if positive is None:
            skipped.append((row.get("source_doi", "?"), target_doi, "no positive chunks"))
            continue

        neg_result = get_negatives_with_metadata(
            query_vec, pubdate_int, all_target_dois, config.collection, db, K, config.milvus_top_k
        )
        if neg_result is None:
            skipped.append((row.get("source_doi", "?"), target_doi, f"< {K} negatives"))
            continue

        neg_vecs, neg_sims, neg_ranks = neg_result
        queries.append(query_vec)
        positives.append(positive.astype(np.float32))
        negatives.append(neg_vecs)
        neg_cosine_sims.append(neg_sims)
        neg_retrieval_ranks.append(neg_ranks)

    if skipped:
        print(f"\n[{split_name}] Skipped {len(skipped)} rows:")
        for src, tgt, reason in skipped[:10]:
            print(f"  source={src}  target={tgt}  reason={reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    N = len(queries)
    print(f"[{split_name}] Writing {N} rows to HDF5")

    queries_arr = np.array(queries, dtype=np.float32)
    positives_arr = np.array(positives, dtype=np.float32)
    negatives_arr = np.array(negatives, dtype=np.float32)
    neg_cosine_sims_arr = np.array(neg_cosine_sims, dtype=np.float32)
    neg_retrieval_ranks_arr = np.array(neg_retrieval_ranks, dtype=np.int32)

    out_path = os.path.join(output_dir, f"{split_name}.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("queries", data=queries_arr)
        f.create_dataset("positives", data=positives_arr)
        f.create_dataset("negatives", data=negatives_arr)
        f.create_dataset("neg_cosine_sims", data=neg_cosine_sims_arr)
        f.create_dataset("neg_retrieval_ranks", data=neg_retrieval_ranks_arr)
        f.attrs["embedder"] = config.embedder
        f.attrs["collection"] = config.collection
        f.attrs["dim"] = queries_arr.shape[1] if N > 0 else 0
        f.attrs["K"] = K
        f.attrs["build_date"] = str(date.today())
        f.attrs["dataset_source"] = config.dataset_source

    print(f"  queries:             {queries_arr.shape}")
    print(f"  positives:           {positives_arr.shape}")
    print(f"  negatives:           {negatives_arr.shape}")
    print(f"  neg_cosine_sims:     {neg_cosine_sims_arr.shape}")
    print(f"  neg_retrieval_ranks: {neg_retrieval_ranks_arr.shape}")
    print(f"  -> {out_path}")


def main(config_path: str) -> None:
    config = BuildConfig.from_yaml(config_path)
    os.makedirs(config.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create(config.embedder, device=device, normalize=config.normalize)
    db = MilvusDB()

    print(f"Loading data from {config.dataset_source}")
    df = pd.read_json(config.dataset_source, lines=True)

    # Convert pubdate "YYYY-MM-DD" → int YYYYMMDD for Milvus filter
    df["pubdate_int"] = df["pubdate"].astype(str).apply(pubdate_str_to_int)

    # Keep all citation DOIs before explode for negative filtering
    df["all_target_dois"] = df["citation_dois"]

    # One row per (query, target_doi)
    df["target_doi"] = df["citation_dois"]
    df = df.explode("target_doi").reset_index(drop=True)
    df = df.drop_duplicates(subset=["sent_no_cit", "target_doi"]).reset_index(drop=True)
    print(f"Total rows after explode+dedup: {len(df)}")

    # Embed all queries
    vectors = embed_in_batches(df["sent_no_cit"].tolist(), embedder)
    df["vector"] = list(vectors)

    # Train/val/test split
    train_df = df.sample(frac=config.train_frac, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n=== {split_name.upper()} ({len(split_df)} rows) ===")
        build_split(split_df, split_name, config, db, config.output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m citeline.nn.contrastive.build_dataset <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
