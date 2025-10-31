"""
Compare the RankFuser RRF implementation with the notebook RRF implementation
to understand why they produce different hitrate@50 values.
"""
import json
import numpy as np
import pandas as pd
import re
import math
from citeline.rank_fuser import RankFuser

# Tokenizer (from notebook)
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def tokenize(text):
    return [token.lower() for token in _WORD_RE.findall(text or "")]

def okapi_bm25_scores(query_text, doc_texts, k1=1.5, b=0.75):
    """BM25 implementation from notebook"""
    q_tokens = tokenize(query_text)
    docs_tokens = [tokenize(t) for t in doc_texts]
    N = len(docs_tokens)
    if N == 0:
        return np.zeros((0,), dtype=float)

    doc_len = np.array([len(toks) for toks in docs_tokens], dtype=float)
    avgdl = doc_len.mean() if N else 0.0

    tf_list = []
    for toks in docs_tokens:
        tf = {}
        for token in toks:
            tf[token] = tf.get(token, 0) + 1
        tf_list.append(tf)

    query_terms = set(q_tokens)
    df = {token: sum(1 for tf in tf_list if token in tf) for token in query_terms}
    scores = np.zeros(N, dtype=float)

    for i, tf in enumerate(tf_list):
        dl = doc_len[i]
        norm = k1 * (1.0 - b + b * (dl / avgdl)) if avgdl > 0 else k1
        score = 0.0
        for token in query_terms:
            f = tf.get(token, 0.0)
            if f <= 0:
                continue
            idf = math.log(1.0 + (N - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            score += idf * ((f * (k1 + 1.0)) / (f + norm))
        scores[i] = score
    return scores

def notebook_rrf_rerank(query, results_df, w_dataset=0.58, w_bm25=0.42, rrf_k=60):
    """Rerank using notebook's approach"""
    # Dataset ranks = position [1, 2, 3, ...]
    dataset_ranks = np.arange(1, len(results_df) + 1, dtype=float)

    # Compute BM25 scores
    doc_texts = results_df['text'].tolist()
    query_text = query['sent_no_cit']
    bm25_scores = okapi_bm25_scores(query_text, doc_texts)

    # Rank BM25 scores
    order = np.argsort(-bm25_scores, kind="mergesort")
    bm25_ranks = np.empty_like(bm25_scores, dtype=float)
    bm25_ranks[order] = np.arange(1, len(bm25_scores) + 1, dtype=float)

    # Weighted RRF
    ds_scores = 1.0 / (rrf_k + dataset_ranks)
    bm_scores = 1.0 / (rrf_k + bm25_ranks)
    combined = w_dataset * ds_scores + w_bm25 * bm_scores

    # Sort by combined score
    sorted_idx = np.argsort(-combined)
    reranked_df = results_df.iloc[sorted_idx].reset_index(drop=True)
    reranked_df['notebook_rrf'] = combined[sorted_idx]
    reranked_df['dataset_rank'] = dataset_ranks[sorted_idx]
    reranked_df['bm25_rank'] = bm25_ranks[sorted_idx]

    return reranked_df

def rankfuser_rerank(query, results_df, config={'bm25_scratch': 0.42, 'similarity': 0.58}, k=60):
    """Rerank using RankFuser approach"""
    rf = RankFuser(config, k=k)
    return rf._rerank_single(query, results_df)

def main():
    search_results_file = 'experiments/multiple_query_expansion/results/search_results.jsonl'

    print("Comparing RRF implementations on first 10 queries...")
    print("="*80)

    num_different = 0

    with open(search_results_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break

            data = json.loads(line)
            query = pd.Series(data['record'])
            results_df = pd.DataFrame(data['results'])

            # Method 1: RankFuser
            reranked_rf = rankfuser_rerank(query, results_df.copy())
            top10_rf = reranked_rf['doi'].tolist()[:10]

            # Method 2: Notebook
            reranked_nb = notebook_rrf_rerank(query, results_df.copy())
            top10_nb = reranked_nb['doi'].tolist()[:10]

            # Compare
            if top10_rf != top10_nb:
                num_different += 1
                print(f"\nQuery {i}: TOP 10 DIFFERS")
                print(f"  Original first DOI: {results_df.iloc[0]['doi']}")

                # Find first difference
                for j in range(10):
                    if top10_rf[j] != top10_nb[j]:
                        print(f"  First diff at position {j}:")
                        print(f"    RankFuser: {top10_rf[j]}")
                        print(f"    Notebook:  {top10_nb[j]}")
                        break

                # Show some scores for debugging
                print(f"\n  RankFuser scores for top 3:")
                for idx in range(3):
                    row = reranked_rf.iloc[idx]
                    print(f"    {idx+1}. doi={row['doi'][:20]}... rrf={row['rrf']:.6f} sim={row['similarity']:.4f} bm25={row.get('bm25_scratch', 'N/A')}")

                print(f"\n  Notebook scores for top 3:")
                for idx in range(3):
                    row = reranked_nb.iloc[idx]
                    print(f"    {idx+1}. doi={row['doi'][:20]}... rrf={row['notebook_rrf']:.6f} ds_rank={row['dataset_rank']:.0f} bm25_rank={row['bm25_rank']:.0f}")
            else:
                print(f"Query {i}: ✓ Match")

    print("\n" + "="*80)
    print(f"Summary: {num_different}/{i+1} queries had different top-10 rankings")

    if num_different == 0:
        print("\nAll queries match! The implementations are equivalent on these queries.")
        print("The hitrate difference must come from something else (different data, aggregation, etc.)")
    else:
        print(f"\n{num_different} queries have different rankings. This explains the hitrate difference!")

if __name__ == "__main__":
    main()
