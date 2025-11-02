"""
Verify that using 'position' metric instead of 'similarity' matches the notebook approach
"""

import json
import pandas as pd
from citeline.rank_fuser import RankFuser
from compare_rrf_methods import notebook_rrf_rerank

search_results_file = "experiments/multiple_query_expansion/results/search_results.jsonl"

print("Testing 'position' metric vs notebook approach")
print("=" * 80)

num_matches = 0

with open(search_results_file, "r") as f:
    for i, line in enumerate(f):
        if i >= 20:  # Test 20 queries
            break

        data = json.loads(line)
        query = pd.Series(data["record"])
        results_df = pd.DataFrame(data["results"])

        # Method 1: RankFuser with 'position' metric
        config_position = {"bm25_scratch": 0.42, "position": 0.58}
        rf = RankFuser(config_position, rrf_k=60)
        reranked_rf = rf._rerank_single(query, results_df.copy())
        top10_rf = reranked_rf["doi"].tolist()[:10]

        # Method 2: Notebook
        reranked_nb = notebook_rrf_rerank(query, results_df.copy(), w_dataset=0.58, w_bm25=0.42)
        top10_nb = reranked_nb["doi"].tolist()[:10]

        # Compare
        if top10_rf == top10_nb:
            num_matches += 1
            print(f"Query {i}: ✓ Match")
        else:
            print(f"Query {i}: ✗ MISMATCH")
            for j in range(min(5, len(top10_rf))):
                match_str = "✓" if top10_rf[j] == top10_nb[j] else "✗"
                print(f"  {j+1}. {match_str} RF: {top10_rf[j][:25]}...  NB: {top10_nb[j][:25]}...")

print("\n" + "=" * 80)
print(f"Summary: {num_matches}/{i+1} queries match")

if num_matches == i + 1:
    print("\n✓ SUCCESS! All queries match. The 'position' metric correctly replicates")
    print("  the notebook's behavior for interleaved results.")
else:
    print(f"\n✗ {i+1-num_matches} queries still don't match. Further investigation needed.")
