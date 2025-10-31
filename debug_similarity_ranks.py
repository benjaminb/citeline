"""
Debug why RankFuser and notebook produce different results.
Check if the 'similarity' ranks differ from position ranks.
"""
import json
import pandas as pd
from citeline.metrics import Metric

search_results_file = 'experiments/multiple_query_expansion/results/search_results.jsonl'

print("Checking if similarity ranks == position ranks [1, 2, 3, ...]")
print("="*80)

num_mismatched = 0

with open(search_results_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break

        data = json.loads(line)
        query = pd.Series(data['record'])
        results_df = pd.DataFrame(data['results'])

        # Get similarity scores from the 'metric' column
        similarity_scores = results_df['metric']

        # Rank them (as RankFuser does)
        similarity_ranks = similarity_scores.rank(ascending=False, method="first")

        # Position ranks (as notebook uses)
        position_ranks = pd.Series(range(1, len(results_df) + 1), index=results_df.index, dtype=float)

        # Check if they match
        if not (similarity_ranks == position_ranks).all():
            num_mismatched += 1
            print(f"\nQuery {i}: RANKS DIFFER!")
            print(f"  First 10 similarity values: {similarity_scores.head(10).tolist()}")
            print(f"  Are they sorted descending? {(similarity_scores.diff().dropna() <= 0).all()}")

            # Find where they differ
            diff_idx = (similarity_ranks != position_ranks)
            if diff_idx.sum() > 0:
                print(f"  Number of positions with different ranks: {diff_idx.sum()}")
                # Show first few mismatches
                first_mismatch_idx = diff_idx.idxmax()
                print(f"  First mismatch at position {first_mismatch_idx}:")
                print(f"    Similarity rank: {similarity_ranks.iloc[first_mismatch_idx]}")
                print(f"    Position rank: {position_ranks.iloc[first_mismatch_idx]}")
                print(f"    Similarity value: {similarity_scores.iloc[first_mismatch_idx]}")

                # Check for duplicate similarities around this position
                sim_val = similarity_scores.iloc[first_mismatch_idx]
                duplicates = (similarity_scores == sim_val).sum()
                if duplicates > 1:
                    print(f"    This similarity value appears {duplicates} times (TIES!)")
        else:
            print(f"Query {i}: ✓ Ranks match")

print("\n" + "="*80)
print(f"Summary: {num_mismatched}/{i+1} queries have different similarity ranks vs position ranks")

if num_mismatched > 0:
    print("\nThis is the root cause! The Similarity metric re-ranks the values,")
    print("which differs from just using position [1, 2, 3, ...] when there are ties")
    print("or when the results are not perfectly sorted by similarity.")
