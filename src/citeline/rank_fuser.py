# import pandas as pd
# from citeline.metrics import Metric
# from tqdm import tqdm

# """
# The experiments, when asked to output all search results, store in a parquet format that can read into a pandas DataFrame.
# For an experiment on n records, the parquet->DataFrame will have n rows and 2 columns:
#   - 'record': the dict representing the original record from that dataset, with keys:
#     'citation_dois', 'expanded_query', 'pubdate', 'resolved_bibcodes', 'sent_cit_masked', 'sent_idx', 'sent_no_cit', 'sent_original', 'source_doi'
#   - 'results': a list of dicts representing the entities returned from the database search, in 'metric' order
#     contains keys: 'citation_count', 'doi', 'metric', 'pubdate', 'text'
#     NOTE: remember that in Milvus metrics IP and COSINE are 'higher is better' while L2 is 'lower is better'
# """


# class RankFuser:
#     """
#     A class that produces a weighted sum of scores from multiple scoring functions,
#     then uses those weights to rerank a set of results
#     """

#     def __init__(self, config: dict[str, float], top_k, rrf_k=60):
#         """
#         Initializes the RankFuser with a configuration dictionary that maps scoring function names to their weights.

#         Args:
#             config (dict[str, float]): A dictionary where keys are metric names and values are their respective weights.
#                                         Find the metric names in the decorators preceding each Metric subclass in metrics.py
#             top_k (int): The number of top results to consider for reranking. If less than number of results, truncates search results to top_k before reranking.
#             rrf_k (int): The damping parameter for Reciprocal Rank Fusion (RRF). Default is 60.
#         """
#         self.config = config
#         self.metrics = [Metric.get_metric(name) for name in config.keys()]
#         self.weights = list(config.values())
#         self.top_k = top_k
#         self.rrf_k = rrf_k

#     def rerank(self, data: list[dict]) -> list[dict[str, pd.Series | pd.DataFrame]]:
#         """
#         Expects a list of dictionaries, each containing "record" and "results" keys.
#         - "record": the dict representing the original query record
#         - "results": ordered list of dicts representing the search results

#         Returns:
#             list[dict]: The data with reranked results
#             dict:
#             - "record": pd.Series of the original query record
#             - "results": pd.DataFrame of the reranked search results
#         """
#         rows = []

#         for row in tqdm(data, desc="Reranking results"):
#             query = pd.Series(row["record"])
#             results = pd.DataFrame(row["results"][: self.top_k])  # Truncate to top_k before reranking
#             reranked_results = self._rerank_single(query, results)
#             rows.append({"record": query, "results": reranked_results})

#         return rows

#     def _rerank_single(self, query: pd.Series, results: pd.DataFrame) -> pd.DataFrame:
#         """
#         Reranks the results DataFrame based on the weighted sum of scores from the configured metrics.

#         Args:
#             query (pd.Series): The input record for which results are being reranked.
#             results (pd.DataFrame): The DataFrame containing results to be reranked.

#         Returns:
#             pd.DataFrame: The reranked results.
#         """
#         results_df = results.copy()
#         results_df["rrf"] = 0

#         # For each metric, compute scores, get ranks, and build weighted RRF
#         for metric, weight in zip(self.metrics, self.weights):
#             scores = metric(query, results_df)
#             results_df[metric.name] = scores
#             ranks = scores.rank(ascending=False, method="first")
#             results_df["rrf"] += weight / (ranks + self.rrf_k)

#         # Sort by the weighted score in descending order
#         return results_df.sort_values("rrf", ascending=False).reset_index(drop=True)
