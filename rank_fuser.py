import pandas as pd
from metrics import get_metric
"""
The experiments, when asked to output all search results, store in a parquet format that can read into a pandas DataFrame.
For an experiment on n records, the parquet->DataFrame will have n rows and 2 columns:
  - 'record': the dict representing the original record from that dataset, with keys:
    'citation_dois', 'expanded_query', 'pubdate', 'resolved_bibcodes', 'sent_cit_masked', 'sent_idx', 'sent_no_cit', 'sent_original', 'source_doi'
  - 'results': a list of dicts representing the entities returned from the database search, in 'metric' order
    contains keys: 'citation_count', 'doi', 'metric', 'pubdate', 'text'
    NOTE: remember that in Milvus metrics IP and COSINE are 'higher is better' while L2 is 'lower is better'
"""

class RankFuser():
    """
    A class that produces a weighted sum of scores from multiple scoring functions,
    then uses those weights to rerank a set of results
    """
    def __init__(self, config: dict[str, float]):
        """
        Initializes the RankFuser with a configuration dictionary that maps scoring function names to their weights.
        
        Args:
            config (dict[str, float]): A dictionary where keys are scoring function names and values are their respective weights.
        """
        self.config = config
        self.metrics = [get_metric(name) for name in config.keys()]
        self.weights = list(config.values())

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.DataFrame:
        """
        Reranks the results DataFrame based on the weighted sum of scores from the configured metrics.
        
        Args:
            query (pd.Series): The query for which results are being reranked.
            results (pd.DataFrame): The DataFrame containing results to be reranked.
        
        Returns:
            pd.DataFrame: The reranked results DataFrame.
        """

        # Calculate scores for each metric
        scores = [metric(query, results) for metric in self.metrics]
        
        # Compute the weighted sum of scores
        weighted_scores = sum(weight * score for weight, score in zip(self.weights, scores))
        results["weighted_score"] = weighted_scores
        
        # Sort by the weighted score in descending order
        return results.sort_values("weighted_score", ascending=False).reset_index(drop=True)
