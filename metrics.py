from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

"""
A metric is a class who's call method takes a 'query' (pd.Series) and 'results' (pd.DataFrame) and 
computes a score for each result.
"""


class Metric(ABC):
    registry = {}

    @classmethod
    def register(cls, metric_name: str):
        """
        Register a metric class with a name.
        """

        def decorator(subclass):
            cls.registry[metric_name] = subclass
            subclass.name = metric_name
            return subclass

        return decorator

    @classmethod
    def get_metric(cls, metric_name: str, db=None):
        """
        Get a registered metric class by name.
        """
        if not metric_name in cls.registry:
            raise KeyError(f"Metric '{metric_name}' not found. Available metrics: {list(cls.registry.keys())}")
        metric_class = cls.registry.get(metric_name)
        return metric_class(db=db)

    def __init__(self, db=None):
        self.db = db

    @abstractmethod
    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        pass


@Metric.register("log_citations")
class LogCitations(Metric):

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        if "citation_count" not in results.columns:
            raise ValueError(
                f"Results DataFrame must contain 'citation_count' column. It has columns {results.columns}"
            )

        citation_counts = results["citation_count"]
        # TODO: Consider normalizing by years since publication, since older papers have more time to accumulate citations
        # TODO: would base 10 be more appropriate?
        return np.log1p(citation_counts)  # log(1 + citation_count) to handle zero citations


@Metric.register("recency")
class Recency(Metric):

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        """
        Returns the -log of years since publication
        """
        if "pubdate" not in results.columns:
            raise ValueError("Results DataFrame must contain 'pubdate' column")

        query_date = pd.to_datetime(str(query["pubdate"]), format="%Y%m%d")
        results_pubdates = results["pubdate"].apply(pd.to_datetime, format="%Y%m%d")
        # Calculate years since publication
        days_since_pub = (query_date - results_pubdates).dt.days
        years_since_pub = days_since_pub / 365.25
        assert (
            years_since_pub >= 0
        ).all(), f"Found negative years_since_pub values: {years_since_pub[years_since_pub < 0].tolist()}"
        return -np.log1p(years_since_pub)

@Metric.register("similarity")
class Similarity(Metric):
    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        """
        Assumes that the 'distance' key in the entity represents a similarity score,
         e.g. inner product or cosine similarity. This metric passes through that 
        distance key value
        """
        if "distance" not in results.columns:
            raise ValueError("Results DataFrame must contain 'distance' column")

        return results["distance"]


def get_modernbert_crossencoder() -> callable:
    """
    Returns a reranker that uses a pre-trained ModernBERT model to score results.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("tomaarsen/reranker-ModernBERT-large-gooaq-bce")
    # model = CrossEncoder("tomaarsen/reranker-ModernBERT-base-gooaq-bce") # smaller model for faster inference

    def modernbert_crossencoder(query: pd.Series, results: pd.DataFrame, db=None) -> pd.Series:
        """
        Reranks results using the ModernBERT model.
        """
        pairs = [[query["sent_no_cit"], row["text"]] for _, row in results.iterrows()]
        scores = model.predict(pairs)
        return scores

    return modernbert_crossencoder


# class RankFuser:
#     """
#     A class that produces a weighted sum of scores from multiple scoring functions,
#     then uses those weights to rerank a set of results
#     """

#     def __init__(self, config: dict[str, float]):
#         """
#         Initializes the RankFuser with a configuration dictionary that maps scoring function names to their weights.

#         Args:
#             config (dict[str, float]): A dictionary where keys are scoring function names and values are their respective weights.
#         """
#         self.config = config
#         self.metrics = [get_metric(name) for name in config.keys()]
#         self.weights = list(config.values())

#     def __call__(self, query: pd.Series, results: pd.DataFrame, db=None) -> pd.DataFrame:
#         """
#         Reranks the results DataFrame based on the weighted sum of scores from the configured metrics.

#         Args:
#             query (pd.Series): The query for which results are being reranked.
#             results (pd.DataFrame): The DataFrame containing results to be reranked.

#         Returns:
#             pd.DataFrame: The reranked results DataFrame.
#         """
#         scores = [metric(query, results, db) for metric in self.metrics]
#         weighted_scores = sum(weight * score for weight, score in zip(self.weights, scores))
#         results["weighted_score"] = weighted_scores
#         return results.sort_values("weighted_score", ascending=False).reset_index(drop=True)


# METRICS = {
#     "cosine_similarity": get_cosine_similarity_metric,
#     "recency": get_recency_metric,
#     "log_citations": get_log_citations_metric,
#     "modernbert": get_modernbert_crossencoder,
# }


# def get_metric(name: str) -> callable:
#     if name in METRICS:
#         return METRICS[name]()
#     raise ValueError(f"Unknown metric: {name}")


def main():
    queries = [
        "If you want to go to France's capital go to Paris",
        "There are larger planets than Mercury",
    ]


if __name__ == "__main__":
    main()
