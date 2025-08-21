"""
A metric is a function that takes a query (pd.Series) and its results (pd.DataFrame) and returns a list of scores,
each score reflecting the similarity or relevance of the result row to the query
"""

import pandas as pd


def get_cosine_similarity_metric() -> callable:
    def cosine_similarity(query: pd.Series, results: pd.DataFrame, db=None) -> pd.Series:
        """
        Computes cosine similarity as 1 - distance for each result in the results DataFrame.
        """
        return 1 - results["distance"]

    return cosine_similarity


def get_recency_metric() -> callable:
    """
    Returns a metric that computes a score based on the recency of a result publication to the
    query date. To model a bias for recent publications, we use -log(years since publication + 1).
    The + 1 avoids a blowup for publications in the same year as the query.
    """
    import numpy as np

    def log_recency(query: pd.Series, results: pd.DataFrame, db=None) -> pd.Series:
        """
        Computes recency score based on the publication date of each result.
        """
        if "pubdate" not in results.columns:
            raise ValueError("Results DataFrame must contain 'pubdate' column")

        # Calculate years since publication
        query_date = query["pubdate"]
        days_since_pub = (query_date - results["pubdate"]).apply(lambda x: x.days)
        years_since_pub = days_since_pub / 365.25
        assert (
            years_since_pub >= 0
        ).all(), f"Found negative years_since_pub values: {years_since_pub[years_since_pub < 0].tolist()}"
        return -np.log(years_since_pub + 1)

    return log_recency


def get_log_citations_metric() -> callable:
    """
    Returns a metric that computes a score based on the log of citation counts.
    """
    import numpy as np

    def log_citations_metric(query: pd.Series, results: pd.DataFrame, db=None) -> pd.Series:
        """
        Computes log citation score for each result.
        """

        citation_counts = results.citation_counts.tolist()
        # TODO: Consider normalizing by years since publication, since older papers have more time to accumulate citations
        # TODO: would base 10 be more appropriate?
        return np.log1p(citation_counts)  # log(1 + citation_count) to handle zero citations

    return log_citations_metric


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


class RankFuser:
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

    def __call__(self, query: pd.Series, results: pd.DataFrame, db=None) -> pd.DataFrame:
        """
        Reranks the results DataFrame based on the weighted sum of scores from the configured metrics.

        Args:
            query (pd.Series): The query for which results are being reranked.
            results (pd.DataFrame): The DataFrame containing results to be reranked.

        Returns:
            pd.DataFrame: The reranked results DataFrame.
        """
        scores = [metric(query, results, db) for metric in self.metrics]
        weighted_scores = sum(weight * score for weight, score in zip(self.weights, scores))
        results["weighted_score"] = weighted_scores
        return results.sort_values("weighted_score", ascending=False).reset_index(drop=True)


METRICS = {
    "cosine_similarity": get_cosine_similarity_metric,
    "recency": get_recency_metric,
    "log_citations": get_log_citations_metric,
    "modernbert": get_modernbert_crossencoder,
}


def get_metric(name: str) -> callable:
    if name in METRICS:
        return METRICS[name]()
    raise ValueError(f"Unknown metric: {name}")


def main():
    from database.database import VectorSearchResult

    queries = [
        "If you want to go to France's capital go to Paris",
        "There are larger planets than Mercury",
    ]
    single_query = pd.Series({"sent_no_cit": queries[0], "sent_idx": 0, "pubdate": pd.Timestamp("2023-10-01")})
    results = pd.DataFrame(
        [
            VectorSearchResult(
                text="The capital of France is Paris.",
                doi=None,
                pubdate=None,
                distance=0.25,
            ),
            VectorSearchResult(
                text="Jupiter is the largest planet in our solar system.",
                doi=None,
                pubdate=None,
                distance=0.75,
            ),
        ]
    )
    fn = get_cosine_similarity_metric()

    scores = fn(single_query, results)
    print(f"Query: {single_query}")
    print(f"Cosine similarities: {scores}")  # Should print a Series with scores for each result

    fn = get_modernbert_crossencoder()

    scores = fn(single_query, results)

    print(f"Query: {single_query}")
    print(f"ModernBERT scores: {scores}")  # Should print a Series with scores for each result


if __name__ == "__main__":
    main()
