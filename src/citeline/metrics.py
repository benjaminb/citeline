from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch

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
    def __call__(self, query: pd.Series, results: list[dict]) -> pd.Series:
        pass


@Metric.register("reciprocal_rank")
class ReciprocalRank(Metric):
    """
    Returns the rank of each result (1/1, 1/2, 1/3, ...)

    Can also be used to test the reranking pipeline doesn't break order inadvertently.
    """

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        return pd.Series([1 / n for n in range(1, len(results) + 1)], index=results.index)

@Metric.register("retrieval_count")
class RetrievalCount(Metric):
    """
    Assigns to each result entity the number of times that doi appears in the results
    """
    def __call__(self, query, results):
        # Iterate over the results and build a dict mapping doi to count
        # results_df = pd.DataFrame(results)
        doi_counts = results['doi'].value_counts().to_dict()

        # Create a list of counts corresponding to each result
        return results['doi'].map(doi_counts)
    
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


@Metric.register("negative_log_years_old")
class NegativeLogYearsOld(Metric):

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


@Metric.register("bge_reranker")
class BGEReranker(Metric):
    def __init__(self, db=None):
        super().__init__(db=db)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # self.model_name = "BAAI/bge-reranker-large"
        self.model_name = "BAAI/bge-reranker-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        pairs = [[query.sent_no_cit, row.text] for row in results.itertuples()]
        with torch.no_grad():
            batch_size = 16
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                # BGE reranker base model has max_position_embeddings=514, we set to 512 to be safe
                batch_inputs = self.tokenizer(
                    batch_pairs, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.model.device)
                outputs = self.model(**batch_inputs, return_dict=True)
                logits = outputs.logits.view(
                    -1,
                )
                scores.extend(logits.detach().cpu().numpy().tolist())
        return pd.Series(scores, index=results.index)


@Metric.register("roberta_nli")
class RobertaNLI(Metric):
    def __init__(self, db=None):
        super().__init__(db=db)

        # Set up model
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.model = CrossEncoder("sentence-transformers/nli-roberta-base", device=device)

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        if "text" not in results.columns:
            raise ValueError("Results DataFrame must contain 'text' column")

        texts = results["text"].tolist()
        input_pairs = [(query["sent_no_cit"], text) for text in texts]
        with torch.no_grad():
            scores = self.model.predict(input_pairs).tolist()
        return pd.Series(scores, index=results.index)


@Metric.register("similarity")
class Similarity(Metric):
    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        """
        NOTE: Assumes that the 'distance' key in the entity represents a similarity score,
        e.g. inner product or cosine similarity. This metric passes through that
        distance key value
        """
        if "metric" not in results.columns:
            raise ValueError("Results DataFrame must contain 'metric' column")

        return results["metric"]


def main():
    docs = pd.DataFrame(
        {
            "text": [
                "If you want to go to France's capital go to Paris",
                "There are larger planets than Mercury",
            ]
            * 8
        }
    )
    query = pd.Series({"sent_no_cit": "What is the capital of France?"})
    bge = BGEReranker()
    scores = bge(query, docs)
    print(scores)


if __name__ == "__main__":
    main()
