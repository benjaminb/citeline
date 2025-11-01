from abc import ABC, abstractmethod
import math
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
        doi_counts = results["doi"].value_counts().to_dict()

        # Create a list of counts corresponding to each result
        return results["doi"].map(doi_counts)


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


@Metric.register("bm25")
class BM25(Metric):
    def __init__(self, db=None):
        super().__init__(db=db)
        import bm25s

        self.bm25 = bm25s

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        corpus = results["text"].tolist()
        retriever = self.bm25.BM25(corpus=corpus, method="bm25l")
        retriever.index(self.bm25.tokenize(corpus))

        query_text = query["sent_no_cit"]
        _, scores = retriever.retrieve(self.bm25.tokenize(query_text), k=len(corpus))

        return pd.Series(scores[0], index=results.index)


@Metric.register("bm25_scratch")
class BM25Scratch(Metric):
    def __init__(self, db=None):
        super().__init__(db=db)
        import re

        self._WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in self._WORD_RE.findall(text or "")]

    def okapi_bm25_scores(self, query_text: str, doc_texts: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q_tokens = self.tokenize(query_text)
        docs_tokens = [self.tokenize(t) for t in doc_texts]
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
                # score += idf * ((f * (k1 + 1.0)) / (f + norm))
                score += idf * (f / (f + norm))  # Testing no k1 + 1 multiplier
            scores[i] = score
        return scores

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        corpus = results["text"].tolist()
        query_text = query["sent_no_cit"]
        scores = self.okapi_bm25_scores(query_text, corpus)
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


@Metric.register("position")
class Position(Metric):
    """
    Returns a score based on the original position in the results list.
    Higher positions (earlier in the list) get higher scores.

    This is useful for preserving the original ranking order (e.g., when results
    are interleaved from multiple query expansions) while still using RRF.

    The score is simply the reciprocal of position: 1/1, 1/2, 1/3, ...
    When used with rank(), this will produce ranks [1, 2, 3, ...].
    """

    def __call__(self, query: pd.Series, results: pd.DataFrame) -> pd.Series:
        # Return scores that decrease with position: [1.0, 0.5, 0.333, 0.25, ...]
        # When these are ranked with ascending=False, they'll produce ranks [1, 2, 3, ...]
        return pd.Series([1.0 / (i + 1) for i in range(len(results))], index=results.index)


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
