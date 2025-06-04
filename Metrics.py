"""
The Metric class is an interface for rank fusion / reranking.
A Metric is a function taking query and its corresponding DB results as input and returns a float score.

To more efficiently use embedders we use batches of queries and results for input rather than single queries.
Metric: [(query, VectorQueryResult)] -> float
"""

from database.database import VectorQueryResult
from typing import Callable


class Metric:
    def __init__(self, name: str, function: Callable):
        self.name = name
        # Confirm that the function takes two arguments: queries and records
        if not callable(function) or function.__code__.co_argcount != 2:
            raise ValueError(
                "Function must be callable and take exactly two arguments: queries and records."
            )
        self.function = function

    def __call__(self, queries: list[str], records: list[VectorQueryResult]) -> list[list[float]]:
        """
        Compute the metric score for a given query and its results.

        Precondition: The queries and records must be of the same length.

        :param query: The query string.
        :param results: A list of results corresponding to the query.
        :return: A 2D list of float scores; each row corresponds to a query and the columns correspond to the scores for each record.
        """
        return self.function(queries, records)



def get_roberta_entailment_function():
    """
    This function creates a closure for the Roberta NLI model and tokenizer so that
    they are loaded once when roberta_entailment is instantiated, rather than every time it is called.
    """
    import torch
    from torch import nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "cross-encoder/nli-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    def roberta_entailment(queries, records) -> list[list[float]]:
        scores = []
        for query, record_list in zip(queries, records):
            premises = [record.chunk for record in record_list]
            hypotheses = [query] * len(premises)
            inputs = tokenizer(
                premises, hypotheses, return_tensors="pt", truncation=True, padding=True
            ).to(device)
            record_scores = []
            for record in record_list:
                inputs = tokenizer(record.chunk, query, return_tensors="pt", truncation=True).to(
                    device
                )
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1).squeeze().tolist()
                record_scores.append(probs[1])
            scores.append(record_scores)
        return scores

    return roberta_entailment


def get_deberta_entailment_function():
    """
    This function creates a closure for the DeBERTa NLI model and tokenizer so that
    they are loaded once when deberta_entailment is instantiated, rather than every time it is called.
    """
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    model = CrossEncoder("cross-encoder/nli-deberta-v3-base", device=device)

    def deberta_entailment(queries, records) -> list[list[float]]:
        scores = []
        for query, record_list in zip(queries, records):
            premises = [record.chunk for record in record_list]
            hypotheses = [query] * len(premises)
            inputs = list(zip(premises, hypotheses))
            record_scores = model.predict(inputs, apply_softmax=True)
            inference_scores = record_scores[:, 1].tolist()
            print(f"Inference scores: {inference_scores}")
            scores.append(inference_scores)
        return scores

    return deberta_entailment


def main():
    entailment_metric = Metric("roberta_entailment", get_roberta_entailment_function())
    queries = [
        "If you want to go to France's capital go to Paris",
        "There are larger planets than Mercury",
    ]
    records = [
        VectorQueryResult(
            chunk="The capital of France is Paris.",
            chunk_id=1,
            doi=None,
            title=None,
            abstract=None,
            pubdate=None,
            distance=None,
        ),
        VectorQueryResult(
            chunk="Jupiter is the largest planet in our solar system.",
            chunk_id=1,
            doi=None,
            title=None,
            abstract=None,
            pubdate=None,
            distance=None,
        ),
    ]
    # scores = entailment_metric(queries, [records, records])
    # print(scores)  # Should print a list of scores for each query-record pair

    deberta_metric = Metric("deberta_entailment", get_deberta_entailment_function())
    scores_deberta = deberta_metric(queries, [records, records])
    print(scores_deberta)  # Should print a list of scores for each query-record pair


if __name__ == "__main__":
    main()
