import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_stat_matrices(data: list[dict[str, pd.Series | pd.DataFrame]]) -> list:

    # NOTE: Assumes all results have the same top k / number of results
    top_k = len(data[0]["results"])
    hitrate_matrix = np.zeros((len(data), top_k))
    iou_matrix = np.zeros((len(data), top_k))
    recall_matrix = np.zeros((len(data), top_k))

    for i, row in tqdm(enumerate(data), total=len(data), desc="Computing statistics"):
        results = row["results"]
        stats = compute_individual_stats(row["record"], results)
        hitrate_matrix[i] = stats["hitrate"]
        iou_matrix[i] = stats["iou"]
        recall_matrix[i] = stats["recall"]

    return {"hitrates": hitrate_matrix, "ious": iou_matrix, "recalls": recall_matrix}


def compute_individual_stats(example: pd.Series, results: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Computes statistics for a single example and its results, returning
    a dictionary { "hitrate": np.ndarray, "iou": np.ndarray, "recall": np.ndarray }
    """
    hit_dois = set()
    target_dois = example["citation_dois"]
    retrieved_dois = set()
    union_dois = set(target_dois)
    hitrate_at_k = np.zeros(len(results))
    iou_at_k = np.zeros(len(results))
    recall_at_k = np.zeros(len(results))

    for i, result in enumerate(results.itertuples()):
        # Add the retrieved DOI at this rank to relevant sets
        doi = result.doi
        retrieved_dois.add(doi)
        union_dois.add(doi)
        if doi in target_dois:
            hit_dois.add(doi)

        # Compute stats
        recall = len(hit_dois) / len(target_dois) if target_dois else 0
        hitrate = int(recall > 0)
        iou = len(hit_dois) / len(union_dois) if union_dois else 0

        # Store stats
        recall_at_k[i] = recall
        hitrate_at_k[i] = hitrate
        iou_at_k[i] = iou

    return {"hitrate": hitrate_at_k, "iou": iou_at_k, "recall": recall_at_k}


def compute_averages(stat_matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Computes the average statistics across all examples for each metric.
    """
    return {
        "hitrate": np.mean(stat_matrices["hitrates"], axis=0),
        "iou": np.mean(stat_matrices["ious"], axis=0),
        "recall": np.mean(stat_matrices["recalls"], axis=0),
    }


def compute_statistics(data: list[dict[str, list[dict] | pd.Series | pd.DataFrame]]) -> dict[str, np.ndarray]:
    """
    Computes the overall statistics from the raw or reranked results data.

    The input data is a list of dictionaries: {"record": {...}, "results": [...]} where
    - record: pd.Series for example input record
    - results could be a list of dictionaries for search results, or a pd.DataFrame if the data came from a reranker
    """

    if isinstance(data[0]["results"], list):
        data = convert_data_to_dfs(data)
    stat_matrices = compute_stat_matrices(data)
    return compute_averages(stat_matrices)


def convert_data_to_dfs(data: list[dict[str, dict | list[dict]]]) -> dict[str, float]:
    """
    Converts 'results' to a list of pd.DataFrame
    """
    converted_data = []
    for dictionary in data:
        results = dictionary["results"]
        df = pd.DataFrame(results)
        converted_data.append({"record": pd.Series(dictionary["record"]), "results": df})
    return converted_data


def stat_diffs(old_stats: dict[str, np.ndarray], new_stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Computes the improvement between two sets of statistics. Assumes the
    statistics are computed over the same set of examples.
    """
    return {
        "hitrate": new_stats["hitrate"] - old_stats["hitrate"],
        "iou": new_stats["iou"] - old_stats["iou"],
        "recall": new_stats["recall"] - old_stats["recall"],
    }
