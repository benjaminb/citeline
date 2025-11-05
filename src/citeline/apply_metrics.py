import argparse
import pandas as pd
import json
import yaml
from time import time
from tqdm import tqdm
from citeline.database.milvusdb import MilvusDB
from citeline.metrics import Metric


def argument_parser():
    """
    example usage: python apply_metrics.py <path to yaml config file>
    """
    parser = argparse.ArgumentParser(description="Apply metrics to search results")
    parser.add_argument("config", help="Path to the YAML config file")
    # add optional --dry-run flag
    parser.add_argument("--dry-run", action="store_true", help="Run on 2 rows without writing to file")
    return parser.parse_args()


def check_file_preconditions(filepath: str):
    """Check that the search results file meets preconditions.

    Preconditions
    1. JSONL file of search results needs to have form [{record: {...}, results: [{r1}, {r2}, ...]} , ...]
    2. The record needs to have an 'query*' field
    3. The results field needs to be a list of dicts
    """
    assert filepath.endswith(".jsonl"), "File must be a .jsonl file"
    with open(filepath, "r") as f:
        first_line = json.loads(f.readline())

        # Checks on the query record
        assert "record" in first_line, "Each line must have a 'record' field"
        record = first_line["record"]
        assert isinstance(record, dict), "'record' field must be a dict"
        query_columns = [key for key in record.keys() if key.startswith("query")]
        assert len(query_columns) > 0, "Record must have at least one column starting with 'query'"
        assert any(key.startswith("query") for key in first_line["record"].keys()), "Record must have a 'query*' field"

        # Checks on the search results
        assert "results" in first_line, "Each line must have a 'results' field"
        assert isinstance(first_line["results"], list), "'results' field must be a list"
        assert all(isinstance(result, dict) for result in first_line["results"]), "All results must be dicts"
    print("File preconditions met.")


def main():
    start = time()

    args = argument_parser()
    path_to_config = args.config
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    # Get config values
    infile_path = config["infile"]
    outfile_path = config["outfile"]
    metric_names = config["metrics"]

    # Currently no metrics use db
    metrics = [Metric.get_metric(name, db=None) for name in metric_names]

    check_file_preconditions(infile_path)

    if args.dry_run:
        print("Dry run mode: processing 2 rows without writing to file.")
        with open(infile_path, "r") as infile:
            for i, line in enumerate(infile):
                if i >= 2:
                    break
                row: dict = json.loads(line)  # {record: {...}, results: [{r1}, {r2}, ...]}
                record = pd.Series(row["record"])
                results = pd.DataFrame(row["results"])

                print(f"\nProcessing record {i+1}:")
                for metric in metrics:
                    scores = metric(record, results)
                    colname = f"score_{metric.name}"
                    results[colname] = scores
                    print(f"Applied metric '{metric.name}':")
                    print(results[[colname]].head())
        end = time()
        print(f"\nDry run total time: {end - start:.2f} seconds")
        return

    with open(infile_path, "r") as infile, open(outfile_path, "w") as outfile:
        for i, line in enumerate(tqdm(infile)):
            row: dict = json.loads(line)  # {record: {...}, results: [{r1}, {r2}, ...]}
            record = pd.Series(row["record"])
            results = pd.DataFrame(row["results"])

            for metric in metrics:
                scores = metric(record, results)
                colname = f"score_{metric.name}"
                results[colname] = scores

            row["results"] = results.to_dict(orient="records")
            outfile.write(json.dumps(row) + "\n")
            if i % 1000 == 0:  # Flush every 1000 lines
                outfile.flush()

    end = time()
    print(f"Total time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
