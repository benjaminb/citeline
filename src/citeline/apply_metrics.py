import argparse
import os
import tempfile
import pandas as pd
import json
import torch
import yaml
from multiprocessing import Pool, Value, Lock
from time import time
from tqdm import tqdm
from citeline.metrics import Metric


def argument_parser():
    """
    example usage: python apply_metrics.py <path to yaml config file>
    """
    parser = argparse.ArgumentParser(description="Apply metrics to search results")
    parser.add_argument("config", help="Path to the YAML config file")
    # add optional --dry-run flag
    parser.add_argument("--dry-run", action="store_true", help="Run on 2 rows without writing to file")

    # add optional --multi-cuda flag
    parser.add_argument("--multi-cuda", action="store_true", help="Use multiple CUDA devices if available")
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Path to write output file, if not specified, the input file will be modified in-place.",
    )
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

        # Checks on the search results
        assert "results" in first_line, "Each line must have a 'results' field"
        assert isinstance(first_line["results"], list), "'results' field must be a list"
        assert all(isinstance(result, dict) for result in first_line["results"]), "All results must be dicts"
    print("File preconditions met.")


def init_multicuda_worker(metric_names, counter, lock):
    # Initializes Metric instances for process_worker
    global metrics

    # Resolves which CUDA device (0, 1, ..., n) this worker will use
    global device_counter, device_lock
    device_counter = counter
    device_lock = lock

    with device_lock:
        device_id = device_counter.value
        device_counter.value += 1

    print(f"Worker initialized on device {device_id}")
    metrics = []
    for metric_name in metric_names:
        metric = Metric.get_metric(metric_name)
        if hasattr(metric, "model"):
            metric.model = metric.model.to(f"cuda:{device_id}")
        metrics.append(metric)


def process_row(row: dict[str, dict | list]):
    """Takes the raw row dict, applies all metrics, and returns the updated row."""
    record = pd.Series(row["record"])
    results = pd.DataFrame(row["results"])

    for metric in metrics:
        scores = metric(record, results)
        colname = f"score_{metric.name}"
        results[colname] = scores

    row["results"] = results.to_dict(orient="records")
    return row


def yield_row(filepath):
    with open(filepath, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    start = time()

    args = argument_parser()
    path_to_config = args.config
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)

    # Get config values
    infile_path = config["infile"]
    metric_names = config["metrics"]

    # Determine output path
    is_in_place = args.outfile is None
    if is_in_place:
        # Create a temporary file in the same directory for atomic write
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(infile_path), prefix=os.path.basename(infile_path) + ".tmp"
        )
        os.close(temp_fd)  # Close the file descriptor, we'll open it by path
        outfile_path = temp_path
        print(f"Processing in-place. Temporary file: {outfile_path}")
    else:
        outfile_path = args.outfile
        print(f"Writing output to: {outfile_path}")

    check_file_preconditions(infile_path)

    try:
        if args.multi_cuda:
            num_devices = torch.cuda.device_count()
            print(f"Using multi-CUDA with {num_devices} devices.")

            device_counter = Value("i", 0)
            device_lock = Lock()

            with open(outfile_path, "w") as outfile:
                with Pool(
                    processes=num_devices,
                    initializer=init_multicuda_worker,
                    initargs=(metric_names, device_counter, device_lock),
                ) as pool:
                    for i, processed_row in enumerate(
                        tqdm(
                            pool.imap_unordered(process_row, yield_row(infile_path), chunksize=1),
                            desc="Processing rows (multi-CUDA)",
                        )
                    ):
                        outfile.write(json.dumps(processed_row) + "\n")
                        if i % 1000 == 0:  # Flush every 1000 lines
                            outfile.flush()

        else:
            # Currently no metrics use db
            metrics = [Metric.get_metric(name, db=None) for name in metric_names]

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

        # --- ATOMIC RENAME LOGIC ---
        if is_in_place:
            print("Processing complete. Atomically replacing original file.")
            # You might want to run your check_file_preconditions on the temp file here as a final safety check
            # check_file_preconditions(outfile_path)
            os.replace(outfile_path, infile_path)  # os.replace is atomic
            print(f"Successfully updated {infile_path}")

    except Exception as e:
        # If anything goes wrong, clean up the temporary file
        if is_in_place and "outfile_path" in locals() and os.path.exists(outfile_path):
            print(f"An error occurred: {e}. Cleaning up temporary file.")
            os.remove(outfile_path)
        raise  # Re-raise the exception

    end = time()
    print(f"Total time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
