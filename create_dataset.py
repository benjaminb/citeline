"""
1. Get file path from command line args
2. Check if you have a progress log for this file
3. If not, create a new progress log
4. If yes, get where you left off
5. Iterate over the file


"""

import argparse
import os
import pandas as pd


def main():
    # Setup
    parser = argparse.ArgumentParser(description="Creates dataset from file.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the file to create dataset from.",
    )
    args = parser.parse_args()
    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    progress_log_path = f"{file_path}.progress"
    if not os.path.exists(progress_log_path):
        print(f"No progress log found for {file_path}.")
        print("Creating new progress log and starting from beginning.")
        with open(progress_log_path, "w") as f:
            f.write("iloc\n")

    # Load data files
    research = pd.read_json("data/preprocessed/research.jsonl", lines=True)
    reviews = pd.read_json("data/preprocessed/reviews.jsonl", lines=True)
    print(f"Loaded {len(research)} research records and {len(reviews)} review records.")

    # Convert DataFrames to lists of dictionaries
    research_dicts = research.to_dict("records")
    reviews_dicts = reviews.to_dict("records")

    # Build the index for fast lookup by bibcode
    print("Building bibcode index for faster lookups...", end="")
    bibcode_index = {record["bibcode"]: record for record in research_dicts}


if __name__ == "__main__":
    main()
