"""
Builds the {doi: number of entities} mapping for a collection, used as the ideal DCG for
chunk-level nDCG in experiment.py.

python src/citeline/database/chunk_counter.py --collection qwen06_chunks --output src/citeline/database/doi_to_chunks_counts.json

All the *_chunks collections share one corpus, so a single file serves all of them. The
*_contributions collections need their own.
"""

import argparse
import json
from collections import Counter

from citeline.database.milvusdb import MilvusDB


def argument_parser():
    parser = argparse.ArgumentParser(description="Count how many entities each DOI has in a Milvus collection.")
    parser.add_argument("--collection", type=str, required=True, help="Name of the collection to count")
    parser.add_argument(
        "--output",
        type=str,
        default="doi_to_chunks_counts.json",
        help="Path to write the {doi: count} JSON (default: doi_to_chunks_counts.json)",
    )
    return parser.parse_args()


def count_collection(db: MilvusDB, collection: str) -> Counter:
    """Sweeps the whole collection, tallying entities per DOI."""
    db.client.load_collection(collection)
    it = db.client.query_iterator(collection_name=collection, filter="", output_fields=["doi"], batch_size=16384)
    counts = Counter()
    it_counter = 0
    while page := it.next():
        counts.update(e["doi"] for e in page)
        it_counter += 1
        if it_counter % 100 == 0:
            print("#", end="", flush=True)
    it.close()
    print()
    return counts


def main():
    args = argument_parser()

    db = MilvusDB()
    counts = count_collection(db, args.collection)

    values = sorted(counts.values())
    print(
        f"{len(counts)} documents, {sum(values)} entities, "
        f"median {values[len(values)//2]} per doc (min {values[0]}, max {values[-1]})"
    )

    with open(args.output, "w") as f:
        json.dump(dict(counts), f)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
