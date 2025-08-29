import argparse
from pymilvus import MilvusClient, Collection, DataType
import os
from dotenv import load_dotenv
from tqdm import tqdm
from embedders import Embedder
import torch
import pandas as pd

load_dotenv("../.env")


def argument_parser():
    """
    python milvusdb.py --create-collection --name <collection_name> --data-source <data_source> --embedder <embedder_name> [--normalize] [--batch-size <16>]
    """
    parser = argparse.ArgumentParser(description="Milvus DB Management")

    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument("--drop-collection", type=str, help="Name of the collection to drop")
    operation_group.add_argument("--create-collection", action="store_true", help="Create a new collection")
    operation_group.add_argument("--list-collections", action="store_true", help="List all collections")
    operation_group.add_argument("--healthcheck", action="store_true", help="Check the health of the Milvus server")
    operation_group.add_argument("--describe-collection", type=str, help="Describe a collection")
    operation_group.add_argument("--create-index", type=str, help="Create an index on a collection")
    operation_group.add_argument("--export-collection", type=str, help="Export a collection to a JSONL file")
    operation_group.add_argument(
        "--rename-collection", nargs=2, metavar=("old_name", "new_name"), help="Rename a collection"
    )

    # Arguments required when creating a collection
    parser.add_argument("--name", type=str, help="Name to give the new collection")
    parser.add_argument("--data-source", type=str, help="Path to the JSONL data file")
    parser.add_argument("--embedder", type=str, help="Name of the embedder (e.g. 'BAAI/bge-en-large-v1.5')")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the embeddings")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding")
    parser.add_argument(
        "--index-type", type=str, default="FLAT", help="Type of index to create (e.g. 'FLAT', 'IVF', etc.)"
    )
    parser.add_argument(
        "--metric-type", type=str, default="IP", help="Metric type for the index (e.g. 'IP', 'L2', etc.)"
    )

    parser.add_argument("--output-file", type=str, help="Path to the output JSONL file")

    args = parser.parse_args()

    # Validate required arguments for create-collection
    if args.create_collection:
        if not all([args.name, args.data_source, args.embedder]):
            parser.error("--create-collection requires --name, --data-source, and --embedder arguments")

    return args


class MilvusDB:
    BASE_FIELDS = [
        {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
        {"field_name": "text", "datatype": DataType.VARCHAR, "max_length": 4096},
        {"field_name": "doi", "datatype": DataType.VARCHAR, "max_length": 64},
        {"field_name": "citation_count", "datatype": DataType.INT64},
        {"field_name": "pubdate", "datatype": DataType.INT64},  # Milvus has no date type, so we use int YYYYMMDD
    ]

    CLEAR_GPU_CACHE_FN = {"cuda": torch.cuda.empty_cache, "mps": torch.mps.empty_cache, "cpu": lambda: None}

    def __init__(self, alias: str = "default"):
        self.client = MilvusClient(alias=alias)

        # Set device and its related clear cache function
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.clear_gpu_cache = self.CLEAR_GPU_CACHE_FN[self.device]

    def healthcheck(self):
        status = self.client.get_server_version()
        print(f"Milvus server version: {status}")

    def _filter_existing_data(self, collection: Collection, data: pd.DataFrame) -> pd.DataFrame:
        """Retrieve existing entities and filter out duplicates from input data"""

        # Flush and refresh
        collection.flush()
        collection.load()
        if collection.num_entities == 0:
            print("Collection is empty. Will insert all data.")
            return data

        print(f"Retrieving {collection.num_entities} existing entities from collection '{collection.name}'...")

        # Use iterator to get ALL entities without offset/limit constraints
        iterator = collection.query_iterator(
            expr="", output_fields=["text", "doi"], batch_size=1000  # Process in smaller batches
        )

        all_existing_entities = []
        progress_bar = tqdm(total=collection.num_entities, desc="Querying existing entities")

        while True:
            batch = iterator.next()
            if not batch:
                break
            all_existing_entities.extend(batch)
            progress_bar.update(len(batch))

        assert (
            len(all_existing_entities) == collection.num_entities
        ), f"Expected {collection.num_entities} entities, but got {len(all_existing_entities)} from query iterator"
        progress_bar.close()
        iterator.close()

        print(f"Retrieved {len(all_existing_entities)} existing entities from collection '{collection.name}'")

        # Create a set of existing (doi, text_prefix) for fast lookup
        existing_keys = set()
        for entity in tqdm(all_existing_entities, desc="Building existing keys set"):
            text_prefix = entity["text"][:100]
            existing_keys.add((entity["doi"], text_prefix))

        print(f"Built {len(existing_keys)} unique (doi, text_prefix) keys from existing entities")
        print(f"Examples:")
        for doi, text_prefix in list(existing_keys)[:5]:  # Show 5 examples
            print(f" - DOI: {doi}, Text Prefix: {text_prefix}")

        # Check data against existing keys - remove rows that already exist in collection
        rows_to_remove = set()
        matches_found = 0

        for idx, row in tqdm(data.iterrows(), desc="Filtering already inserted data"):
            text_prefix = row["text"][:100]
            key = (row["doi"], text_prefix)
            if key in existing_keys:
                rows_to_remove.add(idx)
                matches_found += 1

        print(f"{matches_found} rows in dataset match existing entities")

        # Drop rows already inserted using original indices
        original_data_len = len(data)
        data = data.drop(index=rows_to_remove)
        assert len(data) == original_data_len - matches_found, "Mismatch in dataset length after filtering"
        print(f"Dataset length: {original_data_len}->{len(data)}")
        return data

    def create_index(self, collection_name, index_type: str = "FLAT", metric_type: str = "IP"):
        collection = Collection(collection_name)
        collection.create_index(
            field_name="vector", index_params={"index_type": index_type, "metric_type": metric_type}
        )

    def create_vector_collection(self, name: str, data_source: str, embedder_name: str, normalize: bool, batch_size=16):
        """
        Creates a collection using the base fields and a single vector field inferred by the embedder.

        Note that all collections will have nearly identical schema:
            vector: the vector representation of the document
            text: the text content of the document (original text, contribution, or other document expansion)
            doi: the DOI of the research paper from which the text originates
            pubdate: the publication date of the research paper (int YYYYMMDD format)
        Args:
            name: Name of the collection to create
            data_source: path to a jsonl file with entities containing keys 'text', 'doi', 'citation_count', and 'pubdate' (int YYYYMMDD format)
            embedder_name: Name of the embedder to use for generating vector embeddings
            normalize: Whether to normalize the embeddings
        """
        # Make sure we have CPU count set
        assert "CPUS" in os.environ, "CPUS environment variable not set."
        try:
            num_cpus = int(os.getenv("CPUS"))
        except ValueError:
            raise ValueError(f"Invalid value for CPUS environment variable.")

        # Load and check data, load embedder
        data = pd.read_json(data_source, lines=True)
        assert set(data.columns) == {
            "text",
            "doi",
            "pubdate",
            "citation_count",
        }, f"DataFrame must contain 'text', 'doi', 'citation_count', and 'pubdate' columns (and no others). Dataset given has columns {data.columns}"
        embedder = Embedder.create(model_name=embedder_name, device=self.device, normalize=normalize, for_queries=False)

        # Check if collection already exists and handle resumption
        if name in self.client.list_collections():
            print(f"Collection '{name}' already exists. Checking for existing data...")
            collection = Collection(name)
            collection.load()  # Ensure collection is loaded

            if len(data) == collection.num_entities:
                print("Data source length same as collection, there appears to be nothing to insert.")
                return
            data = self._filter_existing_data(collection, data)

        else:
            # Create new collection
            print(f"Creating new collection '{name}'...")
            # Set up the schema
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            vector_field = {"field_name": "vector", "datatype": DataType.FLOAT_VECTOR, "dim": embedder.dim}
            for field in self.BASE_FIELDS + [vector_field]:
                schema.add_field(**field)

            self.client.create_collection(collection_name=name, schema=schema)
            print(f"Collection '{name}' created")
            collection = Collection(name)
            collection.create_index(field_name="vector", index_params={"index_type": "FLAT", "metric_type": "IP"})

        # Print a table of the collection to be created, num cpus, embedder name and its dimension
        print("=" * 50)
        print("COLLECTION CREATION SUMMARY")
        print("=" * 50)
        print(f"Collection Name     : {name}")
        print(f"Collection Entities : {collection.num_entities}")
        print(f"Num CPUs            : {num_cpus}")
        print(f"Embedder Name       : {embedder_name}")
        print(f"Embedder Dimension  : {embedder.dim}")
        print(f"Normalize           : {normalize}")
        print(f"Batch Size          : {batch_size}")
        print(f"Device              : {self.device}")
        print(f"Data source         : {data_source}")
        print(f"Data size           : {len(data)} rows")
        print("=" * 50)

        self.__embed_and_insert(
            collection=collection, embedder=embedder, data=data, num_cpus=num_cpus, batch_size=batch_size
        )

        print(f"New collection {collection.name}: {collection.num_entities} entities")

    def describe_collection(self, name: str):
        if not name in self.client.list_collections():
            print(f"Collection '{name}' does not exist.")
            return

        try:
            res = self.client.describe_collection(name)
            print(f"Collection '{name}' description:")
            for key, value in res.items():
                print(f"  {key}: {value}")
            collection = Collection(name)
            print("Number of entities:", collection.num_entities)
        except Exception as e:
            print(f"Error describing collection '{name}': {e}")

    def drop_collection(self, name: str):
        if self.client.has_collection(name):
            self.client.drop_collection(name)
            print(f"Collection '{name}' dropped.")
        else:
            print(f"Collection '{name}' does not exist.")

    def export_collection(self, name: str, output_file: str = None):
        """
        Exports a collection to a JSONL file.

        Args:
            name: Name of the collection to export.
            output_file: Path to the output JSONL file. If None, defaults to '<collection_name>_export.jsonl'.
        """
        if not name in self.client.list_collections():
            print(f"Collection '{name}' does not exist.")
            return

        collection = Collection(name)
        collection.load()
        if collection.num_entities == 0:
            print(f"Collection '{name}' is empty. Nothing to export.")
            return

        if output_file is None:
            output_file = f"{name}_export.jsonl"

        print(f"Exporting collection '{name}' with {collection.num_entities} entities to '{output_file}'...")

        iterator = collection.query_iterator(
            expr="", output_fields=["text", "doi", "citation_count", "pubdate", "vector"], batch_size=1000
        )

        with open(output_file, "w") as f:
            progress_bar = tqdm(total=collection.num_entities, desc="Exporting entities")
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for entity in batch:
                    del entity["id"]
                    f.write(f"{entity}\n")
                progress_bar.update(len(batch))
            progress_bar.close()
            iterator.close()

        print(f"Export completed. Data saved to '{output_file}'.")

    def list_collections(self):
        collections = self.client.list_collections()
        collections.sort()
        print(f"Collections: {collections}")
        for collection_name in collections:
            collection = Collection(collection_name)
            print(f" - {collection_name}: {collection.num_entities} entities")

    def rename_collection(self, old_name: str, new_name: str):
        try:
            self.client.rename_collection(old_name, new_name)
            print(f"Collection '{old_name}' renamed to '{new_name}'.")
        except Exception as e:
            print(f"Error renaming collection '{old_name}': {e}")

    def search(
        self,
        collection_name: str,
        query_records: list[dict],
        query_vectors: list[float],
        metric: str = "IP",
        limit: int = 3,
    ) -> list[list[dict]]:
        """
        Searches a collection for top-k results based on the queries and metric.

        Returns:
            A 2D list of search result dicts, i.e.
            [
                [{result 1}, {result 2}, ...] # results for query 1
                [{result 1}, {result 2}, ...], # results for query 2
                ...
            ]

            Each result dict has keys {'metric': float, 'text': str, 'doi': str, 'pubdate': int}

            NOTE: 'metric' corresponds to 'distance' returned by Milvus DB. However this may be distance
            or it may be similarity depending on metric used (e.g. with "L2" metric, smaller is better, but with "IP" larger is better)
            Milvus calls this value 'distance' regardless of the metric used. I have renamed this 'metric' to remind us
            that this could be either, and to remember if largest is best or worst based on the metric chosen
        """

        # In order to apply pubdate filter, we must search one query at a time
        results = []
        for record, vector in zip(query_records, query_vectors):
            hits = self.client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field="vector",
                search_params={"metric_type": metric},
                limit=limit,
                output_fields=["text", "doi", "pubdate", "citation_count"],
                filter=f"pubdate <= {record['pubdate']}",
            )

            # Since we only searched one query, keep the first (and only) list of hits
            retrieved_pubdates = [hit["entity"]["pubdate"] for hit in hits[0]]
            assert all(
                pubdate <= record["pubdate"] for pubdate in retrieved_pubdates
            ), "Retrieved pubdates are not all <= query pubdate"
            results.append(hits[0])

        formatted_results = []
        for hits in results:
            # Add distance to entity and return
            formatted_hits = [hit["entity"] | {"metric": hit["distance"]} for hit in hits]
            formatted_results.append(formatted_hits)
        return formatted_results

    def __embed_and_insert(
        self,
        collection: Collection,
        embedder: Embedder,
        data: pd.DataFrame,
        num_cpus: int,
        batch_size: int = 16,
    ):
        """

        Args:
            collection: The collection to add the vector field to.
            embedder: The embedder to use for generating vector embeddings.
            data: DataFrame assumed to have columns ["text", "doi", "pubdate" (int YYYYMMDD format)]
            num_cpus: The number of CPU cores to use for parallel processing.
            batch_size: The batch size to use for embedding and inserting data.
        """
        from concurrent.futures import ThreadPoolExecutor
        import queue
        import threading

        insert_queue = queue.Queue(maxsize=num_cpus * batch_size * 2)
        insertion_lock = threading.Lock()
        flush_interval = int(os.getenv("FLUSH_INTERVAL", 1000))
        print(f"Using disk flush interval: {flush_interval}", flush=True)
        inserted_count = 0  # Counter the insert_workers use to determine when to flush

        def insert_worker():
            nonlocal inserted_count
            while True:
                try:
                    # Get batch and check if the queue is empty
                    batch_records = insert_queue.get()
                    if batch_records is None:
                        return  # finally will still call task_done()

                    # Insert, then inside lock update progress & check to flush
                    collection.insert(batch_records)
                    with insertion_lock:
                        insert_bar.update(len(batch_records))
                        inserted_count += len(batch_records)
                        if inserted_count >= flush_interval:
                            collection.flush()
                            inserted_count = 0

                except Exception as e:
                    print(f"Insertion worker encountered an error: {e}")
                finally:
                    insert_queue.task_done()

        num_workers = max(1, num_cpus - 1)
        num_entities = len(data)
        futures = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                # thread = executor.submit(insert_worker)
                futures.append(executor.submit(insert_worker))

            # Create progress bars
            with tqdm(total=num_entities, desc="Embedding", unit="docs", position=0) as embed_bar, tqdm(
                total=num_entities, desc="Inserting", unit="docs", position=1
            ) as insert_bar:

                # Main thread: producer (embed using gpu)
                clear_cache_interval = 50 * batch_size  # Will clear GPU cache every 50 batches
                for i in range(0, num_entities, batch_size):
                    if i % clear_cache_interval == 0:
                        self.clear_gpu_cache()

                    batch = data.iloc[slice(i, i + batch_size)]
                    embeddings = embedder(batch["text"])

                    batch_records = batch.to_dict(orient="records")
                    for record, vector in zip(batch_records, embeddings):
                        record["vector"] = vector
                    insert_queue.put(batch_records)

                    embed_bar.update(len(batch))

                for _ in range(num_workers):
                    insert_queue.put(None)
                insert_queue.join()

                # Surface any worker exceptions
                for f in futures:
                    f.result()

        print("All insertions complete. Final flush to disk...", end="")
        collection.flush()
        print("done!")


def main():
    args = argument_parser()

    db = MilvusDB()

    if args.drop_collection:
        db.drop_collection(args.drop_collection)
    elif args.create_collection:
        db.create_vector_collection(
            name=args.name,
            data_source=args.data_source,
            embedder_name=args.embedder,
            normalize=args.normalize,
            batch_size=args.batch_size,
        )
    elif args.create_index:
        db.create_index(collection_name=args.create_index, index_type=args.index_type, metric_type=args.metric_type)
    elif args.describe_collection:
        db.describe_collection(args.describe_collection)
    elif args.export_collection:
        db.export_collection(args.export_collection, output_file=args.output_file)
    elif args.list_collections:
        db.list_collections()
    elif args.rename_collection:
        old_name, new_name = args.rename_collection
        db.rename_collection(old_name, new_name)
    elif args.healthcheck:
        db.healthcheck()


if __name__ == "__main__":
    main()
    # db = MilvusDB()
    # db.client.list_collections()
    # data = "../testdata.jsonl"
    # db.create_vector_collection(
    #     name="test_collection", data_source=data, embedder_name="BAAI/bge-small-en", normalize=False
    # )
    # db.client.list_collections()

    # import numpy as np

    # query = np.random.rand(384).astype(np.float32)
    # db.client.load_collection("test_collection")
    # res = db.client.search(
    #     collection_name="test_collection",
    #     anns_field="vector",
    #     data=[query],
    #     output_fields=["text", "doi", "pubdate"],
    #     limit=3,
    #     search_params={"metric_type": "IP"},
    # )
    # print(f"Search results:")
    # from pprint import pprint

    # pprint(res[0])

    # db.drop_collection("test_collection")
