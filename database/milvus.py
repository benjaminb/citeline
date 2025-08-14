import argparse
from pymilvus import connections, MilvusClient, Collection, FieldSchema, CollectionSchema, DataType
import os
from dotenv import load_dotenv
from tqdm import tqdm
from embedders import get_embedder, Embedder
import torch

load_dotenv("../.env")


def argument_parser():
    parser = argparse.ArgumentParser(description="Milvus DB Management")

    parser.add_argument("--drop-collection", type=str, help="Name of the collection to drop")
    parser.add_argument("--create-collection", type=str, help="JSON string of fields to add")
    parser.add_argument("--add-vector-field", type=str)

    # If adding a vector field, we need the collection name and the name of the embedder to use
    parser.add_argument("--field-name", type=str, help="Name of the vector field to add")
    parser.add_argument("--to-collection", type=str, help="Name of the collection to add the field to")
    parser.add_argument("--embedder", type=str, help="Name of the embedder (e.g. 'BAAI/bge-en-large-v1.5')")

    return parser


class MilvusDB:
    BASE_FIELDS = [
        {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
        {"field_name": "text", "datatype": DataType.VARCHAR, "max_length": 2048},
        {"field_name": "pubdate", "datatype": DataType.INT64},  # Milvus has no date type
        {"field_name": "doi", "datatype": DataType.VARCHAR, "max_length": 64},
    ]

    CLEAR_GPU_CACHE_FN = {"cuda": torch.cuda.empty_cache, "mps": torch.mps.empty_cache, "cpu": lambda: None}

    def __init__(self, alias: str = "default"):
        self.client = MilvusClient(alias=alias)
        # Assumes you're using the best GPU available
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.clear_gpu_cache = self.CLEAR_GPU_CACHE_FN[self.device]

    def _get_field_schemas(self, collection: Collection) -> list[FieldSchema]:
        fields = []
        for field in collection.schema.fields:
            new_field = FieldSchema(
                name=field.name,
                dtype=field.dtype,
                is_primary=field.is_primary,
                auto_id=field.auto_id,
                max_length=field.max_length if field.dtype == DataType.VARCHAR else None,
                dim=(
                    field.dim
                    if field.dtype in [DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR]
                    else None
                ),
                description=field.description,
            )
            fields.append(new_field)

        return fields

    def create_test_collection(self, name: str, added_fields: list[dict]):
        if name in self.client.list_collections():
            print(f"Collection '{name}' already exists.")
            return

        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        for field in self.BASE_FIELDS + added_fields:
            schema.add_field(**field)

        self.client.create_collection(collection_name=name, schema=schema)
        print(f"Collection '{name}' created with new fields: {added_fields}")
        collection = Collection(name)
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "FLAT",
                "metric_type": "IP"
            }
        )

    def drop_collection(self, name: str):
        if self.client.has_collection(name):
            self.client.drop_collection(name)
            print(f"Collection '{name}' dropped.")
        else:
            print(f"Collection '{name}' does not exist.")

    def insert(self, collection_name: str, data: list[dict]) -> int:
        return self.client.insert(collection_name=collection_name, data=data)

    def add_fields_to_collection(self, collection_name: str, fields: list[FieldSchema]):
        """
        Takes an existing collection, creates a copy with new field schemas added,
        and replaces the old collection with the new one.
        """
        if not collection_name in self.client.list_collections():
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        # Get the existing collection
        collection = Collection(collection_name)

        # Create a new schema with the added fields
        new_fields = self._get_field_schemas(collection) + fields
        new_schema = CollectionSchema(
            fields=new_fields,
            description=collection.schema.description,
        )

        # Copy data from the old collection to the new one
        new_collection_name = f"{collection_name}_new"
        self.client.create_collection(name=new_collection_name, schema=new_schema)

        # Replace the old collection with the new one
        self.client.drop_collection(collection_name)
        self.client.create_collection(name=collection_name, schema=new_schema)

    def add_vector_field(
        self, collection_name: str, field_name: str, embedder_name: str, normalize=False, batch_size: int = 16
    ):
        if not collection_name in self.client.list_collections():
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        # Make sure we have CPU count available
        assert "CPUS" in os.environ, "CPUS environment variable not set."
        try:
            num_cpus = int(os.getenv("CPUS"))
        except ValueError:
            raise ValueError(f"Invalid value for CPUS environment variable.")

        from concurrent.futures import ThreadPoolExecutor
        import queue
        import threading

        collection = Collection(collection_name)
        collection.load()

        # Get all field names from the collection
        existing_fields = self._get_field_schemas(collection)
        existing_fieldnames = [field.name for field in existing_fields]
        existing_fieldnames.remove("id")

        embedder = get_embedder(model_name=embedder_name, device=self.device, normalize=normalize)

        # Create new field schema for the vector field
        vector_field = FieldSchema(
            name=field_name,
            dtype=DataType.FLOAT_VECTOR,
            dim=embedder.dim,
            description=f"Vector field for {field_name}",
        )

        new_fields = existing_fields + [vector_field]
        new_schema = CollectionSchema(new_fields)
        new_collection_name = f"{collection_name}_new"
        new_collection = Collection(new_collection_name, new_schema)


        insert_queue = queue.Queue(maxsize=num_cpus * 2)
        # embedding_lock = threading.Lock()
        insertion_lock = threading.Lock()

        # For tracking progress
        num_entities = collection.num_entities
        # producer_progress, consumer_progress = 0, 0

        def insert_worker():
            # nonlocal consumer_progress

            while True:
                try:
                    entities, embeddings = insert_queue.get(timeout=30)
                    if entities is None:
                        insert_queue.task_done()
                        break

                    print(f"Embedding shape: {embeddings.shape}", flush=True)
                    # if entities:
                    #     print(f"First entity: {entities[0]}", flush=True)
                    #     print(f"Entity keys: {entities[0].keys()}", flush=True)
                    #     print(f"Entity values: {list(entities[0].values())}", flush=True)
                    for entity, embedding in zip(entities, embeddings):
                        print({embedding.shape}, flush=True)
                    # let's assume the entities are a list of dicts
                    updated_records = [entity | {field_name: embedding} for entity, embedding in zip(entities, embeddings)]

                    # self.client.insert(collection_name=new_collection_name, data=updated_records)
                    new_collection.insert(data=updated_records)

                    with insertion_lock:
                        # consumer_progress += len(entities)
                        insert_bar.update(len(entities))

                except queue.Empty:
                    print("Insertion worker timed out waiting for data.")
                    break
                except Exception as e:
                    print(f"Insertion worker encountered an error: {e}")
                finally:
                    insert_queue.task_done()

        insert_threads = []
        num_workers = max(1, num_cpus - 1)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                thread = executor.submit(insert_worker)
                insert_threads.append(thread)

            # Create progress bars
            with tqdm(total=num_entities, desc="Embedding", unit="docs", position=0) as embed_bar, tqdm(
                total=num_entities, desc="Inserting", unit="docs", position=1
            ) as insert_bar:

                # Main thread: producer (embed using gpu)
                clear_cache_interval = 50 * batch_size  # Will clear GPU cache every 50 batches
                for offset in range(0, num_entities, batch_size):
                    if offset % clear_cache_interval == 0:
                        self.clear_gpu_cache()
                    print("querying...")
                    entities = collection.query(
                        expr="",
                        output_fields=existing_fieldnames,
                        offset=offset,
                        limit=batch_size,
                    )
                    print(f"Got {len(entities)} entities")

                    texts = [entity["text"] for entity in entities]
                    embeddings = embedder(texts)

                    insert_queue.put((entities, embeddings))

                    # Update producer progress
                    # producer_progress += len(entities) # TODO: not needed? here or in run_experiment?
                    embed_bar.update(len(entities))

                for _ in range(num_workers):
                    insert_queue.put((None, None))

                insert_queue.join()
        print("All insertions complete.")
        print("Flushing to disk...")
        new_collection.flush()
        new_collection.create_index(
            field_name=field_name,
            index_params={
                "index_type": "FLAT",
                "metric_type": "IP"
            }
        )
        print(f"Old collection {collection_name}: {collection.num_entities} entities")
        print(f"New collection {new_collection_name}: {new_collection.num_entities} entities")


if __name__ == "__main__":
    db = MilvusDB()
    db.drop_collection("foo")
    db.drop_collection("foo_new")
    db.create_test_collection(
        name="foo",
        added_fields=[
            {"field_name": "vector", "datatype": DataType.FLOAT_VECTOR, "dim": 3}
        ]
    )

    collection = Collection('foo')

    data = [
        {"text": "Hello world", "pubdate": 20250813, "doi": "abc", "vector": [0.1, 0.2, 0.3]},
        {"text": "Milvus is great", "pubdate": 19950808, "doi": "def", "vector": [0.4, 0.5, 0.6]},
    ]
    res = db.insert(collection_name="foo", data=data)
    collection.flush()
    print(f"Result: {res}")
    print(f"Collection 'foo' has {collection.num_entities} entities.")

    print("Making copies...")
    db.add_vector_field(
        collection_name="foo",
        field_name="bge_small",
        embedder_name="BAAI/bge-small-en",
        normalize=True
    )

    # db.add_vector_field(
    #     collection_name="contributions", field_name="new_vector", embedder_name="BAAI/bge-large-en-v1.5", normalize=True
    # )
