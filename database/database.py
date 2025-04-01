import argparse
import gc
import json
import numpy as np
import os
from datetime import datetime
import inspect
import multiprocessing
import pandas as pd
import psycopg
import queue
import sys
import torch
from typing import Literal
from dataclasses import dataclass
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from semantic_text_splitter import TextSplitter
from time import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time
import cProfile
import pstats

# Add the parent directory to sys.path so we can import Embedders, Enrichers, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# fmt: off
# fmt: on

"""
USAGE:
python database.py --test-connection
python database.py --create-base-table --table-name library --from-path="../data/preprocessed/research.jsonl"
python database.py --create-vector-column --table-name library --target-column chunk --embedder-name "BAAI/bge-small-en" [--normalize] --batch-size 16
python database.py --create-index --table-name lib --target-column bge_norm --index-type hnsw --m 32 --ef-construction 512
"""


def argument_parser():
    parser = argparse.ArgumentParser(description="Database operations")

    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        "--test-connection", "-t", action="store_true", help="Test database connection"
    )
    operation_group.add_argument(
        "--create-vector-table",
        "-V",
        action="store_true",
        help="Create a new vector table",
    )
    operation_group.add_argument(
        "--create-index", "-I", action="store_true", help="Create an index on a table"
    )
    operation_group.add_argument(
        "--add-chunks", "-A", action="store_true", help="Add chunks to the database"
    )
    operation_group.add_argument(
        "--create-base-table",
        "-B",
        action="store_true",
        help="Create the base table for chunks and insert records into it",
    )
    operation_group.add_argument(
        "--create-vector-column",
        "-v",
        action="store_true",
        help="Create a new column in the specified table to store vector embeddings",
    )

    # Create base table arguments (NOTE: --table-name is used by multiple operations)
    parser.add_argument(
        "--from-path",
        "-f",
        type=str,
        help="Path to the dataset (JSONL file) to create the base table from",
    )

    # Used by multiple operations
    parser.add_argument("--table-name", "-T", type=str, default="lib", help="Name of target table")

    # Create vector column arguments
    parser.add_argument(
        "--embedder-name",
        "-e",
        type=str,
        default="BAAI/bge-small-en",
        help="Name of the embedding model to use (default: BAAI/bge-small-en)",
    )
    parser.add_argument(
        "--enricher-name",
        "-E",
        type=str,
        default=None,
        help="Name of the enricher to use (default: None). If provided, it will be used to enrich the chunks before embedding",
    )
    parser.add_argument(
        "--normalize",
        "-n",
        default=False,
        action="store_true",
        help="Normalize embeddings before inserting into the database",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for embedding and insertion operations",
    )
    parser.add_argument(
        "--target-column",
        "-c",
        type=str,
        default="chunk",
        help="Name of the column to embed into vectors (default: chunk)",
    )

    # Create index arguments
    # --table-name already defined
    # --target-column already defined
    parser.add_argument(
        "--index-type",
        "-i",
        default="hnsw",
        type=str,
        help="Type of index to create (default: hnsw)",
    )
    parser.add_argument(
        "--metric",
        "-m",
        default="vector_cosine_ops",
        type=str,
        help="Distance metric to use for the index (default: vector_cosine_ops)",
    )
    parser.add_argument(
        "--m", "-M", default=32, type=int, help="M parameter for HNSW (default: 32)"
    )
    parser.add_argument(
        "--ef-construction",
        "-ef",
        default=512,
        type=int,
        help="efConstruction parameter for HNSW (default: 512)",
    )
    parser.add_argument(
        "--num-lists",
        "-l",
        default=1580,
        type=int,
        help="Number of lists for IVFFlat (default: 1580)",
    )

    # Add chunks arguments
    parser.add_argument("--path", "-p", type=str, help="Path to the dataset to add to the database")
    parser.add_argument(
        "--max-length",
        "-L",
        type=int,
        default=1500,
        help="Maximum length of each chunk",
    )
    parser.add_argument("--overlap", "-o", type=int, default=150, help="Overlap between chunks")

    args = parser.parse_args()

    # Validate args
    if args.create_base_table and not all([args.from_path, args.table_name]):
        parser.error("--create-base-table requires --table-name and --from-path")

    elif args.create_vector_column and not all([args.table_name, args.embedder_name]):
        parser.error("--create-vector-column requires --table-name and --embedder")

    elif args.create_vector_table and not all([args.table_name, args.embedder_name]):
        parser.error("--create-vector-table requires --table-name and --embedder")

    elif args.create_index:
        if not all([args.table_name, args.target_column, args.index_type]):
            parser.error("--create-index requires --index-type and --table-name")
        if args.index_type == "ivfflat" and not all([args.num_lists]):
            parser.error("ivfflat index requires --num-lists")
        if args.index_type == "hnsw" and not all([args.m, args.ef_construction]):
            parser.error("hnsw index requires --m and --ef-construction")

    elif args.add_chunks and not all([args.path, args.max_length, args.overlap]):
        parser.error("--add-chunks requires --path, --max-length, and --overlap")

    return args


"""
DATACLASSES
"""


@dataclass
class SingleVectorQueryResult:
    chunk_id: int
    doi: str
    title: str
    abstract: str
    chunk: str
    distance: float


@dataclass
class SingleVectorDoiResult:
    chunk_id: int
    doi: str
    title: str
    abstract: str
    chunk: str
    vector: np.array


@dataclass
class Chunk:
    id: int
    doi: str
    text: str


@dataclass
class ChunkAndVector:
    text: str
    vector: np.array


"""
PICKLEABLE DB HELPER FUNCTIONS
These must be outside the class definition to be pickled, which is necessary for multiprocessing.
"""


# Producer function
def producer_proc(
    all_ids, all_chunks, results_queue, embedder_name, normalize, batch_size, num_consumers
):
    # Set up embedder
    from Embedders import get_embedder

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = get_embedder(embedder_name, device=device, normalize=normalize)

    for i in range(0, len(all_chunks), batch_size):
        texts, ids = all_chunks[i : i + batch_size], all_ids[i : i + batch_size]
        embeddings = embedder(texts)
        results_queue.put((ids, embeddings))
    # Signal that producer is done
    for _ in range(num_consumers):
        results_queue.put(None)


# Consumer function
def consumer_proc(
    db_params, results_queue, embedder_dim, table_name, vector_column_name, progress_queue
):
    # Each process creates its own connection
    conn = psycopg.connect(**db_params)
    register_vector(conn)
    cur = conn.cursor()

    # Optional: config
    cur.execute("SET work_mem='2GB';")
    cur.execute("SET maintenance_work_mem='2GB';")

    # Create temp table
    cur.execute(f"CREATE TEMP TABLE temp_embeddings (id int, embedding vector({embedder_dim}))")
    conn.commit()

    while True:
        item = results_queue.get()
        if item is None:
            break

        batch_ids, batch_embeddings = item
        with cur.copy(
            "COPY temp_embeddings (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["int4", "vector"])
            for row_id, embedding in zip(batch_ids, batch_embeddings):
                copy.write_row([row_id, embedding])

        cur.execute(
            f"UPDATE {table_name} SET {vector_column_name} = temp.embedding "
            f"FROM temp_embeddings temp WHERE {table_name}.id = temp.id"
        )
        cur.execute("TRUNCATE temp_embeddings")
        conn.commit()
        progress_queue.put(1)

    cur.close()
    conn.commit()
    conn.close()


def record_to_chunked_records(record, max_length, overlap):
    """
    Standalone chunking function that doesn't need a database connection.
    """
    splitter = TextSplitter(capacity=max_length, overlap=overlap)
    chunks = splitter.chunks(record["body"])

    # Remove NUL chars which PostgreSQL doesn't like
    return [
        {
            "title": record["title"].replace("\x00", ""),
            "abstract": record["abstract"].replace("\x00", ""),
            "doi": record["doi"].replace("\x00", ""),
            "pubdate": record.get("pubdate", None),
            "keywords": record.get("keywords", []),
            "chunk": chunk.replace("\x00", ""),
        }
        for chunk in chunks
    ]


class Database:
    EMBEDDER_SHORTNAMES = {
        "BAAI/bge-small-en": "bge",
        "bert-base-uncased": "bert",
        "adsabs/astroBERT": "astrobert",
        "nasa-impact/nasa-ibm-st.38m": "nasa",
    }

    PGVECTOR_DISTANCE_OPS = {
        "vector_l2_ops": "<->",
        "vector_ip_ops": "<#>",
        "vector_cosine_ops": "<=>",
    }

    @classmethod
    def get_db_params(cls, path_to_env: str = ".env") -> dict[str, str]:
        """Load database parameters from a .env file"""
        load_dotenv(path_to_env, override=True)
        return {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
        }

    def __init__(self, path_to_env: str = "../.env"):
        # Check ell env variables have been set
        load_dotenv(path_to_env, override=True)
        required_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_MEM", "DB_CPUS"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

        # Connect to the database
        self.db_params = Database.get_db_params(path_to_env=path_to_env)
        self.conn = psycopg.connect(**self.db_params)
        register_vector(self.conn)

        self.device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        )
        # For text splitting; instantiated in _create_base_table
        self.splitter = None

    def __del__(self):
        if self.conn:
            self.conn.close()

    def __log_error(self, message: str):
        """Log an error message to a file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Get the calling function's name
        caller_name = inspect.currentframe().f_back.f_code.co_name
        with open("database_errors.log", "a") as f:
            f.write(f"[{timestamp}]: (caller: {caller_name}) {message}\n")

    def __clear_gpu_memory(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def __set_session_resources(self, cursor, optimize_for: Literal['query', 'index', 'insert']):
        """
        Set session resources for PostgreSQL
        """
        db_mem = 0.9 * os.getenv("DB_MEM") # Leave 10% for OS overhead
        db_cpus = os.getenv("DB_CPUS") - 2 # Leave 2 CPUs for OS overhead

        if optimize_for == "query":
            query = f"""
                SET synchronous_commit = 'on';
                SET wal_level = 'replica';
                SET maintenance_work_mem = '0MB';
                SET max_wal_size = 'DEFAULT';
                SET random_page_cost = '1.1';
                SET parallel_tuple_cost = '0.1';
                SET parallel_setup_cost = '1000';
                SET max_parallel_workers = '{max(1, db_cpus)}';
                SET work_mem = '{max(1, db_mem // db_cpus)}GB';
                SET max_parallel_workers_per_gather = '{max(1, db_cpus)}';
                SET shared_buffers = '{int(0.25 * db_mem)}GB';
                SET effective_cache_size = '{int(0.75 * db_mem)}GB';
                SET effective_io_concurrency = '200';
            """

        elif optimize_for == "index":
            query = f"""
                SET synchronous_commit = 'off';
                SET wal_level = 'minimal';
                SET work_mem = '{int(0.01 * db_mem)}MB';
                SET maintenance_work_mem = '{int(0.15 * db_mem)}MB';
                SET max_wal_size = '{int(0.1 * db_mem)}MB';
                SET checkpoint_timeout = '30min';
                SET max_parallel_workers_per_gather = '{db_cpus}';  -- Use all CPUs for query parallelism
                SET max_parallel_maintenance_workers = '{int(0.75 * db_cpus)}';  -- Use 75% of CPUs for index creation (reduces contention)
                SET shared_buffers = '{int(0.25 * db_mem)}MB';
                SET effective_cache_size = '{int(0.5 * db_mem)}MB';
            """

        elif optimize_for == "insert":
            query = f"""
                SET synchronous_commit = 'off';
                SET wal_level = 'minimal';
                SET work_mem = '{int(0.002 * db_mem)}MB';
                SET maintenance_work_mem = '0MB';
                SET max_wal_size = '{int(0.1 * db_mem)}MB';
                SET checkpoint_completion_target = '0.9';
                SET wal_writer_delay = '200ms';
                SET max_parallel_workers_per_gather = '0';
                SET max_parallel_maintenance_workers = '0';
                SET shared_buffers = '{int(0.25 * db_mem)}MB';
                SET effective_cache_size = '{int(0.5 * db_mem)}MB';
            """
        cursor.execute(query)

    def _create_base_table(
        self,
        table_name: str,
        from_path: str,
        max_length: int = 1500,
        overlap: int = 150,
    ):
        """
        Create the base table for chunks and insert records into it.

        Args:
            path (str): Path to the dataset (JSONL file).
            max_length (int): Maximum length of each chunk.
            overlap (int): Overlap between chunks.
        """
        # First, create table so we immediately error if it already exists
        cursor = self.conn.cursor()
        cursor.execute(
            f"""CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                doi TEXT NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                pubdate DATE,
                keywords TEXT[],
                chunk TEXT NOT NULL
                );
            """
        )
        self.conn.commit()

        # Read in data and validate columns
        df = pd.read_json(from_path, lines=True)
        required_columns = ["doi", "title", "abstract", "pubdate", "keywords", "body"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"Input JSONL file must contain the following columns: {required_columns}"
            )

        # Split body into chunks
        records = df.to_dict("records")
        self.splitter = TextSplitter(capacity=max_length, overlap=overlap)

        # Use ProcessPoolExecutor to parallelize chunking
        chunked_records = []
        cores = max(1, os.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=cores) as process_executor:
            futures = [
                process_executor.submit(record_to_chunked_records, record, max_length, overlap)
                for record in records
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunking records"):
                chunked_records.extend(future.result())  # Collect all chunks

        try:
            with cursor.copy(
                f"COPY {table_name} (doi, title, abstract, pubdate, keywords, chunk) FROM STDIN WITH (FORMAT TEXT)"
            ) as copy:
                copy.set_types(["text", "text", "text", "text", "text[]", "text"])
                for record in tqdm(chunked_records, desc="Copying chunks"):
                    copy.write_row(
                        [
                            record["doi"],
                            record["title"],
                            record["abstract"],
                            record["pubdate"],
                            record["keywords"],
                            record["chunk"],
                        ]
                    )

            self.conn.commit()
            print(f"Successfully inserted {len(chunked_records)} chunks")

        except Exception as e:
            self.conn.rollback()
            print(f"Error during COPY: {e}")
            self.__log_error(f"Error during COPY: {e}")
            raise e

    def get_chunks_by_doi(
        self, doi: str, table_name: str = "library", vector_column: str = "bge_norm"
    ):
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT id, doi, title, abstract, chunk, {vector_column} FROM {table_name}
            WHERE doi = '{doi}';
            """
        )
        rows = cursor.fetchall()
        return [SingleVectorDoiResult(*row) for row in rows]

    def create_vector_column(
        self,
        embedder_name: str,
        normalize: bool = False,
        enricher_name: str = None,
        table_name: str = "library",
        target_column: str = "chunk",
        batch_size: int = 32,
    ):
        """
        Create a new column in the specified table to store vector embeddings.

        Args:
            embedder_name (str): Name of the embedding model to use.
            enricher_name (str, optional): Name of the enricher to use. Defaults to None.
            table_name (str): Name of the target table.
            target_column (str): Name of the column to embed into vectors.
            dim (int): Dimension of the vector embeddings.
            enricher_name (str, optional): Name of the enricher to use. Defaults to None.
        """
        # Get embedder dimension
        from Embedders import get_embedder

        embedder = get_embedder(embedder_name, self.device)
        dim = embedder.dim
        del embedder

        if enricher_name:
            from TextEnrichers import get_enricher

            enricher = get_enricher(
                name=enricher_name, path_to_data="../data/preprocessed/research.jsonl"
            )
            print(f"Using enricher: {enricher_name}")

        # Construct column name; use the embedder's short name if written into class, otherwise derive it from model name
        vector_column_name = Database.EMBEDDER_SHORTNAMES.get(
            embedder_name, embedder_name.replace("/", "_").replace("-", "_")
        )
        if normalize:
            vector_column_name += "_norm"
        if enricher_name:
            vector_column_name += f"_{enricher_name}"

        print(f"Attempting to create column '{vector_column_name}' in table '{table_name}'...")

        cursor = self.conn.cursor()
        query = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {vector_column_name} VECTOR({dim});"
        print(f"Executing query: {query}")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {vector_column_name} VECTOR({dim});")
        self.conn.commit()
        print("Column created (or it already existed)")

        # Get all text chunks to embed
        start = time()
        cursor.execute(f"SELECT id, doi, {target_column} FROM {table_name};")
        rows = cursor.fetchall()
        print(f"  SELECT execution time: {time() - start:.2f} seconds")
        all_ids, all_dois, all_chunks = zip(*rows)
        print(f"  Fetched {len(all_ids)} rows from the database")
        del rows
        cursor.close()

        # Setting up a producer/consumer queue so writing to db doesn't block embedding the next batch of chunks
        if enricher_name:  # this assumes enricher was resolved above
            texts_with_dois = zip(all_chunks, all_dois)
            all_chunks = enricher.enrich_batch(texts_with_dois=texts_with_dois)

        # Create a multiprocessing Queue
        results_queue = multiprocessing.Queue()
        progress_queue = multiprocessing.Queue()
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        num_consumers = os.getenv("CPU_COUNT", os.cpu_count() - 2)

        # Start producer process
        producer_process = multiprocessing.Process(
            target=producer_proc,
            args=(
                all_ids,
                all_chunks,
                results_queue,
                embedder_name,
                normalize,
                batch_size,
                num_consumers,
            ),
        )
        producer_process.start()

        # Start consumer processes
        consumers = []
        for _ in range(num_consumers):
            p = multiprocessing.Process(
                target=consumer_proc,
                args=(
                    self.db_params,
                    results_queue,
                    dim,
                    table_name,
                    vector_column_name,
                    progress_queue,
                ),
            )
            p.start()
            consumers.append(p)

        with tqdm(total=total_batches, desc="Writing to database", leave=True) as progress_bar:
            processed = 0
            while processed < total_batches:
                try:
                    # Get progress updates from the queue
                    progress = progress_queue.get(timeout=1)  # Adjust timeout as needed
                    processed += progress
                    progress_bar.update(progress)
                except queue.Empty:
                    # No progress update received, continue waiting
                    pass

        # Wait for producer to finish
        producer_process.join()

        # Wait for all consumers to finish
        for c in consumers:
            c.join()

    def create_vector_table_enriched(
        self,
        table_name: str,
        dim: int,
        embedder,
        enricher,
        work_mem="2048MB",
        maintenance_work_mem="2048MB",
        batch_size=32,
    ):
        """
        1. Creates a vector table
        2. Creates indexes for all distance metrics
        3. Batch embeds all records in the `chunks` table using the given embedder

        available distance metrics:
        vector_l2_ops, vector_ip_ops, vector_cosine_ops, vector_l1_ops, bit_hamming_ops, bit_jaccard_ops
        """

        # Set session resources
        cursor = self.conn.cursor()
        cores = os.cpu_count()
        cursor.execute("SHOW max_worker_processes;")
        max_worker_processes = int(cursor.fetchone()[0])
        max_parallel_workers = max(1, cores - 2)
        max_parallel_maintenance_workers = int(0.2 * max_worker_processes)
        print("=" * 33 + "CONFIG" + "=" * 33)
        print(
            "max_worker_processes | max_parallel_workers | max_parallel_maintenance_workers | work_mem | maintenance_work_mem"
        )
        print(
            f"{max_worker_processes:^21} {max_parallel_workers:^22} {max_parallel_maintenance_workers:^34} {work_mem:^10} {maintenance_work_mem:^21}"
        )
        print("=" * 72)
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")
        cursor.execute(f"SET max_parallel_maintenance_workers={max_parallel_maintenance_workers};")
        cursor.execute(f"SET work_mem='{work_mem}';")
        cursor.execute(
            f"SET maintenance_work_mem='{maintenance_work_mem}';",
        )

        # Create table
        cursor.execute(
            f"""CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                embedding VECTOR({dim}),
                chunk_id INTEGER REFERENCES chunks(id)
                );
            """
        )
        self.conn.commit()
        print(f"Created table {table_name}")

        # Get all chunks for embedding
        enricher = get_enricher(enricher)
        ids_and_chunks = self._get_all_chunks(cursor)
        print(f"Embedding and enriching {len(ids_and_chunks)} chunks...")

        # Embed an insert in batches
        for i in tqdm(
            range(0, len(ids_and_chunks), batch_size),
            desc="Inserting embeddings",
            leave=False,
        ):
            # Clear GPU memory every 50 batches
            if i % 50 == 0:
                self.__clear_gpu_memory()

            # Prepare batch
            batch = ids_and_chunks[i : i + batch_size]
            ids, texts = list(zip(*batch))
            start = time()
            embeddings = embedder(texts)
            print(f"Embedding time: {time() - start:.2f} seconds for {len(texts)} chunks")
            data_batch = [(embedding, id_num) for embedding, id_num in zip(embeddings, ids)]

            # Insert
            start_insert = time()
            execute_values(
                cursor,
                f"INSERT INTO {table_name} (embedding, chunk_id) VALUES %s;",
                data_batch,
            )
            print(f"Insert time: {time() - start_insert}")
            self.conn.commit()

        cursor.close()

    def create_index(
        self,
        table_name: str,
        target_column: str,
        index_type: str,  # 'ivfflat' or 'hnsw'
        #  metric: str,
        num_lists: int = 1580,  # sqrt(num chunks, which is ~2.5M)
        m: int = 64,
        ef_construction: int = 512,
    ):
        # Check input
        assert index_type in [
            "ivfflat",
            "hnsw",
        ], f"Invalid index type: {index_type}. Must be 'ivfflat' or 'hnsw'"
        # assert metric in PGVECTOR_DISTANCE_METRICS, f"Invalid metric: {metric}. I don't have that metric in the PGVECTOR_DISTANCE_METRICS dictionary"

        # Set session resources
        cursor = self.conn.cursor()

        # NOTE: these settings based on how I tend to run the db host on FASRC
        max_worker_processes = 62
        max_parallel_workers = 62
        max_parallel_maintenance_workers = 62
        maintenance_work_mem = "6GB"
        print("=" * 48 + "CONFIG" + "=" * 48)
        print(
            "max_worker_processes | max_parallel_workers | max_parallel_maintenance_workers | maintenance_work_mem"
        )
        print(
            f"{max_worker_processes:^21} {max_parallel_workers:^21} {max_parallel_maintenance_workers:^33} {maintenance_work_mem:^21}"
        )
        print("=" * 102)
        cursor.execute(
            f"SET maintenance_work_mem='{maintenance_work_mem}';",
        )
        cursor.execute(f"SET max_parallel_maintenance_workers={max_parallel_maintenance_workers};")
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")

        # Resolve index name and parameters
        index_name = f"idx_{target_column}_{index_type}"
        print(f"Creating index {index_name}")
        parameters = ""
        start = time()
        if index_type == "hnsw":
            parameters = f"(m = {m}, ef_construction = {ef_construction});"
        elif index_type == "ivfflat":
            parameters = f"(lists = {num_lists});"
        else:
            print(f"Invalid index type: {index_type}")
            return

        # Create index
        query = f"CREATE INDEX {index_name} ON {table_name} USING {index_type} ({target_column} vector_cosine_ops) WITH {parameters}"
        start = time()
        cursor.execute(query)
        print(f"  Index creation time: {time() - start:.2f} seconds")
        self.conn.commit()

        # Cleanup
        end = time()
        print(
            f"Created {index_type} index on {table_name}.{table_name} in {end - start:.2f} seconds"
        )
        cursor.close()

    def query_vector_column(
        self,
        query_vector: np.array,
        target_column: str,
        table_name: str = "lib",
        metric: str = "vector_cosine_ops",
        top_k=5,
        use_index=True,
        probes=40,
    ):
        """
        table_name: name of the vector table
        query_vector: the vector to query
        metric: a key in PGVECTOR_DISTANCE_METRICS to resolve the distance operator
        top_k: number of results to return

        """
        # Resolve the distance operator
        _operator_ = self.PGVECTOR_DISTANCE_OPS[metric]

        # Set the session resources
        cursor = self.conn.cursor()
        max_parallel_workers = 62
        max_parallel_workers_per_gather = 62
        work_mem = "1GB"
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")
        cursor.execute(f"SET max_parallel_workers_per_gather={max_parallel_workers_per_gather};")
        cursor.execute(f"SET work_mem='{work_mem}'")

        # Set index search parameters
        if not use_index:
            cursor.execute(f"SET enable_indexscan = off;")
        else:
            cursor.execute(f"SET enable_indexscan = on;")
            cursor.execute("SET enable_seqscan = off;")
            cursor.execute(f"SET ivfflat.probes={probes};")

        start = time()
        cursor.execute(
            f"""
            SELECT id, doi, title, abstract, chunk, {target_column} {_operator_} %s AS distance
            FROM {table_name}
            ORDER BY distance ASC
            LIMIT {top_k};
            """,
            (query_vector,),
        )
        print(f"  Query execution time: {time() - start:.2f} seconds")

        results = cursor.fetchall()
        print(f"  Found {len(results)} results")
        print(f"top_k: {top_k}")
        # Close up
        cursor.close()

        assert (
            len(results) <= top_k
        ), f"Query returned {len(results)} results, but top_k is set to {top_k}"
        return [SingleVectorQueryResult(*result) for result in results]

    def prewarm_table(self, table_name: str, target_column: str = None):
        """
        Prewarms a table and optionally, specific indexes associated with a target column.

        Args:
            table_name (str): The name of the table to prewarm.
            target_column (str, optional): The name of the column whose indexes should be prewarmed.
                                           If None, all indexes on the table are prewarmed. Defaults to None.
        """
        cursor = self.conn.cursor()
        msg = f"Prewarming table {table_name}" + ".{target_column}" if target_column else ""
        print(msg)

        """
        this query fetches indexes on a given target_column:
        SELECT i.relname AS index_name
        FROM pg_index ix
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_class t ON t.oid = ix.indrelid
        JOIN pg_attribute a ON a.attrelid = ix.indrelid
        WHERE t.relname = 'library'
        AND a.attname = 'bge_norm'
        AND a.attnum = ANY(ix.indkey);
        """
        try:
            # Execute query to get all relevant objects
            cursor.execute(
                f"""
                SELECT relname, pg_prewarm(oid) as blocks_loaded
                FROM pg_class 
                WHERE relname = '{table_name}' 
                OR relname IN (
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE tablename = '{table_name}'
                );
            """
            )

            # Fetch and display results
            results = cursor.fetchall()

            print("\n" + "=" * 70)
            print(f"{'Object Name':<30} {'Blocks Loaded':<15} {'Size (MB)':<15}")
            print("-" * 70)

            total_blocks = 0
            for obj_name, blocks in results:
                if blocks is None:
                    blocks = 0
                total_blocks += blocks
                # Convert blocks to MB (8KB per block / 1024 = MB)
                size_mb = blocks * 8 / 1024
                print(f"{obj_name:<30} {blocks:<15} {size_mb:.2f} MB")

            total_size_mb = total_blocks * 8 / 1024
            print("-" * 70)
            print(f"{'TOTAL':<30} {total_blocks:<15} {total_size_mb:.2f} MB")
            print("=" * 70 + "\n")

            return results
        except Exception as e:
            print(f"Error prewarming table {table_name}: {e}")
            raise e
        finally:
            cursor.close()

    def explain_analyze(
        self,
        query_vector: str,
        table_name: str,
        metric: str = "vector_cosine_ops",
        top_k: int = 50,
        outdir: str = "tests/db/",
    ):
        # Set up db connection
        assert (
            metric in PGVECTOR_DISTANCE_METRICS
        ), f"Invalid metric: {metric}. I don't have that metric in the PGVECTOR_DISTANCE_METRICS dictionary"
        operator = PGVECTOR_DISTANCE_METRICS[metric]

        # Set session resources
        cursor = self.conn.cursor()
        cores = os.cpu_count()
        max_parallel_workers = max(1, cores - 2)
        max_parallel_workers_per_gather = max_parallel_workers - 1
        work_mem = "1GB"
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")
        cursor.execute(f"SET max_parallel_workers_per_gather={max_parallel_workers_per_gather};")
        cursor.execute(f"SET work_mem='{work_mem}'")
        cursor.execute(f"SET enable_indexscan = on;")
        # NOTE: ef_search could be higher
        ef_search = top_k
        if top_k > 1000:
            print(f"  WARNING: Setting ef_search ({top_k}) to 1000, highest supported by pgvector.")
            ef_search = 1000
        cursor.execute(f"SET hnsw.ef_search = {ef_search};")
        cursor.execute("SET enable_seqscan = off;")

        self.prewarm_table(table_name)

        # Prepare and execute query (format everything but the vector first, so it can be printed compactly)
        query = """EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON)
            SELECT {table_name}.chunk_id, chunks.doi, chunks.text, {table_name}.embedding {operator} '{query_vector}' AS distance
            FROM {table_name}
            JOIN chunks ON {table_name}.chunk_id = chunks.id
            ORDER BY {table_name}.embedding {operator} '{query_vector}' ASC
            LIMIT {top_k};
            """.format(
            table_name=table_name,
            operator=operator,
            top_k=top_k,
            query_vector="{query_vector}",
        )
        print(f"Executing query: {query}")
        cursor.execute(query.format(query_vector=query_vector))

        # Report
        query_plan = cursor.fetchall()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"explain_analyze_{table_name}_topk{top_k}_{current_time}.json"
        with open(outdir + filename, "w") as f:
            json.dump(query_plan, f, indent=4)
        print(f"Query plan saved to {outdir+filename}")
        cursor.close()

    def test_connection(self):
        cursor = self.conn.cursor()

        # Print a pretty table showing db name, user, host, and port from db_params
        print("=" * 33 + "CONFIG" + "=" * 33)
        print(f"{'Database':<16} {'User':<16} {'Host':<32} {'Port':<16}")
        print(
            f"{self.db_params['dbname']:<16} {self.db_params['user']:<16} {self.db_params['host']:<32} {self.db_params['port']:<16}"
        )
        print("=" * 72)

        # Execute a simple query
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"Database version: {db_version}")

        cursor.close()


def profile_create_vector_table(db, table_name, embedder):
    with cProfile.Profile() as pr:
        db.create_vector_table(table_name=table_name, dim=768, embedder=embedder)
    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats("cumulative").print_stats(40)


def get_kwargs(args, values):
    return {k: v for k, v in vars(args).items() if k in values}


def main():
    args = argument_parser()
    db = Database()
    db.test_connection()

    dispatch_table = {
        "create_base_table": lambda: db._create_base_table(
            **get_kwargs(args, ["table_name", "from_path", "max_length", "overlap"])
        ),
        "create_vector_column": lambda: db.create_vector_column(
            **get_kwargs(
                args,
                [
                    "table_name",
                    "embedder_name",
                    "normalize",
                    "enricher_name",
                    "target_column",
                    "batch_size",
                ],
            )
        ),
        "create_index": lambda: db.create_index(
            **get_kwargs(
                args,
                [
                    "table_name",
                    "target_column",
                    "index_type",
                    "m",
                    "ef_construction",
                    "num_lists",
                ],
            )
        ),
        "test_connection": lambda: None,
    }

    for operation, fn in dispatch_table.items():
        if getattr(args, operation, False):
            print(f"Executing {operation}...")
            fn()
            return

    print("No valid operation specified")


if __name__ == "__main__":
    main()
