import argparse
import json
import logging
import numpy as np
import os
from datetime import datetime
import inspect

import pandas as pd
import psycopg
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


logger = logging.getLogger(__name__)

"""
USAGE:
python database.py --test-connection
python database.py --create-base-table --table-name lib --from-path="../data/preprocessed/research.jsonl"
python database.py --create-vector-column --table-name lib --target-column chunk --embedder-name "BAAI/bge-small-en" [--normalize] --batch-size 16
python database.py --create-index --table-name lib --target-column bge_norm --index-type ivfflat --num-lists 1472 [--m 32 --ef-construction 512]
"""


def argument_parser():
    parser = argparse.ArgumentParser(description="Database operations")

    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument("--test-connection", "-t", action="store_true", help="Test database connection")
    operation_group.add_argument(
        "--create-vector-table",
        "-V",
        action="store_true",
        help="Create a new vector table",
    )
    operation_group.add_argument("--create-index", "-I", action="store_true", help="Create an index on a table")
    operation_group.add_argument("--add-chunks", "-A", action="store_true", help="Add chunks to the database")
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
        default=16,
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
    parser.add_argument("--m", "-M", default=32, type=int, help="M parameter for HNSW (default: 32)")
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
        default=1472,
        type=int,
        help="Number of lists for IVFFlat (default: 1472)",
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

    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level [default: INFO, DEBUG, WARNING, ERROR]",
    )

    args = parser.parse_args()

    # Validate args
    if args.create_base_table and not all([args.from_path, args.target_table]):
        parser.error("--create-base-table requires --table-name and --from-path")

    elif args.create_vector_column and not all([args.target_table, args.embedder_name]):
        parser.error("--create-vector-column requires --table-name and --embedder")

    elif args.create_vector_table and not all([args.target_table, args.embedder_name]):
        parser.error("--create-vector-table requires --table-name and --embedder")

    elif args.create_index:
        if not all([args.target_table, args.target_column, args.index_type]):
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

# TODO: factor these out to models?


@dataclass
class VectorQueryResult:
    chunk_id: int
    doi: str
    title: str
    abstract: str
    chunk: str
    pubdate: str
    distance: float


@dataclass
class VectorSearchResult:
    """
    Currently this represents the 'contributions' table, which represents a paper by embedding the original
    contributions in the paper (as written by an LLM)
    """

    text: str
    doi: str
    pubdate: str
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


# Consumer function
def consumer_proc(db_params, results_queue, embedder_dim, target_table, vector_column_name, progress_queue):
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
        with cur.copy("COPY temp_embeddings (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for row_id, embedding in zip(batch_ids, batch_embeddings):
                copy.write_row([row_id, embedding])

        cur.execute(
            f"UPDATE {target_table} SET {vector_column_name} = temp.embedding "
            f"FROM temp_embeddings temp WHERE {target_table}.id = temp.id"
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
        "BAAI/bge-large-en-v1.5": "bge_large",
        "bert-base-uncased": "bert",
        "adsabs/astroBERT": "astrobert",
        "nasa-impact/nasa-ibm-st.38m": "nasa",
        "Qwen/Qwen3-Embedding-0.6B": "qwen",
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
        required_vars = [
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD",
            "DB_HOST",
            "DB_PORT",
            "DB_MEM",
            "DB_CPUS",
        ]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

        # Connect to the database
        self.db_params = Database.get_db_params(path_to_env=path_to_env)
        self.conn = psycopg.connect(**self.db_params)
        register_vector(self.conn)

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
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

    def set_session_resources(self, optimize_for: Literal["query", "index", "insert"], verbose=True):
        """
        Set session resources for PostgreSQL

        Also good settings for postegresql.conf (presumably):
        wal_level = 'minimal'; -- minimal amount of data to write to WAL logs
            max_wal_senders = 0; -- required by wal_level = minimal, meaning no replication
        shared_buffers = 'xxGB'; -- amount of memory to use (50% of host if you're the only user)
        """
        db_mem = round(0.9 * float(os.getenv("DB_MEM")), 2)  # In GB. Leave 10% for OS overhead
        db_cpus = int(os.getenv("DB_CPUS")) - 2  # Leave 2 CPUs for OS overhead

        if optimize_for == "query":
            query = f"""
                -- SET synchronous_commit = 'off';
                SET maintenance_work_mem = '1GB';
                -- SET random_page_cost = '1.1';
                -- SET parallel_tuple_cost = '0.1';
                -- SET parallel_setup_cost = '1000';
                SET max_parallel_workers = {max(1, db_cpus)};
                SET work_mem = '{max(1, db_mem // (db_cpus * 2))}GB';
                SET max_parallel_workers_per_gather = {max(1, db_cpus)};
                SET effective_cache_size = '{int(0.75 * db_mem)}GB';
            """

        elif optimize_for == "index":
            query = f"""
                SET synchronous_commit = 'off';
                SET work_mem = '{max(int(0.25 * db_mem), 4)}GB';
                SET maintenance_work_mem = '{int(0.15 * db_mem)}GB';
                -- SET checkpoint_timeout = '30min';
                SET max_parallel_workers_per_gather = '{db_cpus}';  -- Use all CPUs for query parallelism
                SET max_parallel_maintenance_workers = '{int(0.75 * db_cpus)}';  -- Use 75% of CPUs for index creation (reduces contention)
                -- SET shared_buffers = '{int(0.25 * db_mem)}GB';
                SET effective_cache_size = '{int(0.5 * db_mem)}GB';
            """

        elif optimize_for == "insert":
            query = f"""
                SET synchronous_commit = 'on';
                SET work_mem = '{max(int(0.002 * db_mem), 0.5)}GB';
                SET maintenance_work_mem = '4MB';
                -- SET checkpoint_completion_target = '0.9';
                -- SET wal_writer_delay = '200ms';
                SET max_parallel_workers_per_gather = '0';
                SET max_parallel_maintenance_workers = '0';
                SET effective_cache_size = '{int(0.5 * db_mem)}GB';
            """
        if verbose:
            print(f"Executing query:\n{query}")
        with self.conn.cursor() as cursor:
            cursor.execute(query)

    def _create_base_table(
        self,
        target_table: str,
        from_path: str,
        max_length: int = 1500,
        overlap: int = 150,
    ):
        """
        Create the base table for chunks and insert records into it.

        Args:
            target_table (str): Name of the table to create.
            from_path (str): Path to the dataset (JSONL file).
            max_length (int): Maximum length of each chunk.
            overlap (int): Overlap between chunks.
        """
        # First, create table so we immediately error if it already exists
        cursor = self.conn.cursor()
        cursor.execute(
            f"""CREATE TABLE {target_table} (
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
            raise ValueError(f"Input JSONL file must contain the following columns: {required_columns}")

        # Split body into chunks
        records = df.to_dict("records")
        self.splitter = TextSplitter(capacity=max_length, overlap=overlap)

        # Use ProcessPoolExecutor to parallelize chunking
        chunked_records = []
        cores = max(1, os.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=cores) as process_executor:
            futures = [
                process_executor.submit(record_to_chunked_records, record, max_length, overlap) for record in records
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunking records"):
                chunked_records.extend(future.result())  # Collect all chunks

        try:
            with cursor.copy(
                f"COPY {target_table} (doi, title, abstract, pubdate, keywords, chunk) FROM STDIN WITH (FORMAT TEXT)"
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

    def get_paper_by_doi(self, doi: str) -> str:
        """
        Get the full text of a paper by its DOI. Its structure will be:
        Title: {title}

        Abstract: {abstract}

        {body}
        """

        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT title, abstract, body FROM papers WHERE doi = %s;
                """,
                (doi,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No paper found with DOI: {doi}")
            title, abstract, body = row

        return f"Title: {title}\n\nAbstract: {abstract}\n\n{body}"

    def get_chunks_by_doi(self, doi: str, target_table: str = "library", vector_column: str = "bge_norm"):
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT id, doi, title, abstract, chunk, {vector_column} FROM {target_table}
            WHERE doi = '{doi}';
            """
        )
        rows = cursor.fetchall()
        return [SingleVectorDoiResult(*row) for row in rows]

    def insert_document_expansions(
        self,
        expansions: list[str],
        doi: str,
        embedder_name: str,
        normalize: bool,
        table: str = "contributions",  # TODO: change this to something more generic?
        batch_size: int = 16,
    ):
        """
        Insert document expansions into the database. For example, a research paper can be
        represented by a list of its original findings (the expansions list of strings). The
        embedder then embeds these into vectors and inserts them into the database.


        Args:
            expansions (list[str]): List of document expansions to insert.
            doi (str): DOI of the document these expansions belong to.
            embedder_name (str): Name of the embedder to use for vectorization.
            target_column (str): Column in the database where the vectors will be stored.
            batch_size (int): Number of expansions to process in each batch.
        """
        from embedders import get_embedder

        embedder = get_embedder(embedder_name, self.device, normalize)
        embeddings = embedder(expansions)
        print(f"Got embeddings ({embeddings.shape})")
        # Prepare data for insertion
        data = [(doi, exp, emb.tolist()) for exp, emb in zip(expansions, embeddings)]
        # Insert into the database
        with self.conn.cursor() as cursor:
            # Insert in batches
            for i in tqdm(range(0, len(data), batch_size), desc="Inserting document expansions"):
                batch = data[i : i + batch_size]
                cursor.executemany(
                    f"""
                    INSERT INTO {table} (doi, text, embedding)
                    VALUES (%s, %s, %s)
                    """,
                    batch,
                )
            self.conn.commit()

    # def create_vector_column(
    #     self,
    #     embedder_name: str,
    #     normalize: bool = False,
    #     enricher_name: str = None,
    #     target_table: str = "lib",
    #     target_column: str = "chunk",
    #     batch_size: int = 16,
    # ):
    #     """
    #     Create a new column in the specified table to store vector embeddings using threads.
    #     This is more efficient for database operations than using processes.
    #     """
    #     from embedders import get_embedder
    #     from concurrent.futures import ThreadPoolExecutor
    #     import queue

    #     # Initialize the embedder in the main process
    #     logger.info(f"Using embedder: {embedder_name}")
    #     embedder = get_embedder(embedder_name, self.device)
    #     dim = embedder.dim

    #     if enricher_name:
    #         from TextEnrichers import get_enricher

    #         enricher = get_enricher(name=enricher_name, path_to_data="../data/preprocessed/research.jsonl")

    #     # Construct column name
    #     vector_column_name = Database.EMBEDDER_SHORTNAMES.get(
    #         embedder_name, embedder_name.replace("/", "_").replace("-", "_")
    #     )
    #     if normalize:
    #         vector_column_name += "_norm"
    #     if enricher_name:
    #         vector_column_name += f"_{enricher_name}"

    #     self.set_session_resources(optimize_for="insert", verbose=True)

    #     logger.debug(f"Attempting to create column '{vector_column_name}' in table '{target_table}'...")
    #     # Register vector extension and create column
    #     query = f"ALTER TABLE {target_table} ADD COLUMN IF NOT EXISTS {vector_column_name} VECTOR({dim});"
    #     logger.debug(f"Executing query: {query}")
    #     with self.conn.cursor() as cursor:
    #         cursor.execute(query)
    #         self.conn.commit()

    #         # Get all text chunks to embed
    #         start = time()
    #         cursor.execute(f"SELECT id, doi, {target_column} FROM {target_table};")
    #         rows = cursor.fetchall()
    #         logger.debug(
    #             f"Fetched {len(rows)} rows from the database. SELECT execution time: {time() - start:.2f} seconds"
    #         )
    #         all_ids, all_dois, all_chunks = zip(*rows)
    #         del rows

    #     # Enrich chunks if an enricher is provided
    #     # if enricher_name:
    #     #     texts_with_dois = zip(all_chunks, all_dois)
    #     #     all_chunks = enricher.enrich_batch(texts_with_dois=texts_with_dois)

    #     # Create thread-safe queues for tasks and progress tracking
    #     task_queue = queue.Queue()
    #     progress_queue = queue.Queue()

    #     # Calculate number of batches
    #     total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    #     num_workers = int(os.getenv("CPUS", max(1, os.cpu_count()))) - 1

    #     # Thread worker function
    #     def consumer_thread():
    #         # Get a database connection from the connection pool or create a new one
    #         conn = psycopg.connect(**self.db_params)
    #         register_vector(conn)

    #         with conn.cursor() as cur:
    #             # Settings for better INSERT performance
    #             cur.execute("SET work_mem='2GB';")
    #             cur.execute("SET maintenance_work_mem='2GB';")

    #             # Create temp table for efficient batch updates
    #             cur.execute(f"CREATE TEMP TABLE temp_embeddings (id int, embedding vector({dim}))")
    #             conn.commit()

    #             while True:
    #                 try:
    #                     # Get a batch from the queue
    #                     item = task_queue.get(block=True, timeout=5)
    #                     if item is None:  # Sentinel value to signal end of work
    #                         task_queue.task_done()
    #                         break

    #                     batch_ids, batch_embeddings = item

    #                     # Use COPY protocol for fast insertion
    #                     with cur.copy("COPY temp_embeddings (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
    #                         copy.set_types(["int4", "vector"])
    #                         for row_id, embedding in zip(batch_ids, batch_embeddings):
    #                             copy.write_row([row_id, embedding])

    #                     # Update the main table from the temp table
    #                     cur.execute(
    #                         f"UPDATE {target_table} SET {vector_column_name} = temp.embedding "
    #                         f"FROM temp_embeddings temp WHERE {target_table}.id = temp.id"
    #                     )
    #                     cur.execute("TRUNCATE temp_embeddings")
    #                     conn.commit()

    #                     # Report progress
    #                     progress_queue.put(1)
    #                     task_queue.task_done()

    #                 except queue.Empty:
    #                     # No more tasks for a while, check if we should exit
    #                     if task_queue.empty():
    #                         break

    #         # Clean up
    #         conn.close()

    #     # Start the thread pool
    #     with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         # Start worker threads
    #         for _ in range(num_workers):
    #             executor.submit(consumer_thread)
    #         print(f"Started thread pool with {num_workers} workers...")

    #         # Producer in the main thread (GPU operations)
    #         with tqdm(total=total_batches, desc="Embedding and writing to database") as progress_bar:
    #             processed = 0

    #             # Process batches
    #             for i in range(0, len(all_chunks), batch_size):
    #                 texts, ids = all_chunks[i : i + batch_size], all_ids[i : i + batch_size]
    #                 embeddings = embedder(texts)
    #                 task_queue.put((ids, embeddings))

    #                 # Check progress queue for updates
    #                 try:
    #                     while not progress_queue.empty():
    #                         progress = progress_queue.get_nowait()
    #                         processed += progress
    #                         progress_bar.update(progress)
    #                 except queue.Empty:
    #                     pass

    #             # Finishing up: add sentinel values to signal workers to exit
    #             for _ in range(num_workers):
    #                 task_queue.put(None)
    #             task_queue.join()

    #             # Final progress update
    #             try:
    #                 while not progress_queue.empty():
    #                     progress = progress_queue.get_nowait()
    #                     processed += progress
    #                     progress_bar.update(progress)
    #             except queue.Empty:
    #                 pass

    #     print(f"Successfully created and populated vector column: {vector_column_name}")

    def create_index(
        self,
        target_table: str,
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
        self.set_session_resources(optimize_for="index", verbose=True)

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
        query = f"CREATE INDEX {index_name} ON {target_table} USING {index_type} ({target_column} vector_cosine_ops) WITH {parameters}"
        print(f"Executing query: {query}")
        start = time()
        cursor.execute(query)
        print(f"  Index creation time: {time() - start:.2f} seconds")
        self.conn.commit()

        # Cleanup
        end = time()
        print(f"Created {index_type} index on {target_table}.{target_table} in {end - start:.2f} seconds")
        cursor.close()

    def query(self, q: str) -> list[tuple]:
        """
        Directly execute a sql query

        TODO: Be sure to remove this from the final version. This is only for development

        """

        cursor = self.conn.cursor()
        cursor.execute(q)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def vector_search(
        self,
        query_vector: np.array,
        target_table: str,
        target_column: str,
        metric: str = "vector_cosine_ops",
        pubdate: str | None = None,
        top_k=10,
        use_index=True,
        probes: int = None,
        ef_search: int = None,
    ) -> pd.DataFrame:
        """
        Query the specified vector column in the database.

        Args:
            query_vector (np.array): The vector to query.
            target_table (str): The name of the table to query.
            target_column (str): The name of the column to query.
            metric (str): The distance metric to use. Default is 'vector_cosine_ops'.
            pubdate (str | None): Optional publication date filter in YYYY-MM-DD format.
            use_index (bool): Whether to use the index for the query. Default is True.
            top_k (int): Number of nearest neighbors to return. Default is 5.
            probes (int): Number of probes for IVFFlat index. Default is 40.

        Returns:
            pd.DataFrame: A DataFrame containing the results. Columns=["text", "doi", "pubdate", "distance"]
        """
        # Set up operator string, session resources, and pubdate format
        _operator_ = self.PGVECTOR_DISTANCE_OPS[metric]
        self.set_session_resources(optimize_for="query", verbose=False)

        # Convert pubdate to string in YYYY-MM-DD format
        if not pubdate:
            pubdate_str = datetime.now().strftime("%Y-%m-%d")
        elif isinstance(pubdate, pd.Timestamp):
            pubdate_str = pubdate.strftime("%Y-%m-%d")  # Convert to clean YYYY-MM-DD
        elif isinstance(pubdate, str):
            try:
                datetime.strptime(pubdate, "%Y-%m-%d")
                pubdate_str = pubdate
            except ValueError:
                raise ValueError(f"Invalid pubdate format: {pubdate}. Use YYYY-MM-DD format.")
        else:
            pubdate_str = str(pubdate)[:10]  # Fallback: take first 10 chars

        with self.conn.cursor() as cursor:
            # Vector index parameters
            if probes:
                cursor.execute(f"SET ivfflat.probes = {probes};")
            elif ef_search:
                cursor.execute(f"SET hnsw.ef_search = {ef_search};")

            cursor.execute(
                f"""
                SELECT
                    text, doi, pubdate, {target_column} {_operator_} %s AS distance
                FROM {target_table}
                WHERE pubdate <= '{pubdate_str}'
                ORDER BY distance ASC
                LIMIT {top_k};
                """,
                (query_vector,),
            )
            results = cursor.fetchall()
            return pd.DataFrame(results, columns=["text", "doi", "pubdate", "distance"])

    # def query_vector_column(
    #     self,
    #     query_vector: np.array,
    #     target_table: str,
    #     target_column: str,
    #     metric: str = "vector_cosine_ops",
    #     pubdate: str | None = None,
    #     use_index=True,
    #     top_k=5,
    #     probes: int = None,
    #     ef_search: int = None,
    # ) -> list[VectorQueryResult]:
    #     """
    #     Query the specified vector column in the database.

    #     Args:
    #         query_vector (np.array): The vector to query.
    #         target_table (str): The name of the table to query.
    #         target_column (str): The name of the column to query.
    #         metric (str): The distance metric to use. Default is 'vector_cosine_ops'.
    #         pubdate (str | None): Optional publication date filter in YYYY-MM-DD format.
    #         use_index (bool): Whether to use the index for the query. Default is True.
    #         top_k (int): Number of nearest neighbors to return. Default is 5.
    #         probes (int): Number of probes for IVFFlat index. Default is 40.

    #     Returns:
    #         list[VectorQueryResult]: A list of VectorQueryResult objects containing the results.
    #     """
    #     # Set up operator string, session resources, and pubdate format
    #     _operator_ = self.PGVECTOR_DISTANCE_OPS[metric]
    #     self.set_session_resources(optimize_for="query", verbose=False)
    #     if not pubdate:
    #         # Set to today in YYY-MM-DD format
    #         pubdate = datetime.now().strftime("%Y-%m-%d")
    #     else:
    #         try:
    #             datetime.strptime(pubdate, "%Y-%m-%d")
    #         except ValueError:
    #             raise ValueError(f"Invalid pubdate format: {pubdate}. Use YYYY-MM-DD format.")

    #     with self.conn.cursor() as cursor:
    #         # Vector index parameters
    #         if probes:
    #             cursor.execute(f"SET ivfflat.probes = {probes};")
    #         elif ef_search:
    #             cursor.execute(f"SET hnsw.ef_search = {ef_search};")

    #         cursor.execute(
    #             f"""
    #             SELECT
    #                 id, doi, title, abstract, chunk, pubdate, {target_column} {_operator_} %s AS distance
    #             FROM {target_table}
    #             WHERE pubdate <= '{pubdate}'
    #             ORDER BY distance ASC
    #             LIMIT {top_k};
    #             """,
    #             (query_vector,),
    #         )
    #         results = cursor.fetchall()
    #         # return [VectorQueryResult(*result) for result in results]
    #         return pd.DataFrame(results, columns=["text", "doi", "pubdate", "distance"])

    def prewarm_table(self, target_table: str, target_column: str = None):
        """
        Prewarms a table and optionally, specific indexes associated with a target column.

        Args:
            target_table (str): The name of the table to prewarm.
            target_column (str, optional): The name of the column whose indexes should be prewarmed.
                                           If None, all indexes on the table are prewarmed. Defaults to None.
        """
        prewarmed_objects = []

        try:
            with self.conn.cursor() as cursor:
                # Prewarm pubdate index
                cursor.execute("SELECT pg_prewarm('idx_lib_pubdate');")
                results = cursor.fetchall()
                prewarmed_objects.append(("idx_pubdate", results[0][0] if results and results[0][0] else 0))

                # Prewarm table
                cursor.execute(f"SELECT pg_prewarm('{target_table}');")
                results = cursor.fetchall()
                prewarmed_objects.append((target_table, results[0][0] if results and results[0][0] else 0))

                # Prewarm index on target column (the vector column)
                if target_column:
                    cursor.execute(
                        f"""
                        SELECT i.relname AS index_name
                        FROM pg_index ix
                        JOIN pg_class i ON i.oid = ix.indexrelid
                        JOIN pg_class t ON t.oid = ix.indrelid
                        JOIN pg_attribute a ON a.attrelid = ix.indrelid
                        WHERE t.relname = '{target_table}'
                        AND a.attname = '{target_column}'
                        AND a.attnum = ANY(ix.indkey);
                        """
                    )
                    indexes = [row[0] for row in cursor.fetchall()]
                    for index in indexes:
                        cursor.execute(f"SELECT pg_prewarm('{index}');")
                        results = cursor.fetchall()
                        prewarmed_objects.append((index, results[0][0] if results and results[0][0] else 0))

                    # Get sizes of prewarmed objects
                    cursor.execute(
                        f"""
                        SELECT relname, pg_relation_size(oid) AS bytes
                        FROM pg_class
                        WHERE relname = '{target_table}'
                        OR relname = ANY(%s);
                        """,
                        (indexes,),
                    )
                    size_data = {row[0]: row[1] for row in cursor.fetchall()}

            # Print summary table
            print("\n" + "=" * 70)
            print(f"{'Object Name':<30} {'Blocks Loaded':<15} {'Size (GB)':<15}")
            print("-" * 70)

            total_blocks = 0
            total_size_bytes = 0
            for obj_name, blocks in prewarmed_objects:
                blocks = blocks if blocks is not None else 0
                total_blocks += blocks

                # Get size in GB
                size_bytes = size_data.get(obj_name, 0)
                total_size_bytes += size_bytes
                size_gb = size_bytes / (1024 * 1024 * 1024)

                print(f"{obj_name:<30} {blocks:<15} {size_gb:.4f} GB")

            total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
            print("-" * 70)
            print(f"{'TOTAL':<30} {total_blocks:<15} {total_size_gb:.4f} GB")
            print("=" * 70 + "\n")

        except Exception as e:
            print(f"Error prewarming table {target_table}: {e}")
            raise e

    def get_reconstructed_paper(self, doi: str) -> str:
        with self.conn.cursor() as cursor:
            query = "SELECT title, abstract, body FROM papers WHERE doi = %s"
            cursor.execute(query, (doi,))
            result = cursor.fetchone()
            if result:
                title, abstract, body = result
                return f"{title}\n\nAbstract: {abstract}\n\n{body}"
            else:
                raise ValueError(f"No paper found for DOI: {doi}")

    def explain_analyze(
        self,
        query_vector: str,
        target_table: str,
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

        self.prewarm_table(target_table)

        # Prepare and execute query (format everything but the vector first, so it can be printed compactly)
        query = """EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON)
            SELECT {target_table}.chunk_id, chunks.doi, chunks.text, {target_table}.embedding {operator} '{query_vector}' AS distance
            FROM {target_table}
            JOIN chunks ON {target_table}.chunk_id = chunks.id
            ORDER BY {target_table}.embedding {operator} '{query_vector}' ASC
            LIMIT {top_k};
            """.format(
            target_table=target_table,
            operator=operator,
            top_k=top_k,
            query_vector="{query_vector}",
        )
        print(f"Executing query: {query}")
        cursor.execute(query.format(query_vector=query_vector))

        # Report
        query_plan = cursor.fetchall()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"explain_analyze_{target_table}_topk{top_k}_{current_time}.json"
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


def profile_create_vector_table(db, target_table, embedder):
    with cProfile.Profile() as pr:
        db.create_vector_table(target_table=target_table, dim=768, embedder=embedder)
    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats("cumulative").print_stats(40)


def get_kwargs(args, values):
    return {k: v for k, v in vars(args).items() if k in values}


def main():
    args = argument_parser()
    logging.basicConfig(
        filename="logs/database.log",
        filemode="w",
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    db = Database()
    db.test_connection()

    dispatch_table = {
        "create_base_table": lambda: db._create_base_table(
            **get_kwargs(args, ["target_table", "from_path", "max_length", "overlap"])
        ),
        "create_vector_column": lambda: db.create_vector_column(
            **get_kwargs(
                args,
                [
                    "target_table",
                    "embedder_name",
                    "normalize",
                    "enricher_name",
                    "target_column",
                    "batch_size",
                ],
            )
        ),
        "create_vector_column_threaded": lambda: db.create_vector_column_threaded(
            **get_kwargs(
                args,
                [
                    "target_table",
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
                    "target_table",
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
