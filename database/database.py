import argparse
import gc
import json
import numpy as np
import os
import psycopg2
from psycopg2 import Binary
from psycopg2.extras import execute_values
import sys
import torch
from dataclasses import dataclass
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from semantic_text_splitter import TextSplitter
from time import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


from time import time

import cProfile
import pstats

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# fmt: off
# NOTE: moved into chunk_and_insert since it's not used elsewhere
# from preprocessing import load_dataset

# fmt: on

PGVECTOR_DISTANCE_METRICS = {
    'vector_l2_ops': '<->',
    'vector_ip_ops': '<#>',
    'vector_cosine_ops': '<=>',
}


def argument_parser():
    parser = argparse.ArgumentParser(description='Database operations')

    # # Create subparsers for different operations
    # subparsers = parser.add_subparsers(dest='operation', required=True,
    #                                    help='Operation to perform')

    # # Create vector table operation
    # create_vector_parser = subparsers.add_parser('create-vector-table',
    #                                              help='Create a new vector table')

    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        '--test-connection', '-t',
        action='store_true',
        help='Test database connection'
    )
    operation_group.add_argument(
        '--create-vector-table', '-V',
        action='store_true',
        help='Create a new vector table'
    )
    operation_group.add_argument(
        '--create-index', '-I',
        action='store_true',
        help='Create an index on a table'
    )
    operation_group.add_argument(
        '--add-chunks', '-A',
        action='store_true',
        help='Add chunks to the database'
    )

    # Create vector table arguments
    parser.add_argument(
        '--table-name', '-T',
        type=str,
        help='Name of target table'
    )
    parser.add_argument(
        '--embedder', '-e',
        type=str,
        default='BAAI/bge-small-en',
        help='Name of the embedding model to use (default: BAAI/bge-small-en)'
    )
    parser.add_argument(
        '--normalize', '-n',
        default=False,
        action='store_true',
        help='Normalize embeddings before inserting into the database'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for embedding and insertion operations'
    )

    # Create index arguments
    # --table-name already defined
    parser.add_argument(
        '--index-type', '-i',
        default='hnsw',
        type=str,
        help='Type of index to create (default: hnsw)'
    )
    parser.add_argument(
        '--metric', '-m',
        default='vector_cosine_ops',
        type=str,
        help='Distance metric to use for the index (default: vector_cosine_ops)'
    )
    parser.add_argument(
        '--m', '-M',
        default=32,
        type=int,
        help='M parameter for HNSW (default: 32)'
    )
    parser.add_argument(
        '--ef-construction', '-ef',
        default=512,
        type=int,
        help='efConstruction parameter for HNSW (default: 512)'
    )
    parser.add_argument(
        '--num-lists', '-l',
        default=1580,
        type=int,
        help='Number of lists for IVFFlat (default: 1580)'
    )

    # Add chunks arguments
    parser.add_argument(
        '--path', '-p',
        type=str,
        help='Path to the dataset to add to the database'
    )
    parser.add_argument(
        '--max-length', '-L',
        type=int,
        default=1500,
        help='Maximum length of each chunk'
    )
    parser.add_argument(
        '--overlap', '-o',
        type=int,
        default=150,
        help='Overlap between chunks'
    )

    # TODO: update this or fix it
    # Profile operation
    """
    profile_parser = subparsers.add_parser(
        'profile', help='Profile a function')
    profile_parser.add_argument(
        '--function', '-f',
        required=True,
        type=str,
        help='Name of the function to profile'
    )
    """
    args = parser.parse_args()

    # Validate args
    if args.create_vector_table and not all([args.table_name, args.embedder]):
        parser.error(
            "--create-vector-table requires --table-name and --embedder")

    if args.create_index:
        if not args.index_type or not args.table_name:
            parser.error(
                "--create-index requires --index-type and --table-name")
        if args.index_type == 'ivfflat' and notall([args.num_lists]):
            parser.error("--create-index requires --num-lists for ivfflat")
        if args.index_type == 'hnsw' and not all([args.m, args.ef_construction]):
            parser.error(
                "--create-index requires --m and --ef-construction for hnsw")

    if args.add_chunks and not all([args.path, args.max_length, args.overlap]):
        parser.error(
            "--add-chunks requires --path, --max-length, and --overlap")

    return args


"""
DATACLASSES
"""


@dataclass
class SingleQueryResult:
    chunk_id: int
    doi: str
    text: str
    distance: float


@dataclass
class ChunkAndVector:
    text: str
    vector: np.array


"""
DATABASE RELATED FUNCTIONS FOR EXPORT
"""


def get_db_params(env_file: str = '.env') -> dict[str, str]:
    """
    Load database parameters from a .env file
    """
    load_dotenv(env_file, override=True)
    return {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }


"""
Helper functions for vector table insertion. These must be outside the class definition to be pickled,
which is necessary for multiprocessing.
"""


def insert_batch(data_name_and_processor):
    data, name, processor = data_name_and_processor
    conn = psycopg2.connect(**processor.db_params)
    cursor = conn.cursor()
    execute_values(
        cursor,
        f"INSERT INTO {name} (embedding, chunk_id) VALUES %s;",
        data
    )
    conn.commit()
    cursor.close()
    conn.close()


class DatabaseProcessor:
    def __init__(self, db_params):
        self.db_params = db_params
        self.device = 'cuda' if torch.cuda.is_available(
        ) else 'mps' if torch.mps.is_available() else 'cpu'

    def __clear_gpu_memory(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()

    def __remove_nul_chars(self, s: str) -> str:
        """Remove NUL characters from a string, which PostgreSQL does not like"""
        return s.replace('\x00', '')

    def _chunk_record(self, record: dict, max_length: int, overlap: int):
        splitter = TextSplitter(capacity=max_length, overlap=overlap)
        full_text = record['title'] + '\n\nABSTRACT:\n' + \
            record['abstract'] + '\n\n' + record['body']
        chunks = splitter.chunks(full_text)
        doi = record['doi'][0]
        return [(chunk, doi) for chunk in chunks]

    def _insert_chunk(data, table_name, db_params):
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        execute_values(
            cursor,
            f"INSERT INTO {table_name} (embedding, chunk_id) VALUES %s;",
            data
        )
        conn.commit()
        cursor.close()
        conn.close()

    def chunk_and_insert_records(self, path: str, max_length: int = 1500, overlap: int = 150):
        from preprocessing import load_dataset
        records = load_dataset(path)
        all_chunks = []

        # Use ProcessPoolExecutor to parallelize chunking
        cores = min(1, os.cpu_count() - 2)
        with ProcessPoolExecutor(max_workers=cores) as process_executor:
            futures = [process_executor.submit(
                self._chunk_record, record, max_length, overlap) for record in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunking records"):
                all_chunks.extend(future.result())  # Collect all chunks

        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        try:
            batch_size = 1000  # Adjust this based on your needs
            for i in tqdm(range(0, len(all_chunks), batch_size), desc="Inserting chunks"):
                batch = all_chunks[i:i + batch_size]
                data = [(self.__remove_nul_chars(chunk), doi)
                        for chunk, doi in batch]
                execute_values(
                    cursor,
                    "INSERT INTO chunks (text, doi) VALUES %s;",
                    data
                )
                conn.commit()
        finally:
            cursor.close()
            conn.close()

    def create_vector_table_mp(self, name, dim, embedder):
        """
        1. Creates a vector table
        2. Creates indexes for all distance metrics
        3. Batch embeds all records in the `chunks` table using the given embedder
        """
        """
        1. Creates a vector table
        2. Creates indexes for all distance metrics
        3. Batch embeds all records in the `chunks` table using the given embedder

        available distance metrics:
        vector_l2_ops, vector_ip_ops, vector_cosine_ops, vector_l1_ops, bit_hamming_ops, bit_jaccard_ops
        """
        conn = psycopg2.connect(**self.db_params)
        register_vector(conn)
        cursor = conn.cursor()
        cursor.execute(
            f"""CREATE TABLE {name} (
                id SERIAL PRIMARY KEY,
                embedding VECTOR({dim}),
                chunk_id INTEGER REFERENCES chunks(id)
                );
            """)
        conn.commit()
        print(f"Created table {name}")

        # Create indexes
        # for metric in PGVECTOR_DISTANCE_METRICS:
        #     print(f"Creating index on {metric}...", end="")
        #     cursor.execute(
        #         f"CREATE INDEX ON {name} USING hnsw (embedding {metric})")
        #     print("done.")
        conn.commit()

        # Get all chunks for embedding
        ids_and_chunks = self._get_all_chunks(cursor)
        print(f"Embedding {len(ids_and_chunks)} chunks...")

        # Process in larger batches for GPU efficiency
        batch_size = 512
        num_batches = len(ids_and_chunks) // batch_size + 1

        for i in tqdm(range(num_batches), desc="Processing batches"):
            # Prepare batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(ids_and_chunks))
            batch = ids_and_chunks[start_idx:end_idx]

            if not batch:
                continue

            ids, texts = list(zip(*batch))
            # Embeddings are automatically moved to CPU
            embeddings = embedder(texts)

            # Insert using ThreadPoolExecutor for parallel DB operations
            data = [(emb, id_) for emb, id_ in zip(embeddings, ids)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Split data for parallel insertion
                chunk_size = len(data) // 4
                data_chunks = [data[j:j + chunk_size]
                               for j in range(0, len(data), chunk_size)]

                futures = [
                    executor.submit(insert_batch, (chunk, name, self))
                    for chunk in data_chunks if chunk
                ]
                list(tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Inserting batch",
                    leave=False
                ))
        cursor.close()
        conn.close()

    def create_vector_table(self,
                            table_name: str,
                            dim: int,
                            embedder,  # Embedder
                            work_mem='2048MB',
                            maintenance_work_mem='2048MB',
                            batch_size=32):
        """
        1. Creates a vector table
        2. Creates indexes for all distance metrics
        3. Batch embeds all records in the `chunks` table using the given embedder

        available distance metrics:
        vector_l2_ops, vector_ip_ops, vector_cosine_ops, vector_l1_ops, bit_hamming_ops, bit_jaccard_ops
        """

        # Create connection and allocate session memory
        conn = psycopg2.connect(**self.db_params)
        register_vector(conn)
        cursor = conn.cursor()

        # Set session resources
        cores = os.cpu_count()
        cursor.execute("SHOW max_worker_processes;")
        max_worker_processes = int(cursor.fetchone()[0])
        max_parallel_workers = max(1, cores - 2)
        max_parallel_maintenance_workers = int(0.2 * max_worker_processes)
        print("="*33 + "CONFIG" + "="*33)
        print("max_worker_processes | max_parallel_workers | max_parallel_maintenance_workers | work_mem | maintenance_work_mem")
        print(
            f"{max_worker_processes:^21} {max_parallel_workers:^22} {max_parallel_maintenance_workers:^34} {work_mem:^10} {maintenance_work_mem:^21}")
        print("="*72)
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")
        cursor.execute(
            f"SET max_parallel_maintenance_workers={max_parallel_maintenance_workers};")
        cursor.execute(f"SET work_mem='{work_mem}';")
        cursor.execute(f"SET maintenance_work_mem='{maintenance_work_mem}';", )

        # Create table
        cursor.execute(
            f"""CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                embedding VECTOR({dim}),
                chunk_id INTEGER REFERENCES chunks(id)
                );
            """)
        conn.commit()
        print(f"Created table {table_name}")

        # Get all chunks for embedding
        ids_and_chunks = self._get_all_chunks(cursor)
        print(f"Embedding {len(ids_and_chunks)} chunks...")

        # Embed an insert in batches
        num_batches = 1 + len(ids_and_chunks) // batch_size
        for i in tqdm(range(num_batches), desc="Inserting embeddings", leave=False):
            # Clear GPU memory every 50 batches
            if i % 50 == 0:
                self.__clear_gpu_memory()

            # Prepare batch
            batch = ids_and_chunks[i * batch_size:(i + 1) * batch_size]
            ids, texts = list(zip(*batch))

            # Embed texts
            start_embeddings = time()
            embeddings = embedder(texts)
            print(f"Embedding time: {time() - start_embeddings}")

            data = [(embedding, id_num)
                    for embedding, id_num in zip(embeddings, ids)]

            # Insert
            start_insert = time()
            execute_values(
                cursor, f"INSERT INTO {table_name} (embedding, chunk_id) VALUES %s;", data)
            print(f"Insert time: {time() - start_insert}")
            conn.commit()

        cursor.close()

    def create_index(self,
                     table_name: str,
                     index_type: str,  # 'ivfflat' or 'hnsw'
                     metric: str,
                     num_lists: int = 1580,  # sqrt(num chunks, which is ~2.5M)
                     m: int = 32,
                     ef_construction: int = 512):
        # Check input
        assert index_type in [
            'ivfflat', 'hnsw'], f"Invalid index type: {index_type}. Must be 'ivfflat' or 'hnsw'"
        assert metric in PGVECTOR_DISTANCE_METRICS, f"Invalid metric: {metric}. I don't have that metric in the PGVECTOR_DISTANCE_METRICS dictionary"
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        # Set session resources
        cores = os.cpu_count()
        max_worker_processes = max_parallel_workers = max(1, cores - 2)
        max_parallel_maintenance_workers = int(0.8 * max_worker_processes)
        maintenance_work_mem = '4GB'
        print("="*33 + "CONFIG" + "="*33)
        print("max_worker_processes | max_parallel_workers | max_parallel_maintenance_workers | maintenance_work_mem")
        print(
            f"{max_worker_processes:^21} {max_parallel_workers:^21} {max_parallel_maintenance_workers:^33} {maintenance_work_mem:^21}")
        print("="*72)
        cursor.execute(f"SET maintenance_work_mem='{maintenance_work_mem}';", )
        cursor.execute(
            f"SET max_parallel_maintenance_workers={max_parallel_maintenance_workers};")
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")

        # Resolve index name and parameters
        index_name = f"{table_name}_{index_type}_{metric}_m{m}_ef{ef_construction}"
        print(f"Creating index {index_name}")
        parameters = ''
        start = time()
        if index_type == 'hnsw':
            parameters = f"(m = {m}, ef_construction = {ef_construction});"
        elif index_type == 'ivfflat':
            parameters = f"(lists = {num_lists});"
        else:
            print(f"Invalid index type: {index_type}")
            return

        # Create index
        query = f"CREATE INDEX {index_name} ON {table_name} USING {index_type} (embedding {metric}) WITH {parameters}"
        cursor.execute(query)
        conn.commit()

        # Cleanup
        end = time()
        print(
            f"Created {index_type} index on {table_name} with {metric} metric in {end - start:.2f} seconds")
        cursor.close()
        conn.close()

    def _get_all_chunks(self, cursor, columns: list[str] = ['id', 'text']) -> list[dict]:
        cursor.execute(f"SELECT id, text FROM chunks;")
        return cursor.fetchall()

    def get_vectors_by_doi(self, doi: str, vector_table: str) -> list[str]:
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                text,
                {vector_table}.embedding AS embedding
            FROM chunks
            JOIN {vector_table} ON chunks.id = {vector_table}.chunk_id
            WHERE doi = '{doi}';
            """)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [ChunkAndVector(text, np.array(lst)) for text, lst in results]

    def batch_query_vector_table(self,
                                 table_name,
                                 query_vectors,
                                 metric,
                                 top_k=5,
                                 probes=40,
                                 work_mem='8GB',
                                 max_parallel_workers=8,
                                 max_parallel_workers_per_gather=8):
        """
        TODO: this isn't working, figure out why
        table_name: name of the vector table
        query_vectors: list of vectors to query
        metric: a key in PGVECTOR_DISTANCE_METRICS to resolve the distance operator
        top_k: number of results to return for each query vector

        Returns a list of lists, where each inner list contains the results for one query vector
        """

        """
        table_name: name of the vector table
        query_vectors: list of vectors to query
        metric: a key in PGVECTOR_DISTANCE_METRICS to resolve the distance operator
        top_k: number of results to return for each query vector
        threshold: the maximum distance to consider a vector as a neighbor

        Returns a list of lists, where each inner list contains the results for one query vector
        """

        # Resolve the distance operator
        assert metric in PGVECTOR_DISTANCE_METRICS, f"Invalid metric: {metric}. I don't have that metric in the PGVECTOR_DISTANCE_METRICS dictionary"
        operator = PGVECTOR_DISTANCE_METRICS[metric]

        conn = psycopg2.connect(**self.db_params)
        register_vector(conn)
        cursor = conn.cursor()
        dim = query_vectors.shape[1]
        query_vectors_list = [
            f"[{','.join(map(str, vec.astype(float)))}]" for vec in query_vectors
        ]
        # query_vectors_string =

        # Create a temporary table to hold the query vectors
        cursor.execute(
            f"CREATE TEMP TABLE temp_query_vectors (embedding VECTOR({dim}));")
        execute_values(cursor,
                       "INSERT INTO temp_query_vectors (embedding) VALUES %s;",
                       query_vectors_list,
                       template="(%s)")

        query = f"""
            SELECT
                t.chunk_id,
                c.doi,
                c.text,
                t.embedding {operator} qv.embedding AS distance
            FROM {table_name} AS t
            JOIN chunks AS c ON t.chunk_id = c.id
            JOIN temp_query_vectors AS qv
            ON TRUE  -- Creates a CROSS JOIN
            ORDER BY distance ASC
            LIMIT {top_k};
            """
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [[SingleQueryResult(*res) for res in result] for result in results]
        # Set the session resources
        # cursor.execute(f"SET work_mem='{work_mem}';")
        # cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")
        # cursor.execute(
        #     f"SET max_parallel_workers_per_gather={max_parallel_workers_per_gather};")
        # cursor.execute(f"SET ivfflat.probes={probes};")
        # cursor.execute(f"SET enable_seqscan = off;")

        # # Convert numpy array to a list of vector strings
        # query_vectors_list = [
        #     f"'[{','.join(map(str, vec))}]'::vector" for vec in query_vectors]
        # query_vectors_str = "ARRAY[" + ",".join(query_vectors_list) + "]"

        # query = f"""
        # WITH query_vectors AS (
        #     SELECT unnest({query_vectors_str}::vector[]) AS query_vector
        # )
        # SELECT
        #     {table_name}.chunk_id AS chunk_id,
        #     chunks.doi AS doi,
        #     chunks.text AS text,
        #     {table_name}.embedding {operator} query_vector AS distance
        # FROM
        #     {table_name}
        # JOIN
        #     chunks ON {table_name}.chunk_id = chunks.id
        # CROSS JOIN
        #     query_vectors
        # ORDER BY
        #     distance ASC
        # LIMIT {top_k};
        # """

        # cursor.execute(query)
        # results = cursor.fetchall()
        # cursor.close()
        # conn.close()
        # return_results = []
        # for row in results:
        #     return_results.append(SingleQueryResult(*row))
        # return return_results

        # return [[SingleQueryResult(*res) for res in result] for result in results]

    def query_vector_table(self,
                           table_name,
                           query_vector,
                           metric,
                           top_k=5,
                           probes=40,
                           use_index=True):
        """
        table_name: name of the vector table
        query_vector: the vector to query
        metric: a key in PGVECTOR_DISTANCE_METRICS to resolve the distance operator
        top_k: number of results to return

        """

        # Resolve the distance operator
        assert metric in PGVECTOR_DISTANCE_METRICS, f"Invalid metric: {metric}. I don't have that metric in the PGVECTOR_DISTANCE_METRICS dictionary"
        operator = PGVECTOR_DISTANCE_METRICS[metric]

        conn = psycopg2.connect(**self.db_params)
        register_vector(conn)
        cursor = conn.cursor()

        # Set the session resources
        cores = os.cpu_count()
        max_parallel_workers = max(1, cores - 2)
        max_parallel_workers_per_gather = max_parallel_workers - 1
        work_mem = '1GB'
        cursor.execute(f"SET max_parallel_workers={max_parallel_workers};")
        cursor.execute(
            f"SET max_parallel_workers_per_gather={max_parallel_workers_per_gather};")
        cursor.execute(f"SET work_mem='{work_mem}'")

        # Confirm settings
        '''
        cursor.execute("SHOW max_worker_processes;")
        max_worker_processes = int(cursor.fetchone()[0])
        cursor.execute("SHOW max_parallel_workers;")
        max_parallel_workers = int(cursor.fetchone()[0])
        cursor.execute("SHOW max_parallel_workers_per_gather;")
        max_parallel_workers_per_gather = int(
            cursor.fetchone()[0])

        print("="*40 + "CONFIG" + "="*40)
        print("max_worker_processes | max_parallel_workers | max_parallel_workers_per_gather | work_mem")
        print(
            f"{max_worker_processes:^20} | {max_parallel_workers:^20} | {max_parallel_workers_per_gather:^31} | {work_mem:^10}")
        print("="*88)
        '''

        # Set index search parameters
        if not use_index:
            cursor.execute(f"SET enable_indexscan = off;")
        else:
            cursor.execute(f"SET enable_indexscan = on;")
            cursor.execute("SET hnsw.ef_search = 40;")
            # cursor.execute(f"SET ivfflat.probes={probes};")
        #     cursor.execute("SET enable_seqscan = off;")

        cursor.execute(
            f"""
            -- EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON)
            SELECT {table_name}.chunk_id, chunks.doi, chunks.text, {table_name}.embedding {operator} %s AS distance
            FROM {table_name}
            JOIN chunks ON {table_name}.chunk_id = chunks.id
            ORDER BY {table_name}.embedding {operator} %s ASC
            LIMIT %s;
            """,
            (query_vector, query_vector, top_k)
        )

        # Close up
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [SingleQueryResult(*result) for result in results]

    def test_connection(self):
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()

        # Print a pretty table showing db name, user, host, and port from db_params
        print("="*33 + "CONFIG" + "="*33)
        print(f"{'Database':<16} {'User':<16} {'Host':<32} {'Port':<16}")
        print(
            f"{self.db_params['dbname']:<16} {self.db_params['user']:<16} {self.db_params['host']:<32} {self.db_params['port']:<16}")
        print("="*72)

        # Execute a simple query
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"Database version: {db_version}")

        cursor.close()
        conn.close()


def profile_create_vector_table(db, table_name, embedder):
    with cProfile.Profile() as pr:
        db.create_vector_table(table_name=table_name,
                               dim=768, embedder=embedder)
    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats("cumulative").print_stats(40)


def main():
    # NOTE: assumes .env is in the parent directory
    load_dotenv('../.env', override=True)

    # Database setup
    db_params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }
    db = DatabaseProcessor(db_params)
    db.test_connection()

    args = argument_parser()

    if args.add_chunks:

        db.chunk_and_insert_records(
            path=args.path, max_length=args.max_length, overlap=args.overlap)
        return

    if args.create_vector_table:

        # Extract parameters
        table_name, embedder, normalize, batch_size = args.table_name, args.embedder, args.normalize, args.batch_size

        # Create embedding function and get its dimension
        from Embedders import get_embedder
        print(
            f"Creating vector table '{table_name}' with embedder {embedder} on device {db.device}...")
        embedder = get_embedder(embedder, db.device, normalize)
        dim = embedder(['test']).shape[1]

        db.create_vector_table(
            table_name=table_name, dim=dim, embedder=embedder, batch_size=batch_size)

        # TODO: add calls to create indexes
        # db.create_index(table_name, 'ivfflat', 'vector_cosine_ops', 1580)
        return

    if args.create_index:
        if args.index_type == 'hnsw':
            # m, ef_construction = args.m, args.ef_construction
            values = ['table_name', 'index_type',
                      'metric', 'm', 'ef_construction']
            kwargs = {k: v for k, v in vars(args).items() if k in values}
            print(
                f"Creating index on {args.table_name} with type {args.index_type} and metric {args.metric}")
            db.create_index(**kwargs)
        elif args.index_type == 'ivfflat':
            values = ['table_name', 'index_type', 'metric', 'num_lists']
            kwargs = {k: v for k, v in vars(args).items() if k in values}
            print(
                f"Creating index on {args.table_name} with type {args.index_type} and metric {args.metric}")
            db.create_index(**kwargs)
        else:
            print(f"Invalid index type: {index_type}")
        return

    if args.test_connection:
        # Connection test invoked before switching on command line args
        return


if __name__ == '__main__':
    main()
