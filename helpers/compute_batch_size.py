import os
import psycopg2
import cProfile
import pstats
from dotenv import load_dotenv
from time import time
from embedders import get_embedder, EMBEDDING_CLASS
from database.database import Database


def stress_test(model_name: str, db: Database):
    print(f"Embedding model: {model_name}")
    embedder = get_embedder(model_name, "cuda", normalize=False)
    batch_size = 1
    averages = []
    while batch_size < 4096:
        try:
            # Get chunks from the database
            conn = psycopg2.connect(**db.db_params)
            cursor = conn.cursor()
            cursor.execute(f"SELECT text FROM chunks LIMIT {batch_size}")
            rows = cursor.fetchall()
            conn.close()
            chunks = [row[0] for row in rows]
            print(f"Got {len(chunks)} chunks")

            # Embed the chunks
            start = time()
            result = embedder(chunks)
            duration = time() - start
            print(f"Result shape: {result.shape}")
            averages.append(duration / batch_size)
            print(f"Batch size {batch_size} took {duration} seconds ({duration/batch_size} per chunk)")
            batch_size *= 2
        except Exception as e:
            print(e)
            break
    # Print the results
    for i, times in enumerate(averages):
        print(f"Model: {model_name}")
        print(f"Average times per chunk: {times}")
        smallest = min(times)
        best_index = times.index(smallest)
        print(f"Smallest time per chunk: {smallest}")
        print(f"Batch size: {2**best_index}")
        print("=====================================")


def profile(model_name: str, db: Database):
    with cProfile.Profile() as pr:
        stress_test(model_name=model_name, db=db)
    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats("cumulative").print_stats(40)


def main():
    # Load database
    load_dotenv(".env", override=True)
    db_params = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }
    start = time()
    db = Database(db_params)
    print(f"Database loaded in {time() - start} seconds")
    print(db.db_params)

    # Time each embedding model
    for model_name in ["adsabs/astroBERT"]:
        profile(model_name, db)


if __name__ == "__main__":
    main()
