import os
import psycopg2
from dotenv import load_dotenv
from time import time
from Embedders import get_embedder, EMBEDDING_CLASS
from database.database import DatabaseProcessor


def main():
    # Load database
    load_dotenv('.env', override=True)
    db_params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
    }
    db = DatabaseProcessor(db_params)
    print(db.db_params)

    averages = {}
    # Time each embedding model
    for model_name in EMBEDDING_CLASS:
        print(f"Embedding model: {model_name}")
        embedder = get_embedder(model_name, 'cuda', normalize=False)
        averages[model_name] = []

        batch_size = 1
        while batch_size < 4096:
            try:
                # Get chunks from the database
                conn = psycopg2.connect(**db.db_params)
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT text FROM chunks LIMIT {batch_size}")
                rows = cursor.fetchall()
                conn.close()
                chunks = [row[0] for row in rows]
                print(f"Got {len(chunks)} chunks")

                # Embed the chunks
                start = time()
                result = embedder(chunks)
                duration = time() - start
                print(f"Result shape: {result.shape}")
                averages[model_name].append(duration/batch_size)
                print(
                    f"Batch size {batch_size} took {duration} seconds ({duration/batch_size} per chunk)")
                batch_size *= 2
            except Exception as e:
                print(e)
                break

        # Print the results
        times = averages[model_name]
        print(f"Model: {model_name}")
        print(f"Average times per chunk: {times}")
        smallest = min(times)
        best_index = times.index(smallest)
        print(f"Smallest time per chunk: {smallest}")
        print(f"Batch size: {2**best_index}")
        print("=====================================")


if __name__ == '__main__':
    main()
