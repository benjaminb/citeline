import os
from database import DatabaseProcessor
from dotenv import load_dotenv

PATHS_TO_DATA = [
    '../data/json/Astro_Research.json',
    '../data/json/doi_articles.json',
    '../data/json/Earth_Science_Research.json',
    '../data/json/Planetary_Research.json',
    '../data/json/salvaged_articles.json'
]


def main():
    load_dotenv('../.env')
    db_params = {
        'dbname': os.getenv('DB_NAME') ,
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }
    db = DatabaseProcessor(db_params)
    db.test_connection()
    print(db.db_params)
    print("=====================================")

    # Load the data
    for path in PATHS_TO_DATA:
        print(f"Loading data from {path}")
        db.chunk_and_insert_records(path)
        print("=====================================")


if __name__ == "__main__":
    main()
