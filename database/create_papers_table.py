"""
Creates the `papers` table. Run from database/ or update the PAPERS_PATH accordingly.
"""

import pandas as pd
from database import Database

PAPERS_PATH = "../data/preprocessed/research.jsonl"


def clear_null_chars(text: str) -> str:
    """
    Removes null characters from the text.
    """
    return text.replace("\x00", "")


def create_papers_table():
    papers = pd.read_json(PAPERS_PATH, lines=True)
    print(f"Loaded {len(papers)} papers from research.jsonl")

    # Clear null characters from columns we'll be using
    columns = ["doi", "title", "abstract", "body"]
    for column in columns:
        papers[column] = papers[column].apply(clear_null_chars)
    print("Cleared null characters from columns:", columns)

    db = Database()
    try:
        with db.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id SERIAL PRIMARY KEY,
                    doi TEXT UNIQUE,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    body TEXT NOT NULL
                )
                """
            )

            # Prepare data
            data = [
                (row["doi"], row["title"], row.get("abstract", ""), row["body"])
                for _, row in papers.iterrows()
            ]

            # Batch insert
            cur.executemany(
                """
                INSERT INTO papers (doi, title, abstract, body)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doi) DO NOTHING
                """,
                data,
            )

        db.conn.commit()
        print(f"Successfully inserted {len(papers)} papers into the papers table.")

    except Exception as e:
        print(f"Error: {e}")
        db.conn.rollback()
        raise


def main():
    create_papers_table()


if __name__ == "__main__":
    main()
