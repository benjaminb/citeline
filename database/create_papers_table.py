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

    columns = ["doi", "title", "abstract", "citation_count"]

    # Clear null characters from TEXT columns we'll be using
    for column in ["doi", "title", "abstract"]: 
        papers[column] = papers[column].apply(clear_null_chars)
    print("Cleared null characters from columns:", columns)

    db = Database()
    try:
        with db.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id SERIAL PRIMARY KEY,
                    doi TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    citation_count INTEGER NOT NULL
                )
                """
            )
            print("Created 'papers' table (if it didn't already exist)")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers (doi)
                """
            )
            print("Created index on 'doi' column in papers table")

            # Get control set of DOIs in the db
            cur.execute("SELECT DISTINCT doi FROM contributions")
            existing_dois = {row[0] for row in cur.fetchall()}
            print(f"Found {len(existing_dois)} existing DOIs in contributions table")

            # Prepare data
            data = [
                (row["doi"], row["title"], row.get("abstract", ""), row["citation_count"])
                for _, row in papers.iterrows()
                if row["doi"] in existing_dois
            ]
            print(f"Filtered data to {len(data)} papers with DOIs in contributions table")

            # Batch insert
            cur.executemany(
                """
                INSERT INTO papers (doi, title, abstract, citation_count)
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
