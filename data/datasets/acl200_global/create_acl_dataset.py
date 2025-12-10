import os
import requests
from dotenv import load_dotenv

load_dotenv("../../../.env")

assert os.getenv("SEMANTIC_SCHOLAR_API_KEY"), "SEMANTIC_SCHOLAR_API_KEY not found in .env"
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
BASE_URL = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def title_search(self, title: str) -> dict:
        params = {"query": title, "fields": "title,openAccessPdf,url", "limit": 1}
        headers = {"x-api-key": self.api_key}
        try:
            res = requests.get(f"{BASE_URL}/paper/search", params=params, headers=headers)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return {}

        return res.json()

    def get_paper_id(self, data: dict) -> str | None:
        try:
            paper_id = data["data"][0]["paperId"]
            return paper_id
        except (KeyError, IndexError):
            print("Paper ID not found in the response data.")
            return None


def main():
    sem_scholar_client = SemanticScholarClient(API_KEY)
    title = "Attention Is All You Need"
    search_results = sem_scholar_client.title_search(title)
    paper_id = sem_scholar_client.get_paper_id(search_results)
    if paper_id:
        print(f"Paper ID for '{title}': {paper_id}")


if __name__ == "__main__":
    main()
