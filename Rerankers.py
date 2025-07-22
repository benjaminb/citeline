import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from database.database import VectorSearchResult

# Set up logging
logging.basicConfig(
    filename="logs/deepseek.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv()
"""
Rerankers implement an interface (list[VectorSearchResult] -> list[float]), taking the ininitial results from 
a database vector search and returning a list of scores for each result.

Closures contain rerankers that take a db reference and binds models, prompts, and other constants.
"""


def get_roberta_nli_ranker(db: None) -> callable:
    from sentence_transformers import CrossEncoder

    MODEL_NAME = "sentence-transformers/nli-roberta-base"
    device = device if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    model = CrossEncoder(MODEL_NAME, device=device)

    def entailment_ranker(query: str, results: list[VectorSearchResult]) -> list[float]:
        """
        Given a query and a list of VectorSearchResults, returns a list of entailment scores
        for each result based on the query.
        """
        model_inputs = [(query, result.text) for result in results]
        scores = model.predict(model_inputs)
        return scores.tolist()

    return entailment_ranker


def get_deberta_nli_ranker(db: None, use_contradiction=False) -> callable:

    device = device if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    def entailment_ranker(query: str, results: pd.DataFrame) -> pd.DataFrame:
        results_copy = results.copy()
        premises = results_copy["text"].tolist()
        scores = np.zeros(len(premises), dtype=np.float32)

        batch_size = 16
        for i in tqdm(range(0, len(results_copy), batch_size), desc="Processing NLI scores", leave=False):
            batch_premises = premises[i : i + batch_size]
            batch_queries = [query] * len(batch_premises)
            inputs = tokenizer(
                batch_premises,
                batch_queries,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
            batch_outputs = torch.softmax(output.logits, dim=-1)

            entailment_scores, neutral_scores, contradiction_scores = torch.unbind(batch_outputs, dim=-1)
            # TODO use just entailment?
            if use_contradiction:
                batch_scores = torch.maximum(entailment_scores, contradiction_scores)
            else:
                batch_scores = entailment_scores
            scores[i : i + len(batch_scores)] = batch_scores.cpu().numpy()

        results_copy["deberta_nli_score"] = scores
        return results_copy.sort_values("deberta_nli_score", ascending=False).reset_index(drop=True)

    return entailment_ranker


def get_deepseek_boolean(db=None):
    assert "DEEPSEEK_API_KEY" in os.environ, "Please set the DEEPSEEK_API_KEY environment variable"
    assert db is not None, "Database instance must be provided to get_deepseek_boolean"
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    PROMPT_FILE = "llm/prompts/deepseek_citation_identification.txt"
    MAX_PAPER_LEN = 250_000  # ~65k tokens, leaving ~500 tokens for response
    with open(PROMPT_FILE, "r") as file:
        prompt = file.read()

    def deepseek_boolean(query: str, results: list[VectorSearchResult]) -> float:
        """
        Given a list of candidates, returns a list of boolean floats (0.0 or 1.0)
        indicating whether each candidate should be cited in the query or not
        """
        checked_dois = dict()  # To manage duplicate DOIs in results list
        scores = []
        for result in results:
            # Reconstruct doi's paper and configure prompt template
            doi = result.doi
            if doi in checked_dois:
                scores.append(checked_dois[doi])
                continue
            paper = ""
            try:
                full_paper = db.get_reconstructed_paper(doi)
                if len(full_paper) > MAX_PAPER_LEN:
                    logging.warning(f"Paper {doi} is too long, truncating to {MAX_PAPER_LEN} characters")
                    full_paper = full_paper[:MAX_PAPER_LEN]
                paper = full_paper
            except ValueError as e:
                logging.error(f"Error reconstructing paper for DOI {doi}: {e}")

            prompt_formatted = prompt.format(text=query, paper=paper)

            # Try getting DeepSeek API response
            response = None
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": prompt_formatted},
                    ],
                    stream=False,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logging.error(f"Error calling DeepSeek API for DOI {doi}: {e}")

            # Parse the response
            try:
                json_content = json.loads(response.choices[0].message.content)

            except json.JSONDecodeError as e:
                logging.error(
                    f"Error parsing JSON response for DOI {doi}: {e}. Response content: {response.choices[0].message.content}"
                )

            try:
                should_cite = json_content["should_cite"]
            except KeyError as e:
                logging.error(
                    f"Error extracting 'should_cite' from JSON response for DOI {doi}: {e}. Response content: {response.choices[0].message.content}"
                )
            logging.info(f"Raw response: {response}")

            # To remain consistent with other rerankers we return a float
            score = float(should_cite)
            checked_dois[doi] = score
            scores.append(score)
        return scores

    return deepseek_boolean


RERANKERS = {
    "deepseek_boolean": get_deepseek_boolean,
    "roberta_nli": get_roberta_nli_ranker,
    "deberta_nli": get_deberta_nli_ranker,
}


def get_reranker(reranker_name: str, db=None) -> callable:
    if reranker_name in RERANKERS:
        return RERANKERS[reranker_name](db=db)
    raise ValueError(f"Unknown reranker: {reranker_name}")


def main():
    print("imported")
    nli_ranker = get_reranker("deberta_nli")
    premises = [
        "The sky is blue",
        "George Washington was the first president of the United States",
        "As further representation of the study's importance we have included a figure showing the results",
    ]

    premises_df = pd.DataFrame(
        {
            "text": premises,
            "doi": [f"doi_{i}" for i in range(len(premises))],
            "pubdate": ["na"] * len(premises),
            "distance": [0.0] * len(premises),
        }
    )

    print("Confidence on top 3 premises by vector similarity:")
    query = "The sky is not red"
    conf = nli_ranker(query, premises_df)
    print(f"Confidence scores: {conf}")
    print(f"deberta column: {conf['deberta_nli_score']}")


if __name__ == "__main__":
    main()
