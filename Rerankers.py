import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
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


def get_deepseek_boolean(db=None):
    assert "DEEPSEEK_API_KEY" in os.environ, "Please set the DEEPSEEK_API_KEY environment variable"
    assert db is not None, "Database instance must be provided to get_deepseek_boolean"
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    PROMPT_FILE = "llm/prompts/deepseek_citation_identification.txt"
    MAX_PAPER_LEN = 195_000  # ~65k tokens, leaving ~500 tokens for response
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
}


def get_reranker(reranker_name: str, db=None) -> callable:
    if reranker_name in RERANKERS:
        return RERANKERS[reranker_name](db=db)
    raise ValueError(f"Unknown reranker: {reranker_name}")


def main():
    print("imported")
    nli_ranker = get_roberta_nli_ranker(db=None)
    premises = [
        "3, 1969 Thermal continuum radiation 377 carried out. The variation of the ratio g(Z, T, c/X)lg(i, T, c/X) was checked at a number of wavelengths and temperatures in the ranges o*8 to ioo*o io6 °K and i to 30 Â for the twelve elements listed in Table I. The variations found in the ratio of the Gaunt factors would lead to fluctuations of up to 20 per cent in the value of the bracketed term in equation (6). These fluctuations are similar to those noted by Hovenier who found variations of up to 15 per cent in this term. Table I shows values of the bracketed term in equation (5) at a temperature of 5*0 io6 °K and a wavelength of 5 Â. Two sets of element abundances were employed. For the corona the abundance values used are those of Pottasch (1967) and Jordan (1968). In cases where these results have disagreed, an average value Table I Dependence of free-free flux on element abundances Element H He C N O Ne Mg Si S Ar Ca Fe Coronal abundances (Pottasch 1967; Jordan 1967) 10 2 *oio5 400 60 300 40 30 50 20 4* 2* So Photospheric abundances (Goldberg et al. i960) ¿(Z, T, c/A)Z2 Nz |(i,T,C/A) Nh Coronal Photospheric i • 124 0*017 0*005 o*oi8 0*008 0*007 0*012 0*006 0*002 0*001 0*051 o*8io 0*030 0*009 0*060 0*008 0*007 0*010 0*006 0*002 0*001 0*005 i -25° 0*950 6 io i*45 io5 600 100 1000 40 30 40 20 4* 2* 5 Total Note Cosmic abundance of neon is 300 on the above scale (Allen (1963)).",
        "Furthermore, some of the variations of the observed spectrum are not simply due directly to the higher abundance of individual elements near the magnetic pole, but to the changes in atmospheric structure induced by the combination of all such abundance variations acting in concert. Two other features of the inferred abundance distributions are notable. First, the highest abundance regions found for the already cosmically abundant elements Si and Fe are rather large; in our model the two rings out to a = 72° are both high-abundance regions for these elements relative to the equatorial abundances. In contrast, the polar caps of the cosmically lower abundance elements Ti and Cr are smaller; for both these elements the midlatitude (ring 2) abundances are considerably smaller than at the pole. A second, possibly significant feature of the abundance models is that all three iron-peak elements have quantitatively similar absolute abundances both near the pole and near the equator, although in the Sun Fe is roughly 2 dex more abundant than Ti or Cr. Are we seeing some sort of saturation effect? It is interesting to use the observed value of v sin i to constrain the physical characteristics of HD 215441. Since our magnetic model indicates that i = 30° ±5°, we may use the elementary relationship R/Rq = Pv sin i/(50.6 sin i), (4) _1 where P is in days and t; sin i = 7 ± 3 is in km s , to find R/Rq = 2.6 ± 1.2.",
        "As further representative parameters we have chosen: atomic masses m = 10, 20, 30; abundances of the elements A =5 x i0~ ~H, maximum fractional abundance of the ion X = 0.9; background continuum intensity = black body radiation of i04 K; for off-limb observations: coronal Te = 106.5 K, coronal Ne 108.3 cm3. Two representative thicknesses (d1, d2, Table I) of the absorbing layers are chosen: a region (=d1) centered on Te, Table I, and confined between z~ log Te~ 0.2; a region (=d2) centered on Te and confined between z~ log Te 0.4. These two regions roughly correspond to a change in the fractional abundance of the ions (of C, N, 0, Ne, Mg, Si, Fe) of ~1 log X~ 0.1-0.2 and ~1 log XI 0.2-0.3, respectively. Inside each layer the electron temperature and the electron density, given in Table I, are constant.",
    ]
    premises_packaged = [
        VectorSearchResult(text=premise, doi=f"doi_{i}", pubdate="na", distance=0.0)
        for i, premise in enumerate(premises)
    ]
    print("Confidence on top 3 premises by vector similarity:")
    query = "Variations in the ratio of Gaunt factors are known to fluctuate"
    conf = nli_ranker(query, premises_packaged)
    print(f"Confidence scores: {conf}")


if __name__ == "__main__":
    main()
