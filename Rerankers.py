from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
        filename="logs/deepseek.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv()

"""
From documentation: 
For a given sentence pair, it will output three scores corresponding to the labels: contradiction, entailment, neutral.
I.e. [a, b, c] -> [contradiction score, entailment score, neutral score]
TODO: if we take Alberto's idea to first categorize the query as citation type, we can make use of contradiction and entailment
scores separately (e.g. if the query is pushing back on a previous result, we use contradiction score)
"""

true_premise = "Furthermore, the abundance variations are correlated with the type of magnetic field topologv observed. Comparison of Mg vm 315.02 Â with Si vm 319.84 A indicates that the magnesium abundance is relatively unchanged, so that the variations primarily reflect changes in the abundance of neon. Meyer (1985) suggested that the apparent depletion of ele- 1049 Ne VI 40 I. I 4 Mg VI 400.66 Fig. 4.—(a) The Ne vu 465/Ca ix 466 intensity ratio plotted against the Ne vi 401/Mg vi 400 intensity ratio from the observations in Table 1 and an impulsive flare showing the variation in the neon abundance relative to calcium or magnesium, (b) The Ne vii/Ca ix ratio divided by the Ne vi/Mg vi ratio in 4(a) showing the smaller variation in the Mg/Ca abundance ratio. ments with high-first ionization potentials (FIP) in the corona compared to low-FIP elements could be explained by assuming a separation process at the base of the corona which held back the predominantly neutral high-FIP elements. We note that in the presence of temperatures characteristic of the lower transition zone (<105 K) there will be more ionizing collisions at 7.6 eV then at 21.6 eV: hence more Mg+ ions than Ne+ ions at the base of the corona. In the presence of an open field these ions can drift out into the corona to be successively ionized to higher stages. The element abundance distributions observed in the open-field case may therefore directly reflect the relative production rates of first ions at the base."
hypothesis = "The work of  suggests a fundamental distinction in elemental abundances between closed and open magnetic structures, matching the nominal photospheric and coronal abundances, respectively."


def entailment_ranker(model_name: str, device: str = "mps") -> callable:
    """
    Given an NLI model name that takes two strings as input and returns the entailment score at
    the output's index 1, returns the entailment function
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    def entailment_fn(premise: str, hypothesis: str) -> float:
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        return probs[1]  # Entailment score

    return entailment_fn

def get_deepseek_boolean():
    assert 'DEEPSEEK_API_KEY' in os.environ, "Please set the DEEPSEEK_API_KEY environment variable"
    client = OpenAI(api_key=os.getenv['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
    PROMPT_FILE = "llm/prompts/deepseek_citation_identification.txt"
    with open(PROMPT_FILE, 'r') as file:
        prompt = file.read()

    def deepseek_boolean(query, candidate) -> list[bool]:
        """
        Given a query and a list of candidates, returns a list of booleans indicating whether each candidate
        should be cited in the query or not
        """
        prompt_formatted = prompt.format(text=query, paper=candidate)
        # Write a deepseek client that returns only true or false
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt_formatted},
                ],
                stream=False,
                response_format={'type': 'json_object'}
            )
            return response
        except Exception as e:
            logging.error(f"Error in DeepSeek API call: {e}")
            return []
    return deepseek_boolean


def main():
    print("imported")
    # roberta_entailment_ranker = entailment_ranker("cross-encoder/nli-roberta-base")
    # conf = roberta_entailment_ranker(true_premise, hypothesis)
    # print(f"Confidence with the target premise: {conf}")

    # premises = [
    #     "3, 1969 Thermal continuum radiation 377 carried out. The variation of the ratio g(Z, T, c/X)lg(i, T, c/X) was checked at a number of wavelengths and temperatures in the ranges o*8 to ioo*o io6 °K and i to 30 Â for the twelve elements listed in Table I. The variations found in the ratio of the Gaunt factors would lead to fluctuations of up to 20 per cent in the value of the bracketed term in equation (6). These fluctuations are similar to those noted by Hovenier who found variations of up to 15 per cent in this term. Table I shows values of the bracketed term in equation (5) at a temperature of 5*0 io6 °K and a wavelength of 5 Â. Two sets of element abundances were employed. For the corona the abundance values used are those of Pottasch (1967) and Jordan (1968). In cases where these results have disagreed, an average value Table I Dependence of free-free flux on element abundances Element H He C N O Ne Mg Si S Ar Ca Fe Coronal abundances (Pottasch 1967; Jordan 1967) 10 2 *oio5 400 60 300 40 30 50 20 4* 2* So Photospheric abundances (Goldberg et al. i960) ¿(Z, T, c/A)Z2 Nz |(i,T,C/A) Nh Coronal Photospheric i • 124 0*017 0*005 o*oi8 0*008 0*007 0*012 0*006 0*002 0*001 0*051 o*8io 0*030 0*009 0*060 0*008 0*007 0*010 0*006 0*002 0*001 0*005 i -25° 0*950 6 io i*45 io5 600 100 1000 40 30 40 20 4* 2* 5 Total Note Cosmic abundance of neon is 300 on the above scale (Allen (1963)).",
    #     "Furthermore, some of the variations of the observed spectrum are not simply due directly to the higher abundance of individual elements near the magnetic pole, but to the changes in atmospheric structure induced by the combination of all such abundance variations acting in concert. Two other features of the inferred abundance distributions are notable. First, the highest abundance regions found for the already cosmically abundant elements Si and Fe are rather large; in our model the two rings out to a = 72° are both high-abundance regions for these elements relative to the equatorial abundances. In contrast, the polar caps of the cosmically lower abundance elements Ti and Cr are smaller; for both these elements the midlatitude (ring 2) abundances are considerably smaller than at the pole. A second, possibly significant feature of the abundance models is that all three iron-peak elements have quantitatively similar absolute abundances both near the pole and near the equator, although in the Sun Fe is roughly 2 dex more abundant than Ti or Cr. Are we seeing some sort of saturation effect? It is interesting to use the observed value of v sin i to constrain the physical characteristics of HD 215441. Since our magnetic model indicates that i = 30° ±5°, we may use the elementary relationship R/Rq = Pv sin i/(50.6 sin i), (4) _1 where P is in days and t; sin i = 7 ± 3 is in km s , to find R/Rq = 2.6 ± 1.2.",
    #     "As further representative parameters we have chosen: atomic masses m = 10, 20, 30; abundances of the elements A =5 x i0~ ~H, maximum fractional abundance of the ion X = 0.9; background continuum intensity = black body radiation of i04 K; for off-limb observations: coronal Te = 106.5 K, coronal Ne 108.3 cm3. Two representative thicknesses (d1, d2, Table I) of the absorbing layers are chosen: a region (=d1) centered on Te, Table I, and confined between z~ log Te~ 0.2; a region (=d2) centered on Te and confined between z~ log Te 0.4. These two regions roughly correspond to a change in the fractional abundance of the ions (of C, N, 0, Ne, Mg, Si, Fe) of ~1 log X~ 0.1-0.2 and ~1 log XI 0.2-0.3, respectively. Inside each layer the electron temperature and the electron density, given in Table I, are constant.",
    # ]
    # print("Confidence on top 3 premises by vector similarity:")
    # for premise in premises:
    #     conf = roberta_entailment_ranker(premise, hypothesis)
    #     print(f"Confidence on premise: {conf}")


if __name__ == "__main__":
    main()
