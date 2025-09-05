from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Union
from pydantic import BaseModel, RootModel
from models import LLMResponse

MODEL_NAME = "mistral-nemo:latest"  # Replace with your model name


# class LLMFunction:
#     def __init__(self, model_name, system_prompt_path, output_model):
#         """
#         Initialize the LLMFunction with the model name, system prompt path, and output model.
#         """
#         self.llm_function = get_llm_function(
#             model_name=model_name,
#             system_prompt_path=system_prompt_path,
#             output_model=output_model,
#         )

#     def __call__(self, text: str):
#         """
#         Call the LLM function with the provided text.
#         :param text: The input text to process.
#         :return: The output from the LLM function.
#         """
#         return self.llm_function(text)


# def get_llm_function(
#     model_name: str = MODEL_NAME,
#     system_prompt_path: str = None,
#     output_model: Union[BaseModel, RootModel] = None,
# ):
#     """
#     Returns a function that invokes the LLM with the given model name and system prompt.
#     """
#     llm = ChatOllama(
#         model=model_name,
#         temperature=0.0,
#     ).with_structured_output(output_model, method="json_schema")

#     # Read in the system prompt
#     with open(system_prompt_path, "r") as f:
#         sys_msg = f.read()
#     sys_msg = SystemMessage(content=sys_msg)

#     def llm_function(text: str):
#         try:
#             msg = HumanMessage(content=text)
#             return llm.invoke([sys_msg, msg])
#         except Exception as e:
#             print(f"Exception: {e}")
#             return None

#     return llm_function


class ChatResponse:
    """
    NOTE: you must provide a dict to the call method based on what variables are required by the prompt template
    and you extract fields as needed based on the output_model
    """

    def __init__(self, model_name: str, prompt_path: str, output_model: Union[BaseModel, RootModel]):
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.output_model = output_model
        with open(prompt_path, "r") as f:
            self.prompt = f.read()
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0,
        ).with_structured_output(output_model, method="json_schema")

    def __call__(self, input_dict: dict) -> Union[BaseModel, RootModel]:
        formatted_prompt = self.prompt.format(**input_dict)
        response = self.llm.invoke(formatted_prompt)
        return response


def main():
    llm_result = ChatResponse(
        model_name="mistral-nemo", prompt_path="prompts/chunk_summarization_prompt.txt", output_model=LLMResponse
    )

    chunk = """The following two expressions describe the mass loss rates or mass fluxes to an accuracy of A log = 0.17 
    and A log 911=0.17 (root mean square difference): logFm = -5.23+4.601og(Teff/3X 104)—0.481og(geff/103) log 911= 
    -4.83+1.421og( L/106 )+0.61 log( R/30) - 0.99 log( M/30), where 9His in 9H0 yr ~1 ; L, R, M in solar units, and in 
    g s_ 1 cm“2. The correlation coefficient of these fits is 0.95. The relations do not agree with the present predictions 
    for the radiation driven wind theory, nor with those for the fluctuation theory of mass loss. The predictions for the radiation 
    driven wind models might agree with the observations if the values of the parameters which describe the radiation pressure 
    are shghtly modified. A star of 100 91t0 will lose about 15% of its mass during its evolution from the zero age main sequence 
    to the first core contraction phase, and a star of 40 91t0 will lose about 10% of its mass. Subject headings: stars: 
    early-type — stars: mass loss — stars: winds I. INTRODUCTION observations of mass loss rates from seven stars by Abbott et al (1980), 
    however, have led these authors to suggest that 9HocL18. On the other hand, Lamers, Paerels, and de Loore (1980) and Conti and 
    Garmany (1980) showed, on the basis of UV spectra of O stars, that the mass loss rates of O stars show a strong dependence on 
    luminosity class or gravity, in that the rates of Of stars are much larger than those of other O stars, especially main sequence stars. 
    Based on these observations, Chiosi (1981) suggested a parametrization of the mass loss rate in terms of mass, luminosity, and 
    radius, 9HOCL15 (R/M)225, which agrees very well with the predicted dependence for the fluctuation theory of mass loss (Andriesse 1979). 
    The difference between the two sets of parametrization in terms of mass loss during the evolution of a star is enormous. If the mass 
    loss rate depends on the luminosity only, the rate will be about constant when a star evolves from the zero-age main sequence to the 
    supergiant phase. However, if the mass loss depends on L, M, and R as predicted by Andriesse, the mass loss rate will increase by about 
    a factor 25 during this evolution. Luminous early type stars are losing mass at a rate sufficiently large to affect their evolution. 
    In order to understand this effect quantitatively, evolutionary tracks with mass loss have been calculated by various groups (e.g., 
    de Loore, de Greve, and Vanbeveren 1978; Chiosi, Nasi, and Sreenivasan 1978; Maeder 1981)."""

    response = llm_result({"chunk": chunk})
    print(type(response))
    print(response)
    print(response.root)


if __name__ == "__main__":
    main()
