from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json
from pydantic import RootModel, Field, BaseModel

# MODEL_NAME = "llama3.3:latest"  # Replace with your model name
MODEL_NAME = "mistral-nemo:latest"

# LLM_OUTPUT_FORMAT = {
#     "type": "array",
#     "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
# }


class IsValidSentence(RootModel[bool]):
    pass


class SentenceValidator(BaseModel):
    reasoning: str = Field(description="Reasoning for the label")
    label: bool = Field(description="True if the sentence is usable, False otherwise")


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0,
).with_structured_output(SentenceValidator, method="json_schema")

with open("is_sentence_good_prompt.txt", "r") as f:
    sys_prompt = f.read()
system_prompt = SystemMessage(content=sys_prompt)

# with open("citation_extraction_prompt.txt", "r") as f:
#     sys_prompt = f.read()
# system_prompt = SystemMessage(content=sys_prompt)


def is_sentence_valid(text):
    msg = HumanMessage(content=text)
    ai_message = llm.invoke([system_prompt, msg])
    print(f"AI message: {type(ai_message)}")
    try:
        return ai_message.label
    except json.JSONDecodeError as e:
        print(f"(is_sentence_valid) Error decoding llm response: {e}")
        return False


def extract_citations(text):
    msg = HumanMessage(content=text)
    ai_message = llm.invoke([system_prompt, msg])

    try:
        citations = json.loads(ai_message.content)
        return citations
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def main():
    # Example usage
    text = "It also hosts a Compton-thick AGN in the Western component, observed directly in hard X-rays (Della Ceca et al. 2002 ; Ballo et al. 2004 )."
    # citations = extract_citations(text)
    # print(f"Extracted citations: {citations}")
    is_valid = is_sentence_valid(text)
    print(f"Is the sentence valid? {is_valid}")
    print(f"Type returned: {type(is_valid)}")


if __name__ == "__main__":
    main()
