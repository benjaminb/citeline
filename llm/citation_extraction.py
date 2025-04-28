from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json

MODEL_NAME = "llama3.3:latest"  # Replace with your model name

LLM_OUTPUT_FORMAT = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
}

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0,
    format=LLM_OUTPUT_FORMAT,
)

with open("citation_extraction_prompt.txt", "r") as f:
    sys_prompt = f.read()
system_prompt = SystemMessage(content=sys_prompt)


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
    citations = extract_citations(text)
    print(f"Extracted citations: {citations}")


if __name__ == "__main__":
    main()
