from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json

MODEL_NAME = "llama3.2:1b"

LLM_OUTPUT_FORMAT = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
}

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.01,
).with_structured_output(LLM_OUTPUT_FORMAT)

with open("citation_extraction_prompt.txt", "r") as f:
    system_prompt = f.read()


def extract_citations(text):
    msg = HumanMessage(content=text)

    response = llm.invoke([system_prompt, msg])
    ai_message = response.content
    print(f"AI message: {ai_message}")

    try:
        citations = json.loads(ai_message.content)
        return citations
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
