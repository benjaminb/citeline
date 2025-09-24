from transformers import pipeline
import torch


def get_pipeline(model_name: str, device_map: str = "auto", torch_dtype: str = "auto"):
    return pipeline(
        task="text-generation",
        model="openai/gpt-oss-20b",
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )


def main():
    llm = get_pipeline("openai/gpt-oss-20b")
    with open("prompts/original_contributions_revised.txt", "r") as f:
        prompt_template = f.read()
    with open("../../../data/paper_bodies/42.txt", "r") as f:
        paper_body = f.read()
    prompt = prompt_template.format(paper=paper_body)

    # Build the input for gpt-oss
    messages = [{"role": "system", "content": prompt}]

    response = llm(messages, max_new_tokens=256, temperature=0.2)
    from pprint import pprint

    pprint(response)
    print("=====")
    print(response[0]["generated_text"][-1])


if __name__ == "__main__":
    main()
