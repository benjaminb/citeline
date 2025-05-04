import argparse
import numpy as np
import pandas as pd
from segeval import boundary_similarity

# from langchain_experimental.text_splitter import SemanticChunker
from LengthPreservingChunker import LengthPreservingChunker
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import re

"""
LOAD DATA
"""
df = pd.read_json("../data/preprocessed/research.jsonl", lines=True, nrows=1000)
np.random.seed(1)
sample_df = df.sample(3)

"""
NOTE: during data exploration we discovered that the first sample paper had an error in its body:
the body text was concatenated with itself so every sentence appeared twice. Here we truncate it
to make its splits more accurate to what we would want in a real-world scenario.
"""
# Get the actual index label of the first row in the sample
row_index = sample_df.index[0]
sample_df.loc[row_index, "body"] = sample_df.loc[row_index, "body"][:86620]  # Use .loc for setting
print(len(sample_df.loc[row_index, "body"]))  # Verify using .loc as well

"""
IMPORT FIRST PARAGRAPH SENTENCES
"""
from first_paragraph_sentences import ukirt, lacey, kobayashi


def get_truncated_sentences(lst: list[str], max_length: int = 75) -> list:
    return [sentence[:max_length] for sentence in lst]


first_paragraph_sentences = [
    get_truncated_sentences(ukirt),
    get_truncated_sentences(lacey),
    get_truncated_sentences(kobayashi),
]

"""
ENSURE SENTENCES ARE 'ACCURATE'
Meaning: all the 'first sentences' are indeed present in the body of the paper
         all the sentences appear in the same order as they do in the body
"""


def are_sentences_in_body(idx) -> bool:
    """
    Check if all sentences in the list are present in the body of the text.
    """
    sentences = first_paragraph_sentences[idx]
    paper_body = sample_df.iloc[idx]["body"]
    all_there = True
    for i, sentence in enumerate(sentences):
        if sentence not in paper_body:
            print(f"Sentence {i} not found: {sentence}")
            all_there = False
    return all_there


for i in range(3):
    assert are_sentences_in_body(i), f"Paper {i} has missing sentences in the body."
print("All sentences are present in the body of the text.")

sentence_indices = []
for i in range(3):
    sentences = first_paragraph_sentences[i]
    paper_body = sample_df.iloc[i]["body"]
    sentence_indices.append([paper_body.index(sentence) for sentence in sentences])

# Confirm the sentences are all in order
all_in_order = True
for i in range(3):
    sent_idx_list = sentence_indices[i]
    for j in range(1, len(sent_idx_list)):
        if sent_idx_list[j] <= sent_idx_list[j - 1]:
            print(f"Paper {i} Sentence {j} is out of order in paper {i}: {sent_idx_list}")
            print(f"  Index of sentence {j}: {sent_idx_list[j]}")
            print(f"  Index of sentence {j-1}: {sent_idx_list[j-1]}")
            all_in_order = False

assert all_in_order, "Some sentences are out of order in the body of the text."
print("All sentences are in order.")

"""
FORM THE REFERENCE LENGTHS LISTS
This is used as the reference for the boundary similarity computations
"""
reference_lengths = []
for i in range(3):
    # Compute all the paragraph lengths
    idx_list = sentence_indices[i]
    paragraph_lengths = [idx_list[j] - idx_list[j - 1] for j in range(1, len(idx_list))]

    # Add the last paragraph length
    paragraph_lengths.append(len(sample_df.iloc[i]["body"]) - idx_list[-1])
    reference_lengths.append(paragraph_lengths)

for i in range(3):
    print(
        f"Paper {i} min length: {min(reference_lengths[i])}, max length: {max(reference_lengths[i])}, mean length: {np.mean(reference_lengths[i])}, std: {np.std(reference_lengths[i])}"
    )
    assert sum(reference_lengths[i]) == len(
        sample_df.iloc[i]["body"]
    ), f"Paper {i} length mismatch: {sum(reference_lengths[i])} != {len(sample_df.iloc[i]['body'])}"


def evaluate_params(
    model_name: str,
    breakpoint_threshold_type: str,
    breakpoint_threshold_amount: float,
    min_chunk_size: int,
):
    """
    Instantiates a chunker with the given parameters and evaluates its boundary similarity on the reference chunks
    """
    # Set up the chunker
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.mps.is_available() else "cpu"
            )
        },
        encode_kwargs={"normalize_embeddings": False},
    )
    chunker = LengthPreservingChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_size=min_chunk_size,  # in chars
    )

    scores = []
    for i in range(3):
        # Get the predicted chunks
        chunks = chunker.split_text(sample_df.iloc[i]["body"])
        chunk_lengths = [len(chunk) for chunk in chunks]

        # Compute the boundary similarity score
        score = boundary_similarity(reference_lengths[i], chunk_lengths)
        scores.append(float(score))
        print(
            f"[{model_name}]:({breakpoint_threshold_type}:{breakpoint_threshold_amount}), min chunk size {min_chunk_size}: {score}"
        )
    print("=======")

    scores = [float(score) for score in scores]
    average_score = sum(scores) / len(scores)
    with open("chunker_scores.csv", "a") as f:
        f.write(
            f'{model_name},{breakpoint_threshold_type},{breakpoint_threshold_amount},{min_chunk_size},{average_score},"{scores}"\n'
        )
    return average_score


breakpoints = {
    "percentile": [n for n in range(1, 100, 3)],
    "gradient": [n for n in range(1, 100, 3)],
    "standard_deviation": [0.1 * n for n in range(1, 40)],
    "interquartile": [0.1 * n for n in range(1, 40)],
}
MIN_CHUNK_SIZES = [50, 100]


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunker parameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name for embeddings.",
    )
    args = parser.parse_args()
    model = args.model_name

    # Generate configs for grid search
    all_configs_kwargs = []
    for bp_type, bp_amounts in breakpoints.items():
        for bp_amount in bp_amounts:
            for chunk_size in MIN_CHUNK_SIZES:
                # Create the kwargs dictionary
                config_kwargs = {
                    "model_name": model,
                    "breakpoint_threshold_type": bp_type,
                    "breakpoint_threshold_amount": bp_amount,
                    "min_chunk_size": chunk_size,
                }
                all_configs_kwargs.append(config_kwargs)

    # Print the first few configurations to verify
    print(f"Total configurations generated: {len(all_configs_kwargs)}")

    for config in all_configs_kwargs:
        score = evaluate_params(**config)
        print(f"Config: {config}, Score: {score}")
    print("All configurations evaluated.")


if __name__ == "__main__":
    main()
