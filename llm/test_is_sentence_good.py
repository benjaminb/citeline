import json
import os
from tqdm import tqdm
from pydantic import BaseModel, Field
import time
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)

MODEL_NAME = "mistral-nemo:latest"  # Replace with your model name


# Define the Pydantic model for sentence validation
class SentenceValidation(BaseModel):
    reasoning: str = Field(
        description="Brief reasoning for why the sentence is valid or not. Consider if it's natural language, has clear meaning, and is not a figure caption, table content, reference list item, or gibberish."
    )
    isValid: bool = Field(
        description="True if the sentence is a 'good' scientific sentence (natural language, clear meaning), False otherwise (e.g., caption, reference, gibberish, OCR error)."
    )


def main():
    # Load data
    data_file_path = "../data/etc/processed_passages.json"
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {os.path.abspath(data_file_path)}")
        print(f"Current working directory: {os.getcwd()}")
        return

    with open(data_file_path, "r") as file:
        data = json.load(file)
        print(f"Loaded {len(data)} records from {file.name}")

    # Initialize ChatOllama model
    try:
        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0.0,
        )
    except Exception as e:
        print(f"Error initializing ChatOllama: {e}")
        print(
            "Please ensure Ollama server is running and the specified model (e.g., 'mistral') is available."
        )
        return

    # Set up PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=SentenceValidation)

    # Define the prompt template
    prompt_template_str = """
You are a sentence validator for scientific texts.
Your task is to determine if a given sentence is a "good" sentence.
A "good" sentence is one that:
1. Is written in natural language.
2. Has a clear and understandable meaning.
3. Is a complete thought or statement.
4. Is NOT primarily a figure caption (e.g., "Figure 1 shows..." but not "Figure 1. The results."), table data, part of a reference list, an equation, or unintelligible gibberish/OCR errors.

Sentences that are clearly from a bibliography/reference section (e.g., "Author, A. et al. (2020) J. Sci.") are NOT valid.
Sentences with severe OCR errors or that are mostly mathematical symbols without clear prose are NOT valid.
Short, label-like figure/table captions (e.g., "Table 1. Summary statistics.") are NOT valid.
However, if a sentence refers to a figure or table but is itself a complete, meaningful statement (e.g., "As shown in Figure 3, the values increased significantly."), it CAN be valid.

Based on the input sentence, provide your reasoning and a boolean 'isValid' flag.

{format_instructions}

Sentence to validate:
{sentence}
"""
    with open("is_sentence_good_prompt.txt", "r") as f:
        prompt_template_str = f.read()

    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["sentence"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    y_true = []
    y_pred = []

    # Temporarily truncate data
    # data = data[:3]

    num_total_records = len(data)
    num_successfully_predicted = 0
    num_errors_during_prediction = 0

    # Create log filename with timestamp
    log_file_path = f"mislabeled_sentences_{MODEL_NAME.replace(':', '_')}_{int(time.time())}.jsonl"

    print(f"Processing {num_total_records} sentences...")
    print(f"Mislabeled sentences will be logged to: {os.path.abspath(log_file_path)}")

    with open(log_file_path, "w") as log_file:
        for i, record in tqdm(enumerate(data), total=num_total_records):
            sentence_text = record["sentence"]
            expected_isValid = record["isValid"]

            try:
                llm_output = chain.invoke({"sentence": sentence_text})
                predicted_isValid = llm_output.isValid
                reasoning = llm_output.reasoning

                y_true.append(expected_isValid)
                y_pred.append(predicted_isValid)
                num_successfully_predicted += 1

                # Log errors
                if predicted_isValid != expected_isValid:
                    log_entry = {
                        "sentence": sentence_text,
                        "expected": expected_isValid,
                        "predicted": predicted_isValid,
                        "reasoning": reasoning,
                    }
                    log_file.write(json.dumps(log_entry) + ",\n")
                    print(f"Logged error: {log_entry}")
            except Exception as e:
                print(
                    f"\nError processing sentence (idx {i}): '{sentence_text[:100]}...'. Error: {e}"
                )
                num_errors_during_prediction += 1
                # This record will be skipped for metrics as we don't have a valid prediction.
                continue

            if (i + 1) % 50 == 0:  # Log progress periodically
                if len(y_true) > 0 and len(y_pred) > 0:
                    current_accuracy = accuracy_score(y_true, y_pred)
                    print(
                        f"\nProcessed {i+1}/{num_total_records}. Current cumulative accuracy on {len(y_pred)} evaluated examples: {current_accuracy*100:.2f}%"
                    )

    if not y_pred or not y_true:
        print(
            "\nNo predictions were successfully made or no true labels available. Cannot calculate metrics."
        )
        if num_errors_during_prediction > 0:
            print(f"Number of sentences skipped due to errors: {num_errors_during_prediction}")
        return

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Use labels=[False, True] to ensure a 2x2 matrix for TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    print("\n--- Evaluation Metrics ---")
    print(f"Total sentences provided: {num_total_records}")
    print(f"Sentences successfully evaluated: {num_successfully_predicted}")
    if num_errors_during_prediction > 0:
        print(f"Sentences skipped due to prediction errors: {num_errors_during_prediction}")

    print(f"\nMetrics based on {len(y_true)} successfully evaluated sentences:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print("              Pred False  Pred True")
    print(f"Actual False    {cm[0][0]:<10} {cm[0][1]:<10} (TN, FP)")
    print(f"Actual True     {cm[1][0]:<10} {cm[1][1]:<10} (FN, TP)")


if __name__ == "__main__":
    main()
