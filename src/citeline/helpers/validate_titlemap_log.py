"""
Validates title matching decisions in log files using an LLM.

Reads WARNING and ERROR lines from title matching logs and determines if the
algorithm made correct decisions.

Usage:
    python validate_titlemap_log.py --log-file acl_20251219_123521.log --output results.json --model llama3.2
"""

import argparse
import json
import re
from pathlib import Path
from typing import Literal
from tqdm import tqdm
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# Prompt templates
NO_MATCH_PROMPT = """You are validating title matching for a research paper database.

The algorithm could not find a match for this title:
Query: "{query_title}"

Task: Does this query look like a valid research paper title, or is it malformed/corrupted?"""

MATCH_VALIDATION_PROMPT = """You are validating title matching for a research paper database.

Query title (from dataset, may have typos/author names): "{query_title}"
Chosen match (canonical title): "{chosen_title}"

Task: Is the chosen match the SAME research paper as the query?

Consider:
- Typos and OCR errors (e.g., "Biligual" vs "Bilingual")
- Author names appended to query (e.g., "Title Here John Smith" → "Title Here")
- Minor wording differences (e.g., "based" vs "tree-based")
- Prefixes like "Squibs:" or section markers
- BUT: Completely different papers should be marked INCORRECT

Examples:

Query: "Last Words"
Chosen: "Latvian WordNet"
Answer: verdict=INCORRECT, confidence=high, reasoning="Completely different papers - no semantic overlap", suggested_action="Manual review needed"

Query: "Going to the Roots of Dependency Parsing"
Chosen: "Squibs: Going to the Roots of Dependency Parsing"
Answer: verdict=CORRECT, confidence=high, reasoning="Same title with 'Squibs:' prefix added", suggested_action="None"
"""


# Pydantic models for structured output
class MatchValidation(BaseModel):
    """Validation result for a title match."""
    verdict: Literal["CORRECT", "INCORRECT", "UNCERTAIN"] = Field(
        description="Whether the match is correct, incorrect, or uncertain"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the verdict"
    )
    reasoning: str = Field(
        description="Brief explanation of the decision"
    )
    suggested_action: str = Field(
        description="What to do if incorrect/uncertain (e.g., 'None', 'Manual review needed')"
    )


class NoMatchValidation(BaseModel):
    """Validation result when no match was found."""
    is_valid_title: bool = Field(
        description="Whether the query looks like a valid research paper title"
    )
    reasoning: str = Field(
        description="Brief explanation"
    )
    suggested_action: str = Field(
        description="What to do (e.g., 'Manual search needed', 'Title is corrupted')"
    )


def parse_log_line(line: str) -> dict | None:
    """
    Extract match information from log lines.

    Returns dict with:
        - level: WARNING or ERROR
        - query_title: The title from the CSV
        - chosen_title: The canonical title chosen (None for ERROR)
        - raw_line: Original log line
    """
    # WARNING format: "Multiple matches on 'QUERY'; choosing CANONICAL"
    warning_pattern = r"WARNING - Multiple matches on '(.+?)'; choosing (.+?)$"
    warning_match = re.search(warning_pattern, line)
    if warning_match:
        return {
            "level": "WARNING",
            "query_title": warning_match.group(1),
            "chosen_title": warning_match.group(2),
            "raw_line": line.strip()
        }

    # ERROR format: "No close matches found for title: QUERY among fuzzy candidates: ..."
    error_pattern = r"ERROR - No close matches found for title: (.+?) among fuzzy candidates:"
    error_match = re.search(error_pattern, line)
    if error_match:
        return {
            "level": "ERROR",
            "query_title": error_match.group(1),
            "chosen_title": None,
            "raw_line": line.strip()
        }

    return None


def validate_match(llm: ChatOllama, query_title: str, chosen_title: str | None) -> dict:
    """
    Use LLM to validate if a match is correct.

    Returns dict representation of MatchValidation or NoMatchValidation.
    """
    if chosen_title is None:
        # ERROR case - no match found
        structured_llm = llm.with_structured_output(NoMatchValidation)
        prompt = NO_MATCH_PROMPT.format(query_title=query_title)

        try:
            result = structured_llm.invoke(prompt)
            return result.model_dump()
        except Exception as e:
            return {
                "is_valid_title": False,
                "reasoning": f"Failed to validate: {str(e)}",
                "suggested_action": "Manual review needed"
            }

    else:
        # WARNING case - multiple matches, chose one
        structured_llm = llm.with_structured_output(MatchValidation)
        prompt = MATCH_VALIDATION_PROMPT.format(
            query_title=query_title,
            chosen_title=chosen_title
        )

        try:
            result = structured_llm.invoke(prompt)
            return result.model_dump()
        except Exception as e:
            return {
                "verdict": "ERROR",
                "confidence": "low",
                "reasoning": f"Failed to validate: {str(e)}",
                "suggested_action": "Manual review needed"
            }


def main():
    parser = argparse.ArgumentParser(description="Validate title matching log with LLM")
    parser.add_argument("--log-file", type=str, required=True, help="Path to log file")
    parser.add_argument("--output", type=str, default="validation_results.json", help="Output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries to validate (for testing)")
    parser.add_argument("--model", type=str, default="llama3.2", help="Ollama model to use (e.g., llama3.2, mistral)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM (0.0 = deterministic)")
    args = parser.parse_args()

    # Initialize ChatOllama
    print(f"Initializing Ollama with model: {args.model}")
    llm = ChatOllama(
        model=args.model,
        temperature=args.temperature,
    )

    # Read and parse log file
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    print(f"Reading log file: {log_path}")
    with open(log_path) as f:
        lines = f.readlines()

    # Extract WARNING and ERROR lines
    entries = []
    for line in lines:
        if "WARNING" in line or "ERROR" in line:
            parsed = parse_log_line(line)
            if parsed:
                entries.append(parsed)

    print(f"Found {len(entries)} warnings/errors to validate")

    if args.limit:
        entries = entries[:args.limit]
        print(f"Limiting to first {args.limit} entries for testing")

    # Validate each entry
    results = []
    for entry in tqdm(entries, desc="Validating matches"):
        validation = validate_match(
            llm,
            entry["query_title"],
            entry["chosen_title"]
        )

        results.append({
            **entry,
            "validation": validation
        })

    # Write results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    by_verdict = {}
    for result in results:
        verdict = result["validation"].get("verdict", "ERROR")
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1

    for verdict, count in sorted(by_verdict.items()):
        print(f"{verdict:15} : {count:4} ({count/len(results)*100:.1f}%)")

    print(f"\nResults saved to: {output_path}")

    # Show a few incorrect matches
    incorrect = [r for r in results if r["validation"].get("verdict") == "INCORRECT"]
    if incorrect:
        print(f"\n{'='*60}")
        print(f"SAMPLE INCORRECT MATCHES ({len(incorrect)} total)")
        print(f"{'='*60}")
        for r in incorrect[:5]:
            print(f"\nQuery: {r['query_title']}")
            print(f"Chosen: {r['chosen_title']}")
            print(f"Reasoning: {r['validation']['reasoning']}")
            print(f"Action: {r['validation']['suggested_action']}")


if __name__ == "__main__":
    main()
