import csv
import difflib
import json
import logging
import os
from pathlib import Path
import pymupdf
import re
import requests
import string
from abc import ABC, abstractmethod
from datetime import datetime
from dotenv import load_dotenv
from itertools import islice
from semantic_text_splitter import TextSplitter
from time import sleep
from tqdm import tqdm

from pydantic import BaseModel, Field

# Path to project root
PROJECT_ROOT = Path("../" * 3)

PATH_TO_DATA = PROJECT_ROOT / "data" / "datasets"


class ReferenceDatasetProcessor(ABC):
    """
    Subclass this for each reference dataset (ACL, arXiv, etc.)

    The subclass creates these files:
    1. Cleaned dataset: f"cleaned_{data_file}" (original dataset with cleaned titles)
    2. Title map file: f"{dataset_name}_titlemap.json" (tracks successfully processed titles as {original_title: resolved_title})
        - original_title: the title string as it appears in the dataset
        - resolved_title: the title string as represented in the source (e.g., ACL Anthology)
        NOTE: These can be different due to typos in the dataset (missing spaces, punctuation, adding authors, etc.)
    3. Title to ID map: f"{dataset_name}_title_to_id.json" (maps resolved titles to document IDs)
    4. Log file: f"{dataset_name}.log" (logs errors, warnings, and processing info)
    """

    def __init__(self, name: str, data_path: Path, data_file: str):
        """
        Args:
            name (str): Name of the dataset (e.g., "acl", "arxiv")
            data_path (Path): Path to the dataset directory
            data_file (str): Filename of the dataset (e.g., "context_dataset_eval.csv")
        """
        self.name = name
        self.data_path = data_path
        self.data_file = data_file
        self.data_filepath = data_path / data_file

        # Set up path for files
        self.save_dir = data_path / "processed"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set up files
        self.cleaned_dataset = f"cleaned_{data_file}"
        self.cleaned_dataset_path = self.save_dir / self.cleaned_dataset

        self.titlemap_file = f"{name}_titlemap.json"
        self.titlemap_path = self.save_dir / self.titlemap_file

        self.title_to_id_file = f"{name}_title_to_id.json"
        self.title_to_id_path = self.save_dir / self.title_to_id_file

        self.progress_file = f"{name}.progress"
        self.progress_file_path = self.save_dir / f"{name}.progress"

        self.logfile = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Load title map if it exists
        self.titlemap = {}
        if self.titlemap_path.exists():
            with open(self.titlemap_path, "r") as f:
                self.titlemap = json.load(f)
                print(f"Resuming with \033[1m{len(self.titlemap)}\033[0m title spellings mapped to canonical titles")
        else:
            print(f"\033[1mNo existing title map found\033[0m at {self.titlemap_path}, starting fresh.")

        # Load title to ID map if it exists
        self.title_to_id = {}
        if self.title_to_id_path.exists():
            with open(self.title_to_id_path, "r") as f:
                self.title_to_id = json.load(f)
                print(f"Resuming with \033[1m{len(self.title_to_id)}\033[0m titles resolved to IDs]]")
        else:
            print(f"\033[1mNo existing title to ID map found\033[0m at {self.title_to_id_path}, starting fresh.")

        # Set up logging
        logging.basicConfig(
            filename=self.logfile,  # path to the log file
            level=logging.INFO,  # minimum level to log
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.norm_title_to_canonical = self.build_norm_title_to_canonical_map()
        self.normed_canonical_titles = set(self.norm_title_to_canonical.keys())

    @abstractmethod
    def build_norm_title_to_canonical_map(self) -> dict[str, str]:
        """Subclasses implement this to return a mapping of normed titles to canonical titles from the source"""
        pass

    @abstractmethod
    def resolve_pubdate(self, data) -> int:
        """Subclasses implement this to return YYYYMMDD int from provided data (paper reference or other)"""
        pass

    @abstractmethod
    def title_search(self, title: str) -> str | None:
        """
        Subclasses implement this to return list of possible title matches from source

        This should do an exact match search first. If only one title matches, return that.
        Otherwise, perform a _normed_title_search

        Returns:
            str | None: The matched title string, or None if no precise match found
        """
        pass

    def _front_half(self, s: str) -> str:
        """Returns the front half of a string."""
        return s[: len(s) // 2]

    def _back_half(self, s: str) -> str:
        """Returns the back half of a string."""
        return s[len(s) // 2 :]

    def _get_batch(self, reader: csv.DictReader, batch_size: int = 20):
        """Yields batches of rows from the CSV reader."""
        while True:
            batch = list(islice(reader, batch_size))
            if not batch:
                break
            yield batch

    def _normed_title_search(self, title: str) -> str | None:
        """Norms the title and searches against normed titles"""
        normed_title = self._norm_title(title)
        # normed_reference_titles = self.get_normed_reference_titles()

        # Truncate the normed names in case punctuation or minor typos prevent substring matching
        normed_title_front, normed_title_back = self._front_half(normed_title), self._back_half(normed_title)
        candidates = [
            t
            for t in self.normed_canonical_titles
            if any(
                [
                    normed_title_front in t,
                    normed_title_back in t,
                    self._front_half(t) in normed_title,
                    self._back_half(t) in normed_title,
                ]
            )
        ]

        # Only one match: return it
        if len(candidates) == 1:
            self.logger.info(f"Single match found in substring search for title '{title}': {candidates[0]}")
            return candidates[0]

        # Multiple matches: use difflib to find closest match. Returns [] if candidates=[]
        diff_matches = difflib.get_close_matches(normed_title, candidates, n=3, cutoff=0.6)

        # No matches: fuzzy search on all normed titles
        if not diff_matches:
            diff_matches = difflib.get_close_matches(normed_title, self.normed_canonical_titles, n=3, cutoff=0.5)

        # Still no matches: log error
        if len(diff_matches) == 0:
            self.logger.error(
                f"No close matches found for title: {title} among fuzzy candidates: {candidates[:5]}{'...' if len(candidates) > 5 else ''}"
            )
            return None

        canonical_title = self.norm_title_to_canonical[diff_matches[0]]
        # Multiple matches: log warning and return best match
        if len(diff_matches) > 1:
            self.logger.warning(f"Multiple matches on '{title}'; choosing {canonical_title}")

        return canonical_title

    def _norm_title(self, title: str) -> str:
        """Normalizes titles for matching, removing punctuation and whitespace, and lowercasing.
        E.g. "Multi-Word Units: How Efficient" -> "multiwordunitshowefficient"
        """
        # return title.translate(str.maketrans("", "", string.punctuation + string.whitespace)).lower()
        return "".join(char.lower() for char in title if char.isalnum())

    def create_canonical_titlemap(self):
        """Attempts to resolve the typos in titles in the dataset to canonical titles from the source."""
        # Get the length of the dataset to be processed to inform tqdm bar
        with open(self.data_filepath, "r") as f:
            total_lines = sum(1 for line in f)

        with open(self.data_filepath, "r") as infile:
            reader = csv.DictReader(infile)
            pbar = tqdm(total=total_lines, desc="Processing dataset", unit="rows")

            reader = csv.DictReader(infile)
            for batch in self._get_batch(reader, batch_size=100):
                for row in batch:
                    target_title = row["target_title"]
                    citing_title = row["citing_title"]

                    pbar.update(1)
                    for title in (target_title, citing_title):
                        if title in self.titlemap:
                            continue

                        # If title resolves, it goes into the map. Otherwise, the unresolve title goes in as key with None value
                        # The task becomes to resolve these None values later
                        canonical_title = self.title_search(title)
                        self.titlemap[title] = canonical_title

                # Write out updated title map
                with open(self.titlemap_path, "w") as titlemap_file:
                    json.dump(self.titlemap, titlemap_file, indent=2)

            pbar.close()


class ACLProcessor(ReferenceDatasetProcessor):

    def __init__(self, data_path: Path, data_file: str):
        from acl_anthology import Anthology

        self.anthology = Anthology.from_repo()
        super().__init__(name="acl", data_path=data_path, data_file=data_file)

    def build_norm_title_to_canonical_map(self) -> dict[str, str]:
        return {self._norm_title(str(p.title)): str(p.title) for p in self.anthology.papers()}

    def resolve_pubdate(self, paper) -> int:
        """
        The ACL api gives year (as str) and month as full name
          paper.year -> "1999"
          paper.month -> "March"

        We don't have day info, so we default to the datetime.strptime default (day=1).
        NOTE: pubdate is only used to artificially prevent citing future papers during evals.
          The maximum error is 30 days, and it's unlikely any paper published cites so recently published work. BB
        """
        month_num = 1  # Default to January

        if paper.month:
            month_str = str(paper.month).strip().lower()
            for name, num in self.month_map.items():
                if name in month_str:
                    month_num = num
                    break
            else:
                self.logger.warning(
                    f"Could not parse month for paper {paper.full_id} with month '{paper.month}'. Defaulting to January."
                )

        return int(f"{paper.year}{month_num:02d}01")  # YYYYMMDD as int

    def title_search(self, title: str) -> str | None:
        title_str = str(title)
        title_normed = self._norm_title(title_str)

        # Exact match search
        if title_normed in self.norm_title_to_canonical:
            return self.norm_title_to_canonical[title_normed]  # Return the canonical title
        # Fallback to normed title search
        return self._normed_title_search(title)


class TitleMapPostprocessor:
    def __init__(self, model_name: str, input_path: str, output_path: str):

        # Read in log file
        self.input_path = input_path
        with open(input_path, "r") as file:
            lines = file.readlines()
        self.lines = lines
        print(f"Read in {len(lines)} lines...")

        # Set up LLM
        from citeline.llm.llm_function import LLMFunction
        from citeline.llm.models import TitleCheckResponse

        self.titlechecker = LLMFunction(
            model_name=model_name,
            prompt_path="../llm/prompts/reference_title_check.txt",
            output_model=TitleCheckResponse,
        )

        self.logline_pattern = re.compile(r"WARNING - Multiple matches on '(.+?)'; choosing (.+?)$")
        self.output_path = output_path

    def get_titles_from_line(self, line: str) -> tuple[str, str]:
        """
        Given a WARNING log line from the output of a ReferenceDatasetProcessor (uncertain fuzzy match),
        extract and return
        (dataset title, candidate canonical title)
        """
        match = self.logline_pattern.search(line)
        if not match or len(match.groups()) != 2:
            print(f"Could not extract title from line:\n  {line}")
            return None, None
        return match.group(1), match.group(2)

    def process(self):
        for line in tqdm(self.lines):
            dataset_title, candidate_title = self.get_titles_from_line(line)
            if dataset_title is None or candidate_title is None:
                continue
            titles = {"dataset_title": dataset_title, "candidate_title": candidate_title}
            llm_response = self.titlechecker(titles)

            # Log the lines that need manual check
            if not llm_response.is_match:
                with open(self.output_path, "a") as outfile:
                    response_dict = titles | {"reasoning": llm_response.reasoning}
                    json.dump(response_dict, outfile)
                    outfile.write("\n")

    # take a path to a log to process
    # method to parse log lines
    # llm function to process log lines
    # output path for results file

"""
Upgrade the ACL dataset

Preconditions:
1. Every title in the dataset is in the title map
2. Every value in the title map uniquely finds a paper in the ACL Anthology with an ACL ID
3. Every ACL ID is retrievable from the ACL Anthology API and is more or less clean (contains full body text, minimal OCR errors)


1. load the dataset and canonical title map
1.5 load the tran and eval datasets
2. For each row, map the target and citing titles to the canonical titles
3. get the paper IDs for the canonical titles
4. add keys citing_id and target_id
5. Get the corresponding row from train or eval dataset
6. replace the rows in the train or eval dataset with the updated rows
"""



def main():
    # data_path = PATH_TO_DATA / "acl200_global"
    # data_file = "context_dataset.csv"
    # processor = ACLProcessor(data_path=data_path, data_file=data_file)
    # processor.create_canonical_titlemap()
    # Further processing logic would go here
    postprocessor = TitleMapPostprocessor(
        model_name="gpt-oss:20b",
        input_path="acl_20251219_123521.log",
        output_path="acl_titles_to_check.jsonl",
    )
    postprocessor.process()


if __name__ == "__main__":
    main()
