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

    @abstractmethod
    def get_normed_reference_titles(self) -> set:
        """Subclasses implement this to return set of all unique normed reference titles in the dataset"""
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

    def _normed_title_search(self, title: str) -> str | None:
        """Norms the title and searches against normed titles"""
        normed_title = self._norm_title(title)
        normed_reference_titles = self.get_normed_reference_titles()

        # Truncate the normed names in case punctuation or minor typos prevent substring matching
        normed_title_front, normed_title_back = self._front_half(normed_title), self._back_half(normed_title)
        matches = [
            t
            for t in normed_reference_titles
            if any(
                [
                    normed_title_front in t,
                    normed_title_back in t,
                    self._front_half(t) in normed_title,
                    self._back_half(t) in normed_title,
                ]
            )
        ]

        # No matches or multiple matches
        if not matches:
            self.logger.error(f"No substring matches found for title: {title}")
            return None
        if len(matches) == 1:
            self.logger.warning(f"Single match found in substring search for title '{title}': {matches[0]}")
            return matches[0]

        # Multiple matches: use difflib to find closest match
        diff_matches = difflib.get_close_matches(title, matches, n=3, cutoff=0.8)
        if not diff_matches:
            self.logger.error(f"No close matches found for title: {title} among fuzzy candidates: {matches}")
            return None

        # Either 1 or more fuzzy matches
        if len(diff_matches) > 1:
            self.logger.warning(
                f"Multiple matches found for title '{title}': {matches}. Using difflib to find closest match."
            )

        return diff_matches[0]

    def _norm_title(self, title: str) -> str:
        """Normalizes titles for matching (lowercase, strip whitespace)."""
        return title.translate(str.maketrans("", "", string.punctuation + string.whitespace)).lower()

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
                            canonical_title = self.titlemap[title]
                        else:
                            # Try direct title search
                            canonical_title = self.title_search(title)
                            if not canonical_title:
                                continue
                            self.titlemap[title] = canonical_title

                # Write out updated title map
                with open(self.titlemap_path, "w") as titlemap_file:
                    json.dump(self.titlemap, titlemap_file, indent=2)

            pbar.close()


class ACLProcessor(ReferenceDatasetProcessor):

    def __init__(self, data_path: Path, data_file: str):
        from acl_anthology import Anthology

        super().__init__(name="acl", data_path=data_path, data_file=data_file)
        self.anthology = Anthology.from_repo()

        self.normed_reference_titles = {self._norm_title(str(p.title)) for p in self.anthology.papers()}

    def get_normed_reference_titles(self) -> set:
        return self.normed_reference_titles

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
        for paper in self.anthology.papers():
            if self._norm_title(str(paper.title)) == title_normed:
                return str(paper.title)  # Return the canonical title
        # Fallback to normed title search
        return self._normed_title_search(title)


def main():
    data_path = PATH_TO_DATA / "acl200_global"
    data_file = "context_dataset.csv"
    processor = ACLProcessor(data_path=data_path, data_file=data_file)
    processor.create_canonical_titlemap()
    # Further processing logic would go here


if __name__ == "__main__":
    main()
