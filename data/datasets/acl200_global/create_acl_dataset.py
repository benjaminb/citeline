import csv
import difflib
import json
import logging
import os
import pymupdf
import re
import requests
from datetime import datetime
from dotenv import load_dotenv
from itertools import islice
from semantic_text_splitter import TextSplitter
from time import sleep
from tqdm import tqdm

from acl_anthology import Anthology


# Semantic Scholar API key is loaded
# load_dotenv("../../../.env")
# assert os.getenv("SEMANTIC_SCHOLAR_API_KEY"), "SEMANTIC_SCHOLAR_API_KEY not found in .env"
# API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Logging set up
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/acl_pull_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log",  # path to the log file
    level=logging.INFO,  # minimum level to log
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:

    def block_is_text(block: tuple) -> bool:
        return block[6] == 0  # block type 0 indicates text

    def clean_block_text(text: str) -> str:
        """
        Cleans up the PDF formatting artifacts preserved in pymupdf 'blocks': newlines for line breaks and
        hyphenation (word division) at line ends.

        These are artifacts of PDF processing for paper; not necessary or helpful for digital texts or NLP.
        """
        # Replace hyphen at end of line followed by newline with nothing (rejoin hyphenated words)
        text = re.sub(r"-\s*\n\s*", "", text)

        # Replace remaining newlines with spaces
        text = re.sub(r"\n+", " ", text)

        # Clean up multiple spaces
        text = re.sub(r" +", " ", text)

        return text.strip()

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            if block_is_text(block):
                cleaned_text = clean_block_text(block[4])
                text += cleaned_text + "\n"

    doc.close()
    return text.strip()


class ACLProcessingError(Exception):
    """Custom exception for ACLProcessor errors."""

    pass


class ACLClient:

    PDF_SAVE_DIR = "pdfs/"
    SAVE_FILE = "preprocessed/acl_papers.jsonl"
    PROCESSED_SAVE_DIR = "preprocessed/"
    PROGRESS_PATH = os.path.join(PROCESSED_SAVE_DIR, "acl.progress")  # acl_id1\nacl_id2\n...

    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    def __init__(self, dataset: str = "context_dataset.csv"):
        self.anthology = Anthology.from_repo()
        self.data_path: str = dataset
        self.acl_titles_normed: set = {str(p.title).lower() for p in self.anthology.papers()}

        # Ensure save dirs exist
        os.makedirs(self.PDF_SAVE_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_SAVE_DIR, exist_ok=True)
        self.splitter = TextSplitter(capacity=1500, overlap=150)

        if os.path.exists(self.PROGRESS_PATH):
            with open(self.PROGRESS_PATH, "r") as f:
                self.processed_titles = set(line.strip() for line in f)
        else:
            print(f"Progress file not found at \033[1m{self.PROGRESS_PATH}\033[0m. Starting fresh.")
            self.processed_titles = set()

    def _get_batch(self, reader: csv.DictReader, batch_size: int = 20):
        """Yields batches of rows from csv.DictReader."""
        while True:
            batch = list(islice(reader, batch_size))
            if not batch:
                break
            yield batch

    def _resolve_paper_pubdate(self, paper) -> int:
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
                logger.warning(
                    f"Could not parse month for paper {paper.full_id} with month '{paper.month}'. Defaulting to January."
                )

        return int(f"{paper.year}{month_num:02d}01")  # YYYYMMDD as int

    def title_to_paper(self, title: str):

        title_normed = title.lower().strip()

        # If title has typo, try finding its closest fuzzy match
        if not title_normed in self.acl_titles_normed:
            matches = difflib.get_close_matches(
                word=title_normed, possibilities=self.acl_titles_normed, n=3, cutoff=0.6
            )

            # No close matches: raise error
            if not matches:
                msg = f"Title not found in ACL anthology, and no close matches found: {title}"
                logger.error(msg)
                raise ACLProcessingError(msg)

            logger.warning(
                f"Title not found exactly; using closest match: ORIGINAL: {title_normed}, MATCH: {matches[0]}"
            )
            # Use closest match
            title_normed = matches[0]

        # We should now have a title that matches a paper in the anthology
        candidates = []
        for paper in self.anthology.papers():
            if title_normed in str(paper.title).lower():
                candidates.append(paper)

        if len(candidates) > 1:
            # Return an exact candidate if possible, best match otherwise
            for candidate in candidates:
                if title_normed == str(candidate.title).lower().strip():
                    return candidate
            else:
                matches = difflib.get_close_matches(
                    word=title_normed, possibilities=[str(c.title).lower().strip() for c in candidates], n=1, cutoff=0.0
                )
                logger.warning(f"Multiple candidates found for title '{title}'; returning best match '{matches[0]}'")
                return matches[0]

        if len(candidates) == 0:
            raise ACLProcessingError(f"No matching paper found for title: {title}")
        return candidates[0]

    def paper_to_text(self, paper) -> str:
        # Try to get the pdf url from the paper object; fallback to constructing from full_id
        if hasattr(paper, "pdf") and hasattr(paper.pdf, "url"):
            pdf_url = paper.pdf.url
        else:
            pdf_url = f"https://aclanthology.org/{paper.full_id}.pdf"

        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            msg = f"Failed to download PDF for ACL ID {paper.full_id}: {e}"
            logger.error(msg)
            raise ACLProcessingError(msg) from e
        return pdf_bytes_to_text(response.content)

    def acl_dataset_to_jsonl(self):
        """
        Takes the ACL dataset at self.data_path and downloads the PDFs for each target_title and citing_title
        and saves their full texts to JSONL"""

        # Get dataset length
        with open(self.data_path, "r") as f:
            total_lines = sum(
                1 for line in f
            )  # We WOULD subtract 1 for header, but since we update pbar at start of loop that cancels out

        with open(self.data_path, "r") as data_file, open(self.PROGRESS_PATH, "a") as progress_file, open(
            self.SAVE_FILE, "a"
        ) as save_file:
            reader = csv.DictReader(data_file)

            with tqdm(total=total_lines, desc="Fetching papers") as pbar:
                for batch in self._get_batch(reader):
                    titles = set()
                    for row in batch:
                        titles.add(row["target_title"])
                        titles.add(row["citing_title"])

                    for title in titles:
                        pbar.update(1)

                        title_lower = title.lower().strip()
                        if title_lower in self.processed_titles:
                            continue

                        try:
                            # Offline processing
                            paper = self.title_to_paper(title)
                            paper_title = str(
                                paper.title
                            )  # Use the official title from Paper object, dataset has typos
                            paper_title_lower = paper_title.lower().strip()
                            if paper_title_lower in self.processed_titles:
                                continue
                            acl_id = paper.full_id
                            pubdate = self._resolve_paper_pubdate(paper)

                            # HTTP request to get PDF text
                            sleep(1)  # respect rate limits
                            text = self.paper_to_text(paper)

                            # Construct record
                            record = {"title": paper_title, "acl_id": acl_id, "pubdate": pubdate, "text": text}

                            # Save record and update progress
                            save_file.write(json.dumps(record) + "\n")

                            # Save progress (both dataset title, which may have typos, and official paper title)
                            progress_file.write(title_lower + "\n")
                            self.processed_titles.add(title_lower)
                            if paper_title_lower != title_lower:
                                progress_file.write(paper_title_lower + "\n")
                                self.processed_titles.add(paper_title_lower)

                        except ACLProcessingError as e:
                            logger.error(f"Skipping title '{title}' due to error: {e}")
                            continue

                    # Flush batch to disk
                    save_file.flush()
                    progress_file.flush()

                # Final flush to disk
                save_file.flush()
                progress_file.flush()


class ACLProcessor:
    """
    The main purpose of this class is to take string titles of papers and resolve them to the text
    of their bodies."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    PDF_SAVE_DIR = "pdfs/"
    SAVE_FILE = "preprocessed/research.jsonl"
    PROCESSED_SAVE_DIR = "preprocessed/"
    PROGRESS_PATH = os.path.join(PROCESSED_SAVE_DIR, "titles.progress")  # title1\ntitle2\n...

    def __init__(self, api_key: str, data_path: str = "context_dataset_train.csv", batch_size: int = 50):
        """
        Args:
            api_key: Semantic Scholar API key
            data_path: path to ACL200 dataset (train or eval).
                (CSV: masked_cit_context,masked_token_target,citing_title,citing_abstract,target_title,target_abstract)
        """
        self.api_key = api_key
        self.data_path = data_path
        print(f"Using data: \033[1m{self.data_path}\033[0m")
        self.batch_size = batch_size

        # Ensure save dirs exist
        os.makedirs(self.PDF_SAVE_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_SAVE_DIR, exist_ok=True)
        self.splitter = TextSplitter(capacity=1500, overlap=150)

        if os.path.exists(self.PROGRESS_PATH):
            with open(self.PROGRESS_PATH, "r") as f:
                self.processed_titles = set(line.strip() for line in f)
        else:
            print(f"Progress file not found at \033[1m{self.PROGRESS_PATH}\033[0m. Starting fresh.")
            self.processed_titles = set()

    def _batch_reader(self, reader: csv.DictReader):
        """Yields batches of rows from a csv.DictReader."""
        while True:
            batch = list(islice(reader, self.batch_size))
            if not batch:
                break
            yield batch

    def _get_processed_titles(self) -> set:
        """Reads the progress file and returns a set of already processed titles."""
        if not os.path.exists(self.PROGRESS_PATH):
            return set()
        with open(self.PROGRESS_PATH, "r") as file:
            processed_titles = {line.strip() for line in file}
        return processed_titles

    def acl_dataset_to_jsonl(self):
        """
        Takes in an ACL dataset csv file and produces a jsonl file of records with full body text.
        """
        # Get dataset length
        with open(self.data_path, "r") as f:
            total_lines = sum(1 for line in f) - 1  # subtract 1 for header

        with open(self.data_path, "r") as data_file, open(self.PROGRESS_PATH, "a") as progress_file, open(
            self.SAVE_FILE, "a"
        ) as save_file:
            reader = csv.DictReader(data_file)

            with tqdm(total=total_lines, desc="Fetching papers") as pbar:
                for batch in self._batch_reader(reader):
                    target_titles = [row["target_title"] for row in batch]
                    for title in target_titles:

                        pbar.update(1)

                        if title in self.processed_titles:
                            print(f"Title already processed, skipping: {title}")
                            continue

                        sleep(1)  # Sleep to respect rate limits
                        try:
                            record = self.title_to_record(title)

                            # Save record and update progress
                            save_file.write(json.dumps(record) + "\n")
                            progress_file.write(title + "\n")
                            self.processed_titles.add(title)

                        except ACLProcessingError as e:
                            logger.error(f"Skipping title {title} due to error: {e}")
                            continue

                    # Flush files after batch
                    save_file.flush()
                    progress_file.flush()

                # Final flush to disk
                save_file.flush()
                progress_file.flush()

    def title_to_ids_and_pubdate(self, title: str) -> dict:
        """

        Returns:
            an empty dictionary, indicating some failure,
            or a dict with "acl" and "doi" ids
        """
        params = {"query": title, "fields": "externalIds,publicationDate", "limit": 1}
        headers = {"x-api-key": self.api_key}
        try:
            res = requests.get(f"{self.BASE_URL}/paper/search", params=params, headers=headers)
        except requests.exceptions.RequestException as e:
            logger.error(f"Title: {title}. Request failed: {e}")
            return {}

        obj = res.json()
        data: list = obj.get("data", [])
        if not data:
            msg = f"Title: {title}. No records found in Semantic Scholar's response. Response: {obj}"
            logger.error(msg)
            raise ACLProcessingError(msg)

        record = data[0] if data else None
        if not record:
            msg = f"Title: {title}. No records found in Semantic Scholar's response list for the given title. Response: {obj}"
            logger.error(msg)
            raise ACLProcessingError(msg)

        # Get pubdate (nonfatal if missing, but log warning)
        pubdate = record.get("publicationDate", None)
        if not pubdate:
            logger.warning(
                f"Title: {title}. No 'publicationDate' found in record. Be sure to get this filled in! Response: {obj}"
            )
        else:
            pubdate = int(pubdate.replace("-", ""))  # Convert YYYY-MM-DD to YYYYMMDD int

        external_ids = record.get("externalIds", None)
        if not external_ids:
            msg = f"Title: {title}. No 'externalIds' key found in Semantic Scholar's record for the given title. Response: {obj}"
            logger.error(msg)
            raise ACLProcessingError(msg)

        acl_id = external_ids.get("ACL", None)
        if not acl_id:
            msg = f"Title: {title}. No 'ACL' id found in external IDs for the given title. Response: {obj}"
            logger.error(msg)
            raise ACLProcessingError(msg)

        # Retrieve DOI; non-fatal if missing
        doi = external_ids.get("DOI", None)
        if not doi:
            msg = f"Title: {title}. No 'DOI' id found in external IDs for the given title. Response: {obj}"
            logger.warning(msg)

        return {"acl": acl_id, "doi": doi, "pubdate": pubdate}

    def acl_id_to_pdf(self, acl_id: str) -> str:
        url = f"https://aclanthology.org/{acl_id}.pdf"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            pdf = response.content
            save_path = os.path.join(self.PDF_SAVE_DIR, f"{acl_id}.pdf")
            with open(save_path, "wb") as f:
                f.write(pdf)
            return save_path

        logger.error(f"Failed to download PDF for ACL ID {acl_id}. Status code: {response.status_code}")

    def pdf_to_text(self, pdf_path: str) -> str:

        def block_is_text(block: tuple) -> bool:
            return block[6] == 0  # block type 0 indicates text

        def clean_block_text(text: str) -> str:
            """
            Cleans up the PDF formatting artifacts preserved in pymupdf 'blocks': newlines for line breaks and
            hyphenation (word division) at line ends.

            These are artifacts of PDF processing for paper; not necessary or helpful for digital texts or NLP.
            """
            # Replace hyphen at end of line followed by newline with nothing (rejoin hyphenated words)
            text = re.sub(r"-\s*\n\s*", "", text)

            # Replace remaining newlines with spaces
            text = re.sub(r"\n+", " ", text)

            # Clean up multiple spaces
            text = re.sub(r" +", " ", text)

            return text.strip()

        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                if block_is_text(block):
                    cleaned_text = clean_block_text(block[4])
                    text += cleaned_text + "\n"

        doc.close()
        return text.strip()

    def title_to_record(self, title: str) -> dict:
        """
        Returns: dict with keys
        'title', 'doi', 'pubdate', 'text' (full body of paper)
        """
        # Get data from Semantic Scholar
        paper_data = self.title_to_ids_and_pubdate(title)
        doi = paper_data["doi"]
        pubdate = paper_data["pubdate"]
        acl_id = paper_data["acl"]

        # Get the PDF from ACL Anthology
        pdf_path = self.acl_id_to_pdf(acl_id)
        if not pdf_path:
            raise ACLProcessingError(f"Failed to get PDF for title: {title}")
        text = self.pdf_to_text(pdf_path)

        return {"title": title, "doi": doi, "pubdate": pubdate, "text": text}

    def save_processed_research(self, record: dict) -> None:
        with open(os.path.join(self.PROCESSED_SAVE_DIR, "research.jsonl"), "a") as file:
            file.write(json.dumps(record) + "\n")

        with open(self.PROGRESS_PATH, "a") as file:
            file.write(record["title"] + "\n")

    def record_to_chunk_records(self, record: dict) -> list[dict]:
        """
        Takes a record who's 'text' is the full paper body (e.g. from title_to_record),
        chunks the body and returns a list of records, chunked & ready for db insertion
        """
        chunks = self.splitter.chunks(record["text"])
        return [
            {
                "title": record["title"],
                "doi": record["doi"],
                "pubdate": record["pubdate"],
                "citation_count": -1,
                "text": chunk,
            }
            for chunk in chunks
        ]


def main():
    # processor = ACLProcessor(API_KEY, data_path="context_dataset_train.csv")
    # processor.acl_dataset_to_jsonl()
    # processor.process_dataset()

    client = ACLClient(dataset="context_dataset.csv")
    client.acl_dataset_to_jsonl()


if __name__ == "__main__":
    main()
