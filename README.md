# Citeline

## Project Overview

Citeline is a research tool designed to help researchers automatically find and resolve relevant citations when writing academic papers. The system analyzes input sentences (and their context) to recommend appropriate citations from a large corpus of research papers.

The project focuses on astrophysics research, using approximately 50,000 research papers from the Astrophysics Data System (ADS) and 3,000 review papers as our dataset. Our approach leverages:

- Vector embeddings to capture semantic relationships between text
- PostgreSQL with the `pgvector` extension for efficient similarity search
- Advanced reranking and rank fusion techniques beyond basic vector distance
- LLM-based citation extraction for improved accuracy
- Custom text segmentation to optimize document chunking

## How It Works

1. We extract sentences and their citations from review papers as our ground truth
2. Research papers are chunked into semantically meaningful segments
3. These chunks are embedded into vector space
4. When a user inputs a sentence needing citations, we:
   - Embed the input sentence
   - Find the most semantically similar chunks using vector similarity
   - Apply additional metrics to refine the results
   - Return the most relevant citations

## Preprocessing

Our dataset is roughly 50,000 research papers from the Astrophysics Data System, and 3,000 'review' papers. Since review papers are written with great care and are intended to comprise the latest scientific knowledge on a particular subject, we use sentences and their in-line citations from review papers as 'ground truth'.

This means we extract sentences from the review papers, identify their inline citations, then strip the inline citations to form a training sample.

### Building training samples

Using the `pysbd` package we segment the review papers into sentences. Originally we used a regex to identify inline citations in sentences, however this proved to be too inaccurate due to the wide variety of placement, capitalization, and formatting seen in practice with these citations. Therefore we've moved to an LLM-based citation extraction method.

To develop the LLM-based workflow, we built a dataset of 100 example sentences and the ouptut (author, year) expected. To get LLM service on the FASRC cluster, we created an image based on the official Ollama docker image. This gives us a local, containerized endpoint for LLM service as well as access to multiple open-source LLMs of various size. After iterating over the prompt several times and testing various LLMs, we find `llama3.3` extracts inline citaitons with ~99% accuracy.

### Chunking the Research Papers

Originally, we chunked the research papers using the Python package `semantic_text_splitter`, which focuses on whitespace deliminations. Using a max length of 1500 chars and overlap of 150, this seemed to create chunks that were sometimes too long, other times too short to capture the true semantic content of a reference passage. In fact, the ideal length seemed to be the paragraph in the original paper. Unfortunately, the newline chars indicating new paragraphs are not present in the ADS dataset and must be inferred. Therefore we switched to a custom subclass of Langchain's `SemanticChunker`, that looks at changes in vector distances over sentences to determine where to split documents.

To find the optimal parameters for the `SemanticChunker`, we took 3 papers at random and identified the lengths (in chars) of their paragraphs. We then used the Python package `segeval` to get the boundary similarity between the true paragraph lengths and the chunks produced by `SemanticChunker`. In fact, we subclassed `SemanticChunker` to override its behaviors that a) it consumes whitespaces it splits on and b) it inserts spaces at joins (from a call to `" ".join`), which mutates the total length of all chunks compared to the original document. However, boundary similarity is only defined on integer lists that add up to the same total. So in order to get `SemanticChunker`'s results to be comparable to the reference results, we overrode those behaviors such that the total lengths of chunks will equal the length of the original document.

We are currently grid searching over multiple embedders, metrics, and other parameters to find the optimal arguments to create the chunker.

## Data Preparation and Preprocessing

- Data received muliple files, each containing a json array of objects; each object representing one record.
- Record properties include:
  - `doi`: a list of doi's for the paper; typically 1-2
  - `title`, `abstract`, `body`, etc.
- We use the 'research' files, plus the `doi_articles` and `salvaged_articles` to create `research.jsonl`
  - These files are the research papers that could be cited by an input sentence.
- In preprocessing these files, we:
  - rename the `doi` property to `dois` (still a list of strings)
  - add a `doi` property, the first doi in the list which we'll use as the paper's unique id going forward
  - add a `loaded_from` property to track which data file the record came from
  - remove duplicate records (tracked by having the same doi)
  - drop any records that don't have all the required keys (as noted in `REQUIRED_KEYS` constant)

## Database

One main table, '`lib`'
Rows contain id, doi, title, abstract, chunk (of body text)
This is denormalized a bit, to save on joins during query time

## Current Status

- Preprocessing code complete; loads raw json records and processes them into jsonl with body sentences segmented
- Implemented vector database with PostgreSQL and pgvector

## Next Steps

- Optimize chunking strategy for research papers
- Complete citation extraction from review papers
- Working on advanced reranking strategies
- Refactor code to support rank fusion (multiple similarity metrics)

## Technology Stack

- Python for data processing and model development
- PostgreSQL with pgvector for vector similarity search
- Langchain for document processing
- Open-source LLMs (via Ollama) for citation extraction
- Custom evaluation metrics for optimizing citation accuracy
