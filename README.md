# Citeline

## Project Overview

Citeline is a research tool designed to help researchers automatically find and resolve relevant citations when writing academic papers. The system analyzes input sentences (and their context) to recommend appropriate citations from a large corpus of research papers.

The project focuses on astrophysics research, using approximately 50,000 research papers from the Astrophysics Data System (ADS) and 3,000 review papers as our dataset. Our approach leverages:

- Vector embeddings to capture semantic relationships between text
- Milvus DB for efficient similarity search
- Advanced reranking and rank fusion techniques
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

After searching over various chunking configurations, we found TextSplitter with capacity 1500 chars and overlap of 150 seemed to produce the most semantically meaningful chunks.

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

We use Milvus as our vector database. See `database/milvusdb.py` for the implementation.

## Current Status

* `Qwen/Qwen3-Embedding-8B` is the top-performing embedding model. Second is the much smaller and faster `Qwen/Qwen3-Embedding-0.6B`. 
* Between document representation 'chunks' and 'contributions', chunks tends to perform better although more effort can be put into constructing the contributions (prompt engineering or other methods)
* The query expansion `add_prev_3` (adding the previous 3 sentences to the query) improves results in most cases
* Interleaving chunks and contributions is the top performing approach so far (for top `k` retrieved results, take `k/2` from chunks and `k/2` from contributions, then interleave their rankings)

## Next Steps
* 'All but the top' process on vectors (remove the top $k$ principle components from vectors)
* Embed into database `summary(chunk)`, and `chunk + summary(chunk)` to see if this improves results
* Experiment with other rerankers / rank fusion strategies (cross encoders, NLI models, etc)
* Experiment with difference vectors between query and reference (computing the average difference vector between query and target, then adding back this average difference to query embedding during search)


## Technology Stack
- Docker / Podman for standalone Milvus DB
- HuggingFace, SentenceTransformers, and PyTorch for embeddings
- Pandas for various stages of data processing