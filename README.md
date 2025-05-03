# Citeline

The goal of this research project is to provide a tool to researchers to help resolve relevant citations when writing research papers. We are currently building a model that takes a sentence (plus additional context, possibly) as input, and returns the citations needed for that sentence.

Primarily the model consists of a vector database implemented in PostgreSQL with the `pgvector` extension, plus a framework for using additional metrics besides vector distance to rerank / fuse ranks.

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

One main table, 'library'
Rows contain id, doi, title, abstract, chunk (of body text)
This is denormalized a bit, to save on joins during query time
