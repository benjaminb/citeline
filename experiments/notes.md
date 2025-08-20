# Experimental Notes & Journal

### Baseline

#### 6.17.25

Having chunked and embedded the research corpus using bge-small, bge-large, and astrobert, I set top-k to 1000 and performed vector search using 326 nontrivial examples from the training set. 

Each example query and search result produces a Jaccard score (intersection over union) and a 'hit percentage' (proportion of target documents retrieved by the search, i.e. recall).


| Model | Average IoU | Recall |
|-------|-------------|--------|
|`astrobert`|0.0004|0.22|
|`bge-small`|0.0015|0.71|
|`bge-large`|0.0019|0.77|

*Using `k = 1000`, cosine similarity on document chunks, IVFFlat index with 1000 probes*

### Document Expansion: representing papers by a list of their contributions

#### 6.17.25
With vanilla RAG and these embedding models it appears that we need a high `k` in order to retrieve the target documents, but more documents retrieved lowers average IoU. One way to address this would be to use document expansion by representing each paper as a list of strings comprising that paper's original contributions.

As a pilot attempt, I took 10 training examples (`small_train.jsonl`) which together cited 15 research papers. Those 15 papers I fed into DeepSeek-R1 prompting it to write out the paper's original contributions. This resulted in 154 individual 'contribution strings' which I embedded using `bge-large` (the best performing model in vanilla RAG), and inserted into a table on the database. 

Performing top-20 vector similarity on this table resulted in an average IoU of 0.1860 and recall of 0.94. Part of this is likely due to the much smaller search space. However the 'original contribution' extraction also may serve to focus each paper's underlying citation value and I should try this method at a larger scale.

| Model | Average IoU | Recall |
|-------|-------------|--------|
|`bge-large`|0.0600|0.68|

#### 8.20.25

Wrote `clean_dataset_refs.ipynb` to remove `[REF]` from datasets' inputs; this is not a token in any model vocabulary, so it's likely tokenizing to `'[', 'ref', ']'` which just adds noise. Re-running baseline experiments against the cleaned dataset.