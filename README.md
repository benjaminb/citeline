# Citeline

## Data Preparation and Preprocessing

* Data received muliple files, each containing a json array of objects; each object representing one record.
* Record properties include:
    * `doi`: a list of doi's for the paper; typically 1-2
    * `title`, `abstract`, `body`, etc.
* We use the 'research' files, plus the `doi_articles` and `salvaged_articles` to create `research.jsonl`
    * These files are the research papers that could be cited by an input sentence.
* In preprocessing these files, we:
    * rename the `doi` property to `dois` (still a list of strings)
    * add a `doi` property, the first doi in the list which we'll use as the paper's unique id going forward
    * add a `loaded_from` property to track which data file the record came from
    * remove duplicate records (tracked by having the same doi)
    * drop any records that don't have all the required keys (as noted in `REQUIRED_KEYS` constant)
