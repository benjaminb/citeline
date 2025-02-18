import psycopg2
from semantic_text_splitter import TextSplitter
from utils import load_dataset

# iterate over all records
# for each record, chunk and insert
