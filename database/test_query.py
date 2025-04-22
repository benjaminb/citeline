from database import Database
from time import time

start = time()
db = Database()
db.test_connection()
print(f"Connection took {time() - start:.3f} seconds")

start = time()
results = db.query("SELECT id, doi, chunk FROM lib;")
print(f"Got {len(results)} results")
print(f"Query took {time() - start:.3f} seconds")
