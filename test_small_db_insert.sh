python database.py delete-collection --name=test-bge-small-en__cosine
python database.py create-collection --model=BAAI/bge-small-en --metric=cosine
python database.py insert-records --source=data/json/Astro_Reviews.json --into=test-bge-small-en__cosine