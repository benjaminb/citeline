python database.py create-collection \
                   --path=./data/processed_for_chroma/reviews/Astro_Reviews.json \
                   --model='BAAI/bge-small-en' \
                   --metric='cosine' \
                   --augment-fn='no_augmentation'

python database.py insert-records \
                     --into='test-bge-small-en__cosine__no_augmentation' \
                     --path=./data/processed_for_chroma/reviews/Astro_Reviews.json \

python database.py create-collection \
                   --model='bert-base-uncased' \
                   --metric='cosine' \
                   --augment-fn='no_augmentation'

python database.py insert-records \
                     --into='test-bert-base-uncased__cosine__no_augmentation' \
                     --path=./data/processed_for_chroma/reviews/Astro_Reviews.json 

python database.py create-collection \
                   --model='nvidia/NV-Embed-v2' \
                   --metric='cosine' \
                   --augment-fn='no_augmentation'

python database.py insert-records \
                     --into='test-NV-Embed-v2__cosine__no_augmentation' \
                     --path=./data/processed_for_chroma/reviews/Astro_Reviews.json 