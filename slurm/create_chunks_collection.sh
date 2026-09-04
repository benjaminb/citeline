#!/bin/bash
#
#SBATCH --job-name=create_chunks_collection
#SBATCH -p gpu_h200 # partition (queue)
#SBATCH -c 12 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem 48000 # memory pool for all cores
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR


# Keep model downloads off the 100GB home quota. The models in the loop below total ~75GB.
# holylabs rather than netscratch, since netscratch is purged periodically.
export HF_HOME=/n/holylabs/LABS/protopapas_lab/Lab/bbasseri/hf_cache
mkdir -p "$HF_HOME"

cd src/citeline/database/milvus
podman compose up -d

sleep 10

cd ..

# collection name, embedder name, batch size
tuples=(
    # "astrobert,adsabs/astroBERT,32"
    "astrollama,UniverseTBD/astrollama,16"
    "astrosage,AstroMLab/AstroSage-8B,8"
    "nasa,nasa-impact/nasa-ibm-st.38m,32"
    "bge,BAAI/bge-large-en-v1.5,32"
    "qwen_8b,Qwen/Qwen3-Embedding-8B,4"
)

for item in "${tuples[@]}"; do
    # Temporarily set IFS to a comma to split the string
    IFS="," read -r collection embedder batch_size <<< "$item"
    
    echo "Collection: $collection, Embedder: $embedder, Batch size: $batch_size"
    python milvusdb.py --create-collection --name "${collection}" --data-source ../../../data/research_chunks.jsonl --embedder "$embedder" --normalize --batch-size "$batch_size"
    echo "Created Milvus collection ${collection}"
done

echo "All collections created successfully."