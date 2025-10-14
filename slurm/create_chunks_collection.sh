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


cd src/citeline/database/milvus
podman compose up -d

sleep 60

cd ../../../..
python milvusdb.py --create-collection --name qwen06_chunks --data-source ../../../data/research_chunks.jsonl --embedder Qwen/Qwen3-Embedding-0.6B --normalize --batch-size 32
echo "Created Milvus collection qwen06_chunks"