#!/bin/bash
#
#SBATCH --job-name=build_qwen
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -c 24 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-00:45 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR

module load python
mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/
git pull

# Start Milvus service
cd database/milvus
podman compose up -d

cd ..
python milvusdb.py --create-collection --name foo --data-source ../data/research_chunks.jsonl --embedder=Qwen/Qwen3-Embedding-8B