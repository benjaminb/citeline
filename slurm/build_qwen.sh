#!/bin/bash
#
#SBATCH --job-name=build_qwen
#SBATCH -p gpu_h200 # partition (queue)
#SBATCH -c 4 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-10:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR

module load python
mamba activate
mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/
git pull

# Start Milvus service
cd database/milvus
podman compose up -d
podman ps

cd ..
python milvusdb.py --create-collection --name qwen8b_chunks_redo \
--data-source ../data/research_chunks.jsonl \
--embedder Qwen/Qwen3-Embedding-8B \
--normalize \
--batch-size 16 

python milvusdb.py --create-collection --name qwen8b_contributions \
--data-source ../data/research_contributions.jsonl \
--embedder Qwen/Qwen3-Embedding-8B \
--normalize \
--batch-size 16 