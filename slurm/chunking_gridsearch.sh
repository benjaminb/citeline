#!/bin/zsh
#
#SBATCH --job-name=test_gpu_with_constraint
#SBATCH -p gpu_requeue # partition (queue)
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --constraint="a100|h200"
#SBATCH -c 24 # number of cores
#SBATCH --mem=64GB # memory pool for all cores
#SBATCH -t 1-01:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%t.log # STDOUT
#SBATCH -e slurm.%x.%t.log # STDERR

echo "which python: $(which python)"
echo "pwd: $(pwd)"
git pull

# Start milvus container. (pwd starts out in slurm/ directory)
cd ../src/citeline/database/milvus
podman compose up -d
sleep 30
cd ..
python milvusdb.py --list-collections
echo "Milvus service started."

cd ../../.. # back up to project root
cd chunking

echo "Starting text splitter gridsearch..."
python text_splitter_gridsearch.py

echo "Text splitter gridsearch completed."
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"