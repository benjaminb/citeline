#!/bin/zsh
#
#SBATCH --job-name=test_running_milvus_with_podman
#SBATCH -p gpu # partition (queue)
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH -c 2 # number of cores
#SBATCH --mem=16GB # memory pool for all cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
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



timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"