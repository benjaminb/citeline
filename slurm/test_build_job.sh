#!/bin/bash
#
#SBATCH --job-name=test_milvus_launch
#SBATCH -p sapphire # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:0 # number of GPUs
#SBATCH --mem 32000 # memory pool for all cores
#SBATCH -t 0-00:30 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR

module load python
mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/
git pull

# Start Milvus service
podman compose -f database/milvus/docker-compose.yml up -d

cd database
python milvusdb.py --list-collections