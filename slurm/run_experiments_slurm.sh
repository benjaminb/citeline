#!/bin/bash
#
#SBATCH --job-name=create_chunks_collection
#SBATCH -p gpu_h200 # partition (queue)
#SBATCH -c 12 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem 48000 # memory pool for all c ores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR


# Keep model downloads off the 100GB home quota. The models in the loop below total ~75GB.
# holylabs rather than netscratch, since netscratch is purged periodically.
export HF_HOME=/n/holylabs/LABS/protopapas_lab/Lab/bbasseri/hf_cache
mkdir -p "$HF_HOME"

cd src/citeline/database/milvus
podman compose up -d

sleep 10

cd ../../../..

bash run_experiments.sh

echo "Experiments completed."

