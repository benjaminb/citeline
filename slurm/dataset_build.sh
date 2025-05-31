#!/bin/bash
#
#SBATCH --job-name=dataset_build
#SBATCH -p gpu # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR
module load python
mamba deactivate && mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull
cd llm
export TMPDIR=/n/holylabs/LABS/protopapas_lab/Lab/bbasseri/tmp
mkdir -p $TMPDIR
podman load -i /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/ollama_llama3.3.tar
podman run --log-level=debug --rm --device nvidia.com/gpu=all -p 11434:11434 ollamaserve
cd ..
python dataset_builder.py
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"