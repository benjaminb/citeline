#!/bin/bash
#
#SBATCH --job-name=test_podman
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH --time=0-00:30 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR

module load python
mamba activate citeline
echo "which python: $(which python)"
echo "which podman: $(which podman)"
echo "PATH: $PATH"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull

# Load container for Ollama service
# env > slurm_env.txt
export OLLAMA_BASE_URL=http://localhost:11434
export TMPDIR=/n/holylabs/LABS/protopapas_lab/Lab/bbasseri/tmp
podman load -i /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/ollama_llama3.3.tar
echo "Images available:"
podman images
sleep 30
echo "After 30 seconds..."
podman run -d --log-level=debug --rm --device nvidia.com/gpu=all -p 11434:11434 ollamaserve:latest
echo "Containers available:"
podman container list
# curl http://localhost:11434/api/generate -d '{"model": "llama3.3:latest", "prompt": "Respond with a single word that is the name of a fruit."}'
# python dataset_builder.py
# timestamp=$(date +"%Y%m%d_%H%M%S")
# echo "ended at: $timestamp"
