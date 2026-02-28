#!/bin/bash
#
#SBATCH --job-name=dataset_build
#SBATCH -p gpu_requeue # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --constraint="a100|h200"
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH --time=3-00:00 # time (D-HH:MM)
#SBATCH --requeue
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
export OLLAMA_BASE_URL=http://localhost:11434
export PODMAN_ROOT=/tmp/$USER-podman-root
export PODMAN_RUNROOT=/tmp/$USER-podman-run
export LOCAL_SCRATCH=/tmp/$USER-podman

echo "Cleaning old podman state..."
rm -rf $PODMAN_ROOT $PODMAN_RUNROOT $LOCAL_SCRATCH

mkdir -p $PODMAN_ROOT
mkdir -p $PODMAN_RUNROOT
mkdir -p $LOCAL_SCRATCH

# Set up cleanup to trigger on exit / sigterm (preemption)
cleanup() {
  echo "Cleaning up containers..."
  podman rm -f $(podman ps -aq) 2>/dev/null || true
}
trap cleanup EXIT

# Load Ollama service
podman load -i /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/ollama_llama3.3.tar
echo "Images available:"
podman images
podman tag localhost/ollamaserve:latest ollamaserve:latest

# Use node-local scratch for container storage
export PODMAN_ROOT=/tmp/$USER-podman-root
export PODMAN_RUNROOT=/tmp/$USER-podman-run
mkdir -p $PODMAN_ROOT
mkdir -p $PODMAN_RUNROOT
podman run -d --log-level=debug --rm --device nvidia.com/gpu=all -p 11434:11434 ollamaserve:latest
echo "Containers available:"
podman container list
curl http://localhost:11434/api/generate -d '{"model": "llama3.3:latest", "prompt": "Respond with a single word that is the name of a fruit."}'

# Python dataset builder
python dataset_builder.py
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"
