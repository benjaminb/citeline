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
export TMPDIR=/n/holylabs/LABS/protopapas_lab/Lab/bbasseri/tmp
export OLLAMA_BASE_URL=http://localhost:11434
podman load -i /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/ollama_llama3.3.tar

# Change to neutral dir before running podman; podman automatically tries to mount the current dir as a volume?
cd /tmp
podman run -d \
  --name ollama-server \
  --log-level=debug \
  --rm \
  --device nvidia.com/gpu=all \
  -p 11434:11434 \
  --workdir /tmp \
#   --security-opt label=disable \
  ollamaserve

cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline

echo "Waiting for Ollama model to load..."
while true; do
    # Check if we can exec into the container and if ollama ps shows a loaded model
    if podman exec ollama-server ollama ps 2>/dev/null | grep -q "llama3.3:latest"; then
        echo "Model loaded successfully!"
        break
    else
        echo "Model not loaded yet, waiting 30 seconds..."
        sleep 30
    fi
done

python dataset_builder.py
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"