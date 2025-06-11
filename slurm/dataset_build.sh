#!/bin/bash
#
#SBATCH --job-name=dataset_build
#SBATCH -p gpu # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-00:30 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR
# module load python
# mamba deactivate && mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull

# Load container for Ollama service
# export OLLAMA_BASE_URL=http://localhost:11434
export TMPDIR=/tmp
podman load -i /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/ollama_llama3.3.tar
podman run -d --name ollama-server --log-level=debug --rm --userns=keep-id --device nvidia.com/gpu=all -p 11434:11434 ollamaserve


# Trigger the model to load
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.3:latest",
  "prompt": "What is water made of?"
}'

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

cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
python dataset_builder.py
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"
