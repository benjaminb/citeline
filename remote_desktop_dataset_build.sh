#!/bin/bash

module load python
mamba activate citeline
echo "which python: $(which python)"
echo "which podman: $(which podman)"
echo "PATH: $PATH"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull

# Load Ollama service
podman load -i /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/ollama_llama3.3.tar
echo "Images available:"
podman images

podman run -d --log-level=debug --rm --device nvidia.com/gpu=all -p 11434:11434 ollamaserve:latest
echo "Containers available:"
podman container list
curl http://localhost:11434/api/generate -d '{"model": "llama3.3:latest", "prompt": "Respond with a single word that is the name of a fruit."}'

# Python dataset builder
python dataset_builder.py
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"
