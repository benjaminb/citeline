#!/bin/bash
#
#SBATCH --job-name=dataset_singularity_build
#SBATCH -p gpu_requeue # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --constraint="a100|h200"
#SBATCH --mem 96000 # memory pool for all cores
#SBATCH --time=3-00:00 # time (D-HH:MM)
#SBATCH --requeue
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR
#
# Runs dataset_builder.py against an Ollama server in a Singularity container.
# Requires the model store to be populated first:  bash slurm/ollama_pull_model.sh
# Replaces slurm/dataset_build.sh, which is broken by the podman/CDI version
# mismatch on the GPU nodes (see slurm/ollama_env.sh for the details).

module load python
mamba activate citeline
echo "which python:      $(which python)"
echo "which singularity: $(which singularity)"

# Must run from the repo root: citation_extraction.py:14-16 opens its prompt
# files by relative path, at import time.
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline || exit 1
git pull
source slurm/ollama_env.sh

# ---- Preflight ----

if [ ! -f "$SIF" ]; then
  echo "ERROR: image not found at $SIF"
  echo "Build it with: singularity build $SIF oci-archive:<path to ollama tar>"
  exit 1
fi

if [ ! -d "$MODEL_STORE" ] || [ -z "$(ls -A "$MODEL_STORE" 2>/dev/null)" ]; then
  echo "ERROR: model store $MODEL_STORE is missing or empty."
  echo "netscratch may have purged it. Run: bash slurm/ollama_pull_model.sh"
  exit 1
fi

echo "GPUs visible to this job:"
nvidia-smi -L || { echo "ERROR: no GPU available"; exit 1; }

# ---- Start Ollama ----

PORT=$(ollama_pick_port)
BASE_URL=http://127.0.0.1:$PORT
OLLAMA_LOG=ollama.${SLURM_JOB_ID:-$$}.log

echo "Starting Ollama on port $PORT (container log: $OLLAMA_LOG)"
ollama_start "$PORT" "$OLLAMA_LOG"

# gpu_requeue preempts with SIGTERM, so trap it as well as EXIT
trap ollama_stop EXIT TERM INT

# Two env vars are needed, and they are not interchangeable:
#   OLLAMA_BASE_URL - dataset_builder.py:15, liveness probe only
#   OLLAMA_HOST     - what actually routes inference. ChatOllama is constructed
#                     without base_url (citation_extraction.py:26-29), so the
#                     underlying ollama client reads OLLAMA_HOST. Setting only
#                     OLLAMA_BASE_URL passes the health check and then sends
#                     inference to the default port.
# Exported after ollama_start so the container's own --env wins for its listen
# address, which is a bind spec (0.0.0.0:PORT) rather than a URL.
export OLLAMA_BASE_URL=$BASE_URL
export OLLAMA_HOST=$BASE_URL

# Wait for the server AND for the model to be loadable, so a purged or partial
# store fails here instead of a thousand LLM calls later.
ollama_wait_ready "$BASE_URL" "$OLLAMA_PID" "$OLLAMA_LOG" "$MODEL_TAG" || exit 1

# ---- Python dataset builder ----

python dataset_builder.py
status=$?

timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp (dataset_builder.py exit $status)"
exit $status
