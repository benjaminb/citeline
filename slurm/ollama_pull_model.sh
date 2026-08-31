#!/bin/bash
#
#SBATCH --job-name=ollama_pull_model
#SBATCH -p shared
#SBATCH -c 2
#SBATCH --mem 8000
#SBATCH --time=0-04:00
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR
#
# One-time download of the LLM into the persistent model store. Needs no GPU.
#
#   bash slurm/ollama_pull_model.sh      # interactively on any node with internet
#   sbatch slurm/ollama_pull_model.sh    # or as a short job
#
# Re-run this whenever netscratch purges the store. The dataset job checks for
# the model up front and points back here if it is missing.
#
# Downloads roughly 42GB and takes 30+ minutes. It also doubles as a validation
# of the run configuration: it uses the same Singularity flags as the real job.

cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline || exit 1
source slurm/ollama_env.sh

echo "SIF:         $SIF"
echo "Model store: $MODEL_STORE"
echo "Model tag:   $MODEL_TAG"

# ---- Preflight ----

if [ ! -f "$SIF" ]; then
  echo "ERROR: image not found at $SIF"
  echo "Build it with: singularity build $SIF oci-archive:<path to ollama tar>"
  exit 1
fi

if ! mkdir -p "$MODEL_STORE" 2>/dev/null; then
  echo "ERROR: cannot create $MODEL_STORE"
  echo "Check the scratch path: ls -d /n/netscratch/protopapas_lab /n/holyscratch01/protopapas_lab"
  exit 1
fi

echo "Free space on the model store filesystem:"
df -h "$MODEL_STORE"
avail_gb=$(df -BG --output=avail "$MODEL_STORE" | tail -1 | tr -dc '0-9')
if [ -n "$avail_gb" ] && [ "$avail_gb" -lt 50 ]; then
  echo "ERROR: only ${avail_gb}GB free; the model needs ~42GB (50GB recommended)"
  exit 1
fi

if ! curl -sI --max-time 15 https://registry.ollama.ai/ > /dev/null 2>&1; then
  echo "ERROR: cannot reach https://registry.ollama.ai/ from this node."
  echo "Try a login node, or check whether an http_proxy is required here."
  exit 1
fi

# ---- Start a temporary server and pull ----

PORT=$(ollama_pick_port)
BASE_URL=http://127.0.0.1:$PORT
PULL_LOG=ollama_pull.${SLURM_JOB_ID:-$$}.log

echo "Starting Ollama on port $PORT (container log: $PULL_LOG)"
ollama_start "$PORT" "$PULL_LOG"
trap ollama_stop EXIT TERM INT

ollama_wait_ready "$BASE_URL" "$OLLAMA_PID" "$PULL_LOG" || exit 1

echo "Pulling $MODEL_TAG ..."
# /api/pull streams one JSON line per progress update, which would be tens of
# thousands of lines in a slurm log. Keep every status change and sample the
# rest so the log still shows forward progress.
curl -sN "$BASE_URL/api/pull" -d "{\"name\": \"$MODEL_TAG\"}" \
  | awk '!/"completed"/ || NR % 500 == 0 { print; fflush() }'

# ---- Verify ----

echo "Models now in the store:"
tags=$(curl -sf "$BASE_URL/api/tags")
echo "$tags"

if ! printf '%s' "$tags" | grep -qF "$MODEL_TAG"; then
  echo "ERROR: $MODEL_TAG is still not present after the pull."
  echo "Last 50 lines of $PULL_LOG:"
  tail -50 "$PULL_LOG"
  exit 1
fi

echo "Store size:"
du -sh "$MODEL_STORE"
echo "Done. $MODEL_TAG is available at $MODEL_STORE"
