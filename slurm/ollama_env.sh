#!/bin/bash
#
# Shared configuration and helpers for the Singularity-based Ollama service.
# Sourced by slurm/ollama_pull_model.sh and slurm/dataset_singularity_build.sh.
# Not meant to be executed directly.
#
# Why Singularity and not podman: the FASRC GPU nodes ship CDI specs at
# cdiVersion 0.7.0 (/etc/cdi/nvidia.yaml, /var/run/cdi/nvidia.yaml) while the
# installed podman is 4.9.4, which reads at most 0.6.0. It rejects both specs,
# registers zero devices, and fails with "unresolvable CDI devices
# nvidia.com/gpu=all". Singularity's --nv binds the driver libraries directly
# and never consults CDI. See slurm/dataset_build.sh for the old podman version.

REPO=/n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline

# Server-only image: contains the Ollama 0.9.0 binary and NO models.
# Built once with (note oci-archive, not docker-archive):
#   singularity build $SIF oci-archive:/n/holylabs/.../ollama_llama3.3.tar
SIF=/n/holylabs/LABS/protopapas_lab/Lab/bbasseri/ollamaserve.sif

# Persistent model store, bind-mounted to /models inside the container.
# NOTE: netscratch is purged periodically. If the model disappears, re-run
#   bash slurm/ollama_pull_model.sh
# Verify this path is right for your account:
#   ls -d /n/netscratch/protopapas_lab /n/holyscratch01/protopapas_lab
MODEL_STORE=/n/netscratch/protopapas_lab/Lab/bbasseri/ollama_models

# Must match MODEL_NAME in src/citeline/llm/citation_extraction.py:12
MODEL_TAG="llama3.3:latest"


# Echo a free port for the Ollama server. Singularity shares the host network
# (there is no -p publishing), so two jobs on one node would otherwise collide.
ollama_pick_port() {
  local port=11434
  if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    port=$((11435 + ${SLURM_JOB_ID:-$$} % 1000))
    echo "Port 11434 is in use; falling back to $port" >&2
  fi
  echo "$port"
}

# ollama_start <port> <logfile> -- starts the server in the background and sets
# OLLAMA_PID. --nv is used only when a GPU is actually present, so this same
# function works on a CPU node for the model download.
ollama_start() {
  local port=$1 log=$2
  local nv=()
  if nvidia-smi -L > /dev/null 2>&1; then
    nv=(--nv)
  else
    echo "No GPU detected; starting Ollama without --nv (CPU only)"
  fi

  singularity run "${nv[@]}" \
    -B "$MODEL_STORE":/models \
    --env OLLAMA_MODELS=/models \
    --env OLLAMA_HOST="0.0.0.0:$port" \
    "$SIF" > "$log" 2>&1 &
  OLLAMA_PID=$!
}

# ollama_wait_ready <base_url> <pid> <logfile> [required_model_tag]
# Polls /api/tags for up to 180s. Returns non-zero if the server dies or never
# becomes ready, printing the container log so the failure is visible in the
# slurm output rather than surfacing later as connection errors from Python.
ollama_wait_ready() {
  local url=$1 pid=$2 log=$3 want=${4:-}
  local i tags
  for i in $(seq 1 90); do
    tags=$(curl -sf "$url/api/tags" 2>/dev/null)
    if [ -n "$tags" ]; then
      if [ -z "$want" ] || printf '%s' "$tags" | grep -qF "$want"; then
        echo "Ollama ready after $((i * 2))s"
        return 0
      fi
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: Ollama exited during startup. Last 50 lines of $log:"
      tail -50 "$log"
      return 1
    fi
    sleep 2
  done

  if [ -n "$want" ]; then
    echo "ERROR: Ollama is up but '$want' never appeared in /api/tags."
    echo "Model store $MODEL_STORE may be empty or purged; run slurm/ollama_pull_model.sh"
  else
    echo "ERROR: Ollama never became ready after 180s."
  fi
  echo "Last 50 lines of $log:"
  tail -50 "$log"
  return 1
}

# ollama_stop -- terminate the background server, if one is running.
ollama_stop() {
  if [ -n "${OLLAMA_PID:-}" ]; then
    echo "Shutting down Ollama (pid $OLLAMA_PID)..."
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
}
