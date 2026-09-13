#!/bin/bash
#
#SBATCH --job-name=run_experiments
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
if ! podman compose up -d; then
    echo "ERROR: podman compose up failed" >&2
    exit 1
fi

# Milvus standalone can take 30-90s to become ready; poll its health endpoint instead of a fixed sleep.
echo "Waiting for Milvus to become healthy..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:9091/healthz > /dev/null; then
        echo "Milvus is healthy (after ~$((i * 5))s)."
        break
    fi
    if [ "$i" = "60" ]; then
        echo "ERROR: Milvus not healthy after 300s" >&2
        podman ps -a >&2
        podman logs --tail 50 milvus-standalone >&2
        exit 1
    fi
    sleep 5
done

cd ../../../..

bash run_experiments.sh

echo "Experiments completed."

