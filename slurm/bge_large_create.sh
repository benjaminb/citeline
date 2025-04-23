#!/bin/bash
#
#SBATCH --job-name=astrobert_vector_column
#SBATCH -p gpu # partition (queue)
#SBATCH -c 12 # number of cores
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem=64GB # memory pool for all cores
#SBATCH -t 0-06:30 # time (D-HH:MM)
#SBATCH -o slurm.%x.%t.log # STDOUT
#SBATCH -e slurm.%x.%t.log # STDERR
module load python
mamba deactivate && mamba activate citeline
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull
./pg_start.sh
cd database
python database.py --create-vector-column --table-name="lib" --target-column chunk --embedder="BAAI/bge-large-en-v1.5" --batch-size=128
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"