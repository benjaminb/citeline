#!/bin/bash
#
#SBATCH --job-name=bert_vector_column
#SBATCH -p gpu # partition (queue)
#SBATCH -c 4 # number of cores
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-03:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%t.log # STDOUT
#SBATCH -e slurm.%x.%t.log # STDERR
module load python
mamba deactivate && mamba activate citeline
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/database
git pull
python database.py --create-vector-column --table-name="lib" --target-column chunk --embedder="bert-base-uncased" --batch-size=64
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"