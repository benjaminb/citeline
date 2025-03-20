#!/bin/bash
#
#SBATCH --job-name=create_bge_column
#SBATCH -p gpu # partition (queue)
#SBATCH -c 12 # number of cores
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-6:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.%x.out # STDOUT
#SBATCH -e slurm.%N.%j.%x.err # STDERR
module load python
mamba deactivate
mamba activate citeline
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/database
git pull
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "create_bge_column.sh: $timestamp"
python database.py create-vector-table --table-name="library" --target-column-name chunk --embedder="BAAI/bge-small-en" --normalize --batch-size=256
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"