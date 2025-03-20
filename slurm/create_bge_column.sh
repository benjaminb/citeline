#!/bin/bash
#
#SBATCH --job-name=create_bge_column
#SBATCH -p gpu # partition (queue)
#SBATCH -c 12 # number of cores
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-6:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%N.%j.out # STDOUT
#SBATCH -e slurm.%x.%N.%j.err # STDERR
module load python
mamba deactivate
mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/database
git pull
echo "starting python job..."
python database.py create-vector-table --table-name="library" --target-column-name chunk --embedder="BAAI/bge-small-en" --normalize --batch-size=256
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"