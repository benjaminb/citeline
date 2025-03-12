#!/bin/bash
#
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -c 20 # number of cores
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-3:30 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
module load python
mamba activate citeline
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/database
git pull
python database.py create-vector-table --table-name="bge" --embedder="BAAI/bge-small-en" --batch-size=16