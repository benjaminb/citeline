#!/bin/bash
#
#SBATCH -p sapphire # partition (queue)
#SBATCH -c 32 # number of cores
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
module load python
mamba activate citeline
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/database
git pull
python database.py --create-base-table --table-name library --from-path="../data/preprocessed/research.jsonl"