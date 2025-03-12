#!/bin/bash
#
#SBATCH -p sapphire # partition (queue)
#SBATCH -c 48 # number of cores
#SBATCH --mem 128000 # MB memory pool for all cores
#SBATCH -t 0-06:30 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
module load python
mamba activate citeline
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/database
git pull
python database.py create-index --table-name="bge" --index-type='hnsw' --metric=vector_cosine_ops --m=32 --ef-construction=512