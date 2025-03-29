#!/bin/bash
#
#SBATCH --job-name=bert_index
#SBATCH -p sapphire # partition (queue)
#SBATCH -c 64 # number of cores
#SBATCH --mem=270GB # memory pool for all cores
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR
module load python
mamba deactivate && mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull
./pg_start.sh
python database.py --create-index --table-name="lib" --target-column bert --index-type hnsw --m 32 --ef-construction 1024
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"