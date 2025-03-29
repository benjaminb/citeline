#!/bin/bash
#
#SBATCH --job-name=astrobert_index
#SBATCH -p shared # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --mem 8000 # memory pool for all cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR
module load python
mamba deactivate && mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull
./pg_start.sh
cd database
python database.py --create-index --table-name="lib" --target-column astrobert --index-type hnsw --m 32 --ef-construction 512
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"