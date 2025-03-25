#!/bin/bash
#
#SBATCH --job-name=create_bge_index
#SBATCH -p shared # partition (queue)
#SBATCH -c 48 # number of cores
#SBATCH --mem 48000 # memory pool for all cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR
module load python
mamba deactivate && mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull
python preprocessing.py --research
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"