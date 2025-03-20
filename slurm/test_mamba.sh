#!/bin/bash
#
#SBATCH --job-name=test_mamba
#SBATCH -p test # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --mem 8000 # memory pool for all cores
#SBATCH -t 0-0:05 # time (D-HH:MM)
#SBATCH -o slurm.%x.%t.log # STDOUT
#SBATCH -e slurm.%x.%t.log # STDERR
module load python
# mamba activate citeline
# echo "which python (after activate): $(which python)"
mamba deactivate
mamba activate citeline
echo "which python (after re-activate): $(which python)"