#!/bin/bash
#
#SBATCH --job-name=test_podman
#SBATCH -p sapphire # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:0 # number of GPUs
#SBATCH --mem=16G # memory pool for all cores
#SBATCH -t 0-00:30 # time (D-HH:MM)
#SBATCH -o slurm.%x.%j.log # STDOUT
#SBATCH -e slurm.%x.%j.log # STDERR

module load python
mamba activate
mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/
git pull
