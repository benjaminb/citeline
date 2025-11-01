#!/bin/zsh
#
#SBATCH --job-name=test_running_milvus_with_podman
#SBATCH -p test # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --mem=16GB # memory pool for all cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%t.log # STDOUT
#SBATCH -e slurm.%x.%t.log # STDERR
source ~/.zshrc
# module load python
# mamba deactivate && mamba activate citeline
echo "which python: $(which python)"
echo "pwd: $(pwd)"
git pull
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"