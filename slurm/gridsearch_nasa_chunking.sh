#!/bin/bash
#
#SBATCH --job-name=gridsearch_bert_chunking
#SBATCH -p gpu # partition (queue)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem=16GB # memory pool for all cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH -o slurm.%x.%t.log # STDOUT
#SBATCH -e slurm.%x.%t.log # STDERR
module load python
mamba deactivate && mamba activate citeline
echo "which python: $(which python)"
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull
cd chunking
python chunking_grid_search.py --model_name="bert-base-uncased"
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"