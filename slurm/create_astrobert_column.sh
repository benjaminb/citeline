#!/bin/bash
#
#SBATCH --job-name=bert_bge_vector_columns
#SBATCH -p gpu # partition (queue)
#SBATCH -c 4 # number of cores
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem=64GB # memory pool for all cores
#SBATCH -t 0-03:30 # time (D-HH:MM)
#SBATCH -o slurm.%x.%t.log # STDOUT
#SBATCH -e slurm.%x.%t.log # STDERR
module load python
mamba deactivate && mamba activate citeline
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline
git pull
./pg_start.sh
python database.py --create-vector-column --table-name="lib" --target-column-name chunk --embedder="adsabs/astroBERT" --batch-size=128
timestamp=$(date +"%Y%m%d_%H%M%S")
echo "ended at: $timestamp"