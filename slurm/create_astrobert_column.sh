#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH -c 12 # number of cores
#SBATCH --gres=gpu:2 # number of GPUs
#SBATCH --mem 64000 # memory pool for all cores
#SBATCH -t 0-6:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
module load python
mamba activate citeline
cd /n/holylabs/LABS/protopapas_lab/Lab/bbasseri/citeline/database
git pull
python database.py --create-vector-column --table-name="library" --target-column-name chunk --embedder="adsabs/astroBERT" --batch-size=256