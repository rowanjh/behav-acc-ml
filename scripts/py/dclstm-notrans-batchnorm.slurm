#!/bin/bash
#SBATCH --job-name=dcLSTM-batchnorm-notrans        # job name
#SBATCH -N 1			 # node count
#SBATCH -n 8                     # CPU count
#SBATCH -p gpu			 # set job for GPU node
#SBATCH -G rtx5000:1		 # number of GPUs
#SBATCH -t 14:00:00              # time allocation
#SBATCH --mem 16G                 # total memory
#SBATCH -o out-dcLSTM-batchnorm-notrans
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rjacques@orn.mpg.de

module purge
module load anaconda3
source ~/.bashrc
conda activate dclstm
python --version
python -u scripts/py/main.py --batchnorm --notrans
