#!/bin/bash
#SBATCH --partition=gpunodes 
#SBATCH -c 4 
#SBATCH --mem=30G 
#SBATCH --gres=gpu:rtx_4090:1 
#SBATCH -t 1-0


source /w/284/jameschen/.venv/bin/activate

python ./runner_efficiency.py --wrapper outlines

deactivate                       