#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1   # This gives you the first GPU node (but you can specify which GPU you want
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G      # Considering reducing the resources you needed to reduce the queue time
#SBATCH --time=12:00:00

# Start Jupyter notebook
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0