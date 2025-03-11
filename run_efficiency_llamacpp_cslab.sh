#!/bin/bash
#SBATCH --partition=gpunodes 
#SBATCH -c 24
#SBATCH --mem=40G 
#SBATCH --gres=gpu:rtx_a6000:1 
#SBATCH -t 1-0


source /w/284/jameschen/.venv/bin/activate

# export PATH="/usr/local/cuda/bin:${PATH}"
# CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python==0.3.2

python ./runner_efficiency.py --wrapper llamacpp

deactivate                       