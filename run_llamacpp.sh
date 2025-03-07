#!/bin/bash
#SBATCH --account=def-six
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=16G               
#SBATCH --time=02:00:00


module load gcc arrow/18.1.0 python/3.11
module load rust/nightly

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

python ./runner.py --wrapper llamacpp

deactivate                       