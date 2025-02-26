#!/bin/bash
#SBATCH --account=def-six
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=16G               
#SBATCH --time=02:00:00


module load gcc arrow/18.1.0 python/3.11
module load rust/nightly

source /home/chens266/scratch/.venv/bin/activate

python ./runner.py  

deactivate                       