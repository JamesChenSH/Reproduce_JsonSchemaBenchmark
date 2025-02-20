#!/bin/bash
#SBATCH --account=def-six
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=4G               # memory per node
#SBATCH --time=02:00:00


module load gcc arrow/18.1.0 python/3.11
source /home/chens266/scratch/.venv/bin/activate

python ./runner.py  
deactivate                       