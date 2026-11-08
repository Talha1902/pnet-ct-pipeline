#!/bin/bash
#SBATCH --job-name=exp12B_phase
#SBATCH --partition=gpuq
#SBATCH --nodelist=gpunode002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

# gpunode002 = Tesla P40 (24 GB). gpunode001 has Tesla K80s, which PyTorch 2.6 does not support.
# Usage (from this folder):  sbatch run_12B.sbatch            -> all 38 patients
#                            sbatch run_12B.sbatch PAN_34      -> one patient (test)

# venv built from the system /usr/bin/python3.11; the Python/3.11.2 module has no working SSL
source ~/venvs/thesis/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())"

python -u Task_12B_Contrast_Phase.py "$@"
