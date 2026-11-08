#!/bin/bash
#SBATCH --job-name=exp14_rescue
#SBATCH --partition=gpuq
#SBATCH --nodelist=gpunode002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

# gpunode002 = Tesla P40 (24 GB). gpunode001 has Tesla K80s, which PyTorch 2.6 does not support.
# Usage (from this folder):  sbatch run_14.sbatch            -> the 8 patients in PATIENTS
#                            sbatch run_14.sbatch PAN_34      -> one patient (test)

# venv built from the system /usr/bin/python3.11; the Python/3.11.2 module has no working SSL
source ~/venvs/thesis/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())"

python -u Task_14_Rescue_Failures.py "$@"
