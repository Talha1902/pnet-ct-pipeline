#!/bin/bash
#SBATCH --job-name=exp17C_pipeline
#SBATCH --partition=gpuq
#SBATCH --nodelist=gpunode002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

# gpunode002 = Tesla P40 (24 GB). gpunode001 has Tesla K80s, which PyTorch 2.6 does not support.
# About 15 min per case (nnU-Net 5-fold ensemble is most of it).
# Usage (from this folder):  sbatch run_17C.sbatch --validate PAN_09 PAN_14
#                            sbatch run_17C.sbatch --input /path/to/cases --run hospital_batch1

# venv built from the system /usr/bin/python3.11; nnU-Net v1 is called from its own venv
source ~/venvs/thesis/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())"

python -u Task_17C_Run_Pipeline.py "$@"
