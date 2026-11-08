#!/bin/bash
#SBATCH --job-name=exp16_download
#SBATCH --partition=gpuq
#SBATCH --nodelist=gpunode002
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# Pretrained nnU-Net v1 model for MSD Task07 Pancreas (Isensee et al.), 5.0 GB.
# The login node cannot reach zenodo.org; gpunode002 can. No GPU is requested.
MODEL_DIR=~/Thesis_Folder/Models/nnUNet_v1_Task007
mkdir -p "$MODEL_DIR" && cd "$MODEL_DIR"

wget -c -nv -O Task007_Pancreas.zip "https://zenodo.org/record/4003545/files/Task007_Pancreas.zip?download=1"
ls -l Task007_Pancreas.zip
unzip -q -o Task007_Pancreas.zip -d .
find . -maxdepth 5 -type d | head -40
