#!/bin/bash
#SBATCH --job-name=exp16_nnunet
#SBATCH --partition=gpuq
#SBATCH --nodelist=gpunode002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

# nnU-Net v1 pretrained Task007_Pancreas, 3d_fullres, 5-fold ensemble with the default mirroring TTA.
# Usage (from this folder):  sbatch run_16.sbatch            -> all 38 patients, then the comparison
#                            sbatch run_16.sbatch PAN_34      -> one patient (test)
THESIS=~/Thesis_Folder
OUT=$THESIS/Results/_Phase2_Experiment16_SecondModel
INPUT=$OUT/nnunet_input
PRED=$OUT/nnunet_predictions

# nnU-Net v1 looks for trained models in $RESULTS_FOLDER/nnUNet/<config>/<task>
mkdir -p $THESIS/Models/nnUNet_v1_results
ln -sfn $THESIS/Models/nnUNet_v1_Task007 $THESIS/Models/nnUNet_v1_results/nnUNet
export RESULTS_FOLDER=$THESIS/Models/nnUNet_v1_results
export nnUNet_raw_data_base=$OUT/nnunet_raw_unused
export nnUNet_preprocessed=$OUT/nnunet_preprocessed_unused
# torch 2.6 defaults to weights_only loading, which the v1 checkpoints do not support
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# input = one symlink per CT, named <case>_0000.nii.gz as nnU-Net expects
PATIENTS=${@:-$(seq -f "PAN_%02g" 1 38)}
mkdir -p $INPUT $PRED
# only symlinks are cleared, so a test run does not leave other patients in the input
find $INPUT -maxdepth 1 -type l -name 'PAN_*_0000.nii.gz' -delete
for P in $PATIENTS; do
    ln -sfn $THESIS/Results/_Phase2_Experiment10D_Baseline_NIfTI/$P/baseline_ct.nii.gz $INPUT/${P}_0000.nii.gz
done

source ~/venvs/nnunetv1/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
date
nnUNet_predict -i $INPUT -o $PRED -t Task007_Pancreas -m 3d_fullres --num_threads_preprocessing 4 --num_threads_nifti_save 2
date
deactivate

# comparison with DiNTS uses the main environment
source ~/venvs/thesis/bin/activate
python -u Task_16_Compare_Models.py $@
