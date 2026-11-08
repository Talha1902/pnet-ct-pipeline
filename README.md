# pNET CT Pipeline

**Automated segmentation of the pancreas and pancreatic neuroendocrine tumor (pNET) candidates on contrast-enhanced CT, as the first stage of non-invasive tumor grading.**

The grade of a pNET (G1/G2/G3, driven by the Ki-67 index) guides treatment, but today it requires a biopsy. This project builds the imaging half of a pipeline that aims to predict the grade from a routine pretreatment CT. It covers finding the pancreas, locating tumor candidates, rating how far each automatic mask can be trusted, and handing the masks to a radiologist for confirmation before radiomics and classification.

This is an MSc thesis project in Artificial Intelligence at the University of Verona.

---

## Pipeline

```
 DICOM / NIfTI
      │
      ▼
 1. Input        choose the series, convert to NIfTI without resampling
 2. TotalSeg     organ ROI + contrast-phase estimate
 3. Checks       slice spacing, HU range, phase, pancreas found / not cut by scan edge
 4. DiNTS        pretrained MONAI bundle  ─┐
 5. nnU-Net      Task007 3d_fullres, 5 folds + TTA ─┤
 6. Clean-up     anatomy-guided post-processing (both models)
                 confidence level from model agreement
 7. Package      start mask + reference layers for review in 3D Slicer
      │
      ▼
 Radiologist confirms / corrects  →  auto-vs-expert comparison
```

The pipeline is frozen in [`Task_17C_Run_Pipeline.py`](CodeFiles/Phase2/Experiment_17_RadiologistPackage/Task_17C_Run_Pipeline.py). Every step is cached, so an interrupted run continues where it stopped.

## Key findings (38-patient cohort)

- **Two models, one confidence level.** DiNTS and nnU-Net agree with a median Dice of 0.70 on the pancreas region. Their agreement is used to rate each case HIGH, MEDIUM, LOW_TUMOR or LOW_ORGAN.
- **Tumors are found, but sizes vary.** Both models find the same lesion in 27 of 38 cases, but their tumor masks disagree on size (median tumor Dice 0.28). Tumor ROIs therefore need expert correction.
- **Slice thickness matters.** At 3 mm or thicker, tumor masks degrade sharply (median Dice 0.89 at about 1.6 mm, 0.54 at 5 mm). Thin reconstructions (1.5 mm or less) are recommended for new data.
- **Contrast phase.** The cohort is mostly arterial phase, while DiNTS was trained on portal-venous scans. Image-based phase estimation replaces the DICOM phase tag, which is unreliable.
- **Failures are documented.** Four cases fail with both models. Rescue strategies (padding, cropping, test-time augmentation) were tested and reported as negative results.

## Repository structure

```
CodeFiles/
├── Experiment_*                       Phase 1: 2D / 2.5D U-Net trained from scratch (baseline)
└── Phase2/
    ├── Experiment_10Audit             DICOM audit, baseline-scan selection, DICOM → NIfTI
    ├── Experiment_11_MONAI_DiNTS      Pretrained DiNTS inference and audits
    ├── Experiment_11B_Batchinference_All38   Batch inference + automatic cohort QC
    ├── Experiment_12_CaseTable        Per-case table, contrast-phase estimation
    ├── Experiment_13_MaskCleanup      Mask post-processing guided by TotalSegmentator
    ├── Experiment_14_RescueFailures   Padding / cropping / TTA rescue strategies
    ├── Experiment_15_SliceThickness   Synthetic slice-thickening robustness study
    ├── Experiment_16_SecondModel      nnU-Net as a second model; agreement → confidence
    ├── Experiment_17_RadiologistPackage  3D Slicer review tool, expert comparison, frozen pipeline
    └── Report_Figures                 Figures for progress reports
server.py   Run a command on the HPC over SSH
sync.py     Mirror results from the HPC to a local backup
```

## Usage

GPU jobs run on a SLURM cluster:

```bash
# process a folder of new scans (one <case>.nii.gz or <case>/ DICOM folder per case)
sbatch run_17C.sbatch --input /path/to/cases --run batch1

# reproduce the validated cohort results for selected cases
sbatch run_17C.sbatch --validate PAN_09 PAN_14
```

To run commands on the HPC with `server.py` / `sync.py`, copy `.env.example` to `.env` and fill in your own server details.

## Tech stack

Python 3.11 · PyTorch 2.6 (CUDA 12.4) · MONAI 1.5 · nnU-Net · TotalSegmentator 2.18 · SimpleITK · nibabel · pandas · 3D Slicer · SLURM

## Data

This repository uses the public TCIA collection [CTpred-Sunitinib-panNET](https://www.cancerimagingarchive.net/) (38 patients). **No imaging or clinical data is included in this repository.** Download the data from TCIA under its own license.

## Status

This is active research. The segmentation pipeline is frozen and validated. Radiologist review, radiomics and grade classification (G1 vs G2/G3) are the next stages.

## Author

[Talha1902](https://github.com/Talha1902): MSc Artificial Intelligence, University of Verona. Supervisor: Prof. Bogdan Maris.
