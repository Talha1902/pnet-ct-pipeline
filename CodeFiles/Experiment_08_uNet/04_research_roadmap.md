# Research Roadmap for Pancreas CT Thesis

## Phase 1: Understand the Dataset

### Tasks
- Inspect CT and mask files
- Confirm all patient scans are correctly aligned
- Verify image orientation and spacing
- Check which patients have pancreas masks and which do not
- Document excluded patients and reasons for exclusion

### Deliverable
Dataset audit summary with patient counts and quality checks

---

## Phase 2: Build Pancreas Segmentation Baseline

### Tasks
- preprocess CT images
- prepare mask labels
- create train/validation/test splits by patient
- train a baseline 2D or 2.5D U-Net
- evaluate pancreas segmentation using Dice and IoU

### Deliverable
A working pancreas segmentation model with validation metrics

---

## Phase 3: Improve the Model

### Tasks
- compare 2D vs 2.5D vs 3D segmentation
- test HU windowing and normalization
- tune negative sampling and data balancing
- evaluate early stopping and checkpointing

### Deliverable
Improved segmentation model report with ablation observations

---

## Phase 4: Tumor Detection within Pancreas ROI

### Tasks
- crop pancreas region from CT volumes
- create lesion-focused dataset if tumor masks exist
- train a model to localize pancreatic lesions
- evaluate sensitivity and false-positive rate

### Deliverable
Tumor localization pipeline with ROI-based detection results

---

## Phase 5: Tumor Grade Classification (Optional Advanced Extension)

### Tasks
- collect grade labels for each tumor case
- create lesion patches and ROI samples
- train classification model
- evaluate Grade 1 / Grade 2 / Grade 3 performance

### Deliverable
Grading model and confusion matrix

---

## Phase 6: Final Thesis Write-Up

### Tasks
- summarize methodology
- describe dataset, preprocessing, and models
- report quantitative results
- discuss limitations and future work
- justify clinical relevance

### Deliverable
Final thesis document with results and discussion

---

## Recommended Sequence

The most realistic plan is:
1. segmentation baseline,
2. segmentation improvement,
3. ROI extraction,
4. tumor detection,
5. tumor grading only if labels are available.

This sequence is scientifically sound and aligns with your current dataset and project stage.

---

## Practical Advice

Do not force the tumor grading task before the pancreas segmentation pipeline is strong. A robust segmentation model is the base that makes all later tumor-related work possible.

---

This roadmap provides a realistic progression from anatomy learning to abnormality detection.
