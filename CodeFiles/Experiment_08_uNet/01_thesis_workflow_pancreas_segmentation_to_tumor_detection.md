# Pancreas Segmentation to Tumor Detection Workflow

## 1. Research Goal

This thesis focuses on building a medical imaging pipeline for pancreatic analysis in abdominal CT scans.

The proposed workflow is:
1. Learn the normal anatomy of the pancreas.
2. Segment the pancreas from abdominal CT scans.
3. Crop the pancreas region as a region of interest (ROI).
4. Detect or localize abnormal pancreatic lesions inside that ROI.
5. If labels are available, classify tumor grade (Grade 1, Grade 2, Grade 3).

This is a strong strategy because direct tumor detection on the full abdomen is difficult, while pancreas segmentation creates a smaller and more relevant region for lesion analysis.

---

## 2. Why This Approach Is Good

The pancreas is a small organ, and its position and shape vary between patients. Training a model directly on whole CT volumes often causes false positives and weak localization.

A pancreas segmentation model learns:
- the anatomical location of the pancreas,
- surrounding intensity patterns,
- organ boundaries,
- the spatial context of the pancreatic region.

Then the downstream tumor detection model can focus only on the pancreas ROI, where suspicious abnormalities are more likely to appear.

This approach is standard in medical image analysis: first organ segmentation, then lesion detection.

---

## 3. Proposed Pipeline

### Stage 1: Pancreas Segmentation

Input: abdominal CT volume

Output: pancreas mask

Possible models:
- 2D U-Net
- 2.5D U-Net (like your current Experiment 08 work)
- 3D U-Net for volumetric context

Use your own dataset created in 3D Slicer with:
- original CT images
- pancreas masks

This stage is the foundation of the project.

### Stage 2: ROI Extraction

Once the pancreas is segmented:
- crop a bounding box around the pancreas mask,
- optionally add a margin around the organ,
- keep only the pancreas region for the next stage.

This removes unrelated abdominal organs and reduces noise.

### Stage 3: Tumor Detection or Lesion Localization

Input: pancreas ROI

Output: tumorous region or lesion mask

Possible models:
- U-Net for lesion segmentation
- 2.5D model for lesion detection
- 3D model if dataset size and GPU permit

This stage learns to identify abnormal structures inside the pancreas ROI.

### Stage 4: Tumor Grading (Optional Extension)

If pathology-confirmed grade labels exist:
- Grade 1
- Grade 2
- Grade 3

Then use the lesion ROI or the cropped pancreas ROI to train a classifier.

Possible classifier inputs:
- lesion image patch,
- pancreas crop,
- radiomics features,
- deep CNN features.

---

## 4. Dataset Requirements

For segmentation:
- CT NIfTI images
- pancreas masks
- consistent preprocessing
- patient-level split

For tumor detection:
- tumor masks or lesion labels
- tumor presence annotation

For grading:
- pathology grade labels
- enough annotated cases per grade

Without lesion labels and grade labels, the project can still be completed as a pancreas segmentation and lesion localization study.

---

## 5. Preprocessing

Recommended preprocessing:
- orientation standardization
- clip Hounsfield units to a useful range, e.g. [-100, 250]
- normalize intensity
- keep original 512x512 slices if possible
- avoid unnecessary resampling unless required

Maintain consistent train/validation/test splits at patient level.

---

## 6. Evaluation Metrics

For pancreas segmentation:
- Dice coefficient
- IoU
- precision
- recall
- Hausdorff distance (optional)

For tumor detection:
- lesion Dice
- sensitivity
- precision
- F1-score
- patient-wise detection accuracy

For grading:
- accuracy
- macro-F1
- confusion matrix
- sensitivity for each grade

---

## 7. Experimental Plan

### Experiment A: Pancreas Segmentation Baseline
- Train a U-Net on CT pancreas masks
- Validate on patient-level splits
- Compare 2D and 2.5D approaches

### Experiment B: Pancreas ROIs + Tumor Localization
- crop pancreas region
- detect tumor-like regions
- evaluate lesion detection performance

### Experiment C: Tumor Grading (if labels available)
- classify Grade 1 vs Grade 2 vs Grade 3
- use lesion patches or pancreas crops

---

## 8. Expected Thesis Outcome

A realistic thesis outcome is a pipeline that can:
- segment pancreas from CT scans,
- localize abnormal regions in the pancreas,
- provide a foundation for future pancreatic tumor detection and grading work.

This is a meaningful and publishable medical imaging research direction.

---

## 9. Recommendation

Because the current dataset is focused on pancreas anatomy and segmentation, the best immediate thesis route is:

Pancreas segmentation and lesion localization in abdominal CT

Then, if additional labeled tumor data becomes available, extend to G1/G2/G3 classification.

---

## 10. Final Thesis Direction

Recommended thesis title idea:

"Automated Pancreas Segmentation and Tumor Localization in Abdominal CT Images Using Deep Learning"

Alternative title:

"Deep Learning-Based Pancreas Segmentation for Early Tumor Detection in Abdominal CT Scans"

---

This workflow is a strong and realistic approach for a medical imaging thesis using your current dataset and future extension toward tumor detection.
