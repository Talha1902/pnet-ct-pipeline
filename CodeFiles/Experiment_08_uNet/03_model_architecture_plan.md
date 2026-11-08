# Model Architecture Plan for Pancreas Detection Research

## 1. Stage 1: Pancreas Segmentation Model

### Recommended Architecture
- U-Net (baseline)
- 2.5D U-Net (good balance for CT slices)
- 3D U-Net (if enough GPU memory and data)

### Why U-Net?
U-Net is widely used in medical image segmentation because it combines:
- low-level spatial details,
- high-level semantic features,
- strong performance on small anatomical structures.

### Suggested Input
- CT slice or three-slice stack (2.5D)
- Hounsfield unit windowing, e.g. [-100, 250]
- normalized intensity

### Suggested Output
- binary pancreas mask for each CT slice

### Loss Function
- Dice + BCE loss
- DiceCELoss is a good choice for this type of segmentation task

### Optimizer
- Adam
- learning rate: 1e-3

### Training Strategy
- patient-level split
- positive and negative slices balanced
- validation on held-out patients
- early stopping based on validation Dice

---

## 2. Stage 2: ROI-Based Tumor Detection Model

After pancreas segmentation, create a pancreas ROI by cropping around the mask. This reduces the search space and helps the model focus on suspicious regions.

### Proposed Pipeline
1. Segment pancreas
2. Crop pancreas bounding box with margin
3. Feed ROI into lesion detection network
4. Produce probability map or lesion mask

### Model Options
- U-Net for lesion segmentation
- 2.5D model for lesion localization
- 3D CNN if volumetric lesion context is required

### Advantages of ROI-Based Detection
- less background noise
- better focus on the organ
- lower false positives
- better training stability

---

## 3. Stage 3: Tumor Grading Model (Optional)

If pathology labels for Grade 1, Grade 2, and Grade 3 are available:

### Input
- lesion patch
- pancreas crop
- tumor ROI + surrounding tissue

### Model Options
- CNN classifier
- 3D CNN for volumetric grading
- radiomics + machine learning classifier
- hybrid CNN + handcrafted feature model

### Output
- Grade 1
- Grade 2
- Grade 3

### Challenges
- small number of tumor cases per class
- class imbalance
- weak annotation quality
- limited data for each grade

---

## 4. Recommended Implementation Path

### Phase 1: Pancreas Segmentation
This is the most realistic immediate target because your current project and dataset already support it.

### Phase 2: Lesion Localization
Use the pancreas mask to isolate suspicious pixels or regions

### Phase 3: Grading Extension
Only after obtaining enough tumor-labeled data

---

## 5. Suggested Training Settings

### Segmentation Settings
- input: 2.5D three-slice CT stack
- network: U-Net with 3 input channels
- loss: Dice + BCE
- optimizer: Adam
- learning rate: 0.001
- batch size: 8
- early stopping: yes

### Clinical Data Strategy
- train/validation/test split by patient
- avoid leakage across patients
- keep final test set untouched until evaluation

---

## 6. Evaluation Strategy

### Pancreas Segmentation
- Dice
- IoU
- Precision
- Recall

### Lesion Detection
- lesion Dice
- sensitivity
- false positive rate
- detection accuracy

### Grading
- accuracy
- macro-F1
- confusion matrix
- per-class sensitivity

---

## 7. Final Architecture Recommendation

For this thesis, the most practical architecture is:

1. 2.5D U-Net for pancreas segmentation
2. pancreas ROI extraction
3. lesion detection U-Net or CNN inside the ROI
4. optional grade classification network if tumor labels are available

This is a realistic and research-worthy path that matches your current project stage.

---

## 8. Thesis-Friendly Summary

The best architecture for your current thesis is not to jump directly into tumor grade classification. Instead, use a staged pipeline:

pancreas segmentation -> ROI extraction -> tumor detection -> (optional) tumor grading

This improves robustness and matches a proper medical imaging research workflow.
