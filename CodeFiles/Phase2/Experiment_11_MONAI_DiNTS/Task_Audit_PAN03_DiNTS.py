from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# PHASE 2 — EXPERIMENT 11A
# PAN_03 MONAI DiNTS OUTPUT AUDIT
# ============================================================


CT_PATH = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder"
    r"\Results"
    r"\_Phase2_Experiment10D_Baseline_NIfTI"
    r"\PAN_03"
    r"\baseline_ct.nii.gz"
)


PRED_PATH = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder"
    r"\Results"
    r"\_Phase2_Experiment11A_MONAI_DiNTS"
    r"\PAN_03"
    r"\baseline_ct"
    r"\baseline_ct_trans.nii.gz"
)


print("=" * 80)
print("PHASE 2 — EXPERIMENT 11A")
print("PAN_03 DiNTS PREDICTION AUDIT")
print("=" * 80)


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

if not CT_PATH.exists():
    raise FileNotFoundError(
        f"CT not found:\n{CT_PATH}"
    )


if not PRED_PATH.exists():
    raise FileNotFoundError(
        f"Prediction not found:\n{PRED_PATH}"
    )


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

ct_img = nib.load(
    str(CT_PATH)
)

pred_img = nib.load(
    str(PRED_PATH)
)


ct = ct_img.get_fdata()

pred = pred_img.get_fdata()


# ------------------------------------------------------------
# BASIC GEOMETRY
# ------------------------------------------------------------

print("\n--- GEOMETRY ---")

print(
    "CT shape:",
    ct.shape
)

print(
    "Prediction shape:",
    pred.shape
)


shape_match = (
    ct.shape
    ==
    pred.shape
)


affine_match = np.allclose(
    ct_img.affine,
    pred_img.affine,
    atol=1e-4
)


print(
    "Shape match:",
    shape_match
)

print(
    "Affine match:",
    affine_match
)


print(
    "CT spacing:",
    ct_img.header.get_zooms()[:3]
)

print(
    "Prediction spacing:",
    pred_img.header.get_zooms()[:3]
)


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

unique_labels = np.unique(
    pred
)


print("\n--- LABELS ---")

print(
    "Unique prediction values:",
    unique_labels
)


# MONAI labels:
# 0 = background
# 1 = pancreas
# 2 = pancreatic tumor candidate


pancreas_mask = (
    pred == 1
)

tumor_mask = (
    pred == 2
)


pancreas_voxels = int(
    pancreas_mask.sum()
)

tumor_voxels = int(
    tumor_mask.sum()
)


print(
    "Pancreas voxels (label 1):",
    pancreas_voxels
)

print(
    "Tumor-candidate voxels (label 2):",
    tumor_voxels
)


# ------------------------------------------------------------
# PHYSICAL VOLUME
# ------------------------------------------------------------

spacing = np.array(
    pred_img.header.get_zooms()[:3]
)


voxel_volume_mm3 = np.prod(
    spacing
)


pancreas_volume_ml = (
    pancreas_voxels
    * voxel_volume_mm3
    / 1000.0
)


tumor_volume_ml = (
    tumor_voxels
    * voxel_volume_mm3
    / 1000.0
)


print("\n--- VOLUME ---")

print(
    "Voxel volume (mm3):",
    voxel_volume_mm3
)


print(
    "Predicted pancreas volume (mL):",
    round(
        pancreas_volume_ml,
        3
    )
)


print(
    "Predicted tumor-candidate volume (mL):",
    round(
        tumor_volume_ml,
        3
    )
)


# ------------------------------------------------------------
# AXIAL SLICE COVERAGE
# ------------------------------------------------------------

def slice_range(mask):

    positive_slices = np.where(
        np.any(
            mask,
            axis=(0, 1)
        )
    )[0]


    if len(
        positive_slices
    ) == 0:

        return (
            0,
            None,
            None
        )


    return (
        len(
            positive_slices
        ),
        int(
            positive_slices.min()
        ),
        int(
            positive_slices.max()
        )
    )


pancreas_count, pancreas_first, pancreas_last = (
    slice_range(
        pancreas_mask
    )
)


tumor_count, tumor_first, tumor_last = (
    slice_range(
        tumor_mask
    )
)


print("\n--- AXIAL COVERAGE ---")


print(
    "Pancreas-positive slices:",
    pancreas_count
)

print(
    "Pancreas slice range:",
    pancreas_first,
    "to",
    pancreas_last
)


print(
    "Tumor-positive slices:",
    tumor_count
)

print(
    "Tumor slice range:",
    tumor_first,
    "to",
    tumor_last
)


# ------------------------------------------------------------
# RELATION BETWEEN TUMOR AND PANCREAS
# ------------------------------------------------------------

foreground_mask = (
    pancreas_mask
    |
    tumor_mask
)


foreground_voxels = int(
    foreground_mask.sum()
)


print("\n--- TOTAL FOREGROUND ---")

print(
    "Pancreas + tumor voxels:",
    foreground_voxels
)


# ------------------------------------------------------------
# INTENSITY INFORMATION
# ------------------------------------------------------------

print("\n--- CT ---")


print(
    "CT minimum:",
    float(
        np.min(ct)
    )
)

print(
    "CT maximum:",
    float(
        np.max(ct)
    )
)


# ------------------------------------------------------------
# FINAL CHECK
# ------------------------------------------------------------

print("\n" + "=" * 80)


if shape_match and affine_match:

    print(
        "GEOMETRY CHECK: PASS"
    )

else:

    print(
        "GEOMETRY CHECK: REVIEW REQUIRED"
    )


if pancreas_voxels > 0:

    print(
        "PANCREAS LABEL: PRESENT"
    )

else:

    print(
        "PANCREAS LABEL: NOT FOUND"
    )


if tumor_voxels > 0:

    print(
        "TUMOR-CANDIDATE LABEL: PRESENT"
    )

else:

    print(
        "TUMOR-CANDIDATE LABEL: NOT FOUND"
    )


print("=" * 80)