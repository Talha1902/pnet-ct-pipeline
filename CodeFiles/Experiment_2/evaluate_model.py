"""
Experiment 02: Quantitative evaluation of an unseen patient's pancreas segmentation.

This script compares:
1. Original CT volume
2. TotalSegmentator-generated reference mask
3. U-Net predicted mask

It calculates whole-volume metrics, per-slice metrics, checks slice ranges,
saves visual comparisons, and exports false-positive / false-negative masks.

Required packages:
    pip install nibabel numpy matplotlib

Run:
    python evaluate_patient06.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# =========================================================
# 1. EDIT ONLY THESE PATHS
# =========================================================
PATIENT_DIR = Path(r"D:/Thesis Folder/Training_dataset/Pat_06")

CT_PATH = PATIENT_DIR / "Origional_images.nii"
REFERENCE_MASK_PATH = PATIENT_DIR / "masked_images.nii"
PREDICTION_MASK_PATH = PATIENT_DIR / "UNET_AI_Pancreas_Prediction.nii"

OUTPUT_DIR = PATIENT_DIR / "Evaluation_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Your test script already used 0.5.
PREDICTION_THRESHOLD = 0.5


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def load_nifti(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI file and return both the object and floating-point data."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    nifti = nib.load(str(path))
    data = nifti.get_fdata(dtype=np.float32)
    return nifti, data


def binarize(data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert an array to a binary uint8 mask."""
    return (data > threshold).astype(np.uint8)


def calculate_binary_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float | int]:
    """Calculate segmentation metrics from two binary arrays."""
    prediction = prediction.astype(bool)
    reference = reference.astype(bool)

    tp = int(np.logical_and(prediction, reference).sum())
    fp = int(np.logical_and(prediction, np.logical_not(reference)).sum())
    fn = int(np.logical_and(np.logical_not(prediction), reference).sum())
    tn = int(np.logical_and(np.logical_not(prediction), np.logical_not(reference)).sum())

    eps = 1e-8

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tn_voxels": tn,
        "reference_voxels": int(reference.sum()),
        "predicted_voxels": int(prediction.sum()),
    }


def positive_slice_indices(mask: np.ndarray) -> np.ndarray:
    """Return axial slice indices containing at least one foreground voxel."""
    foreground_per_slice = np.sum(mask, axis=(0, 1))
    return np.where(foreground_per_slice > 0)[0]


def safe_slice_range(indices: np.ndarray) -> str:
    """Return a readable range for positive slices."""
    if len(indices) == 0:
        return "None"
    return f"{int(indices.min())}-{int(indices.max())}"


def save_metrics_csv(metrics: dict[str, float | int], output_path: Path) -> None:
    """Save whole-volume metrics in a two-column CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def save_per_slice_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    output_path: Path,
) -> None:
    """Calculate and save metrics for every axial slice."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "slice_index",
            "reference_pixels",
            "predicted_pixels",
            "dice",
            "iou",
            "precision",
            "recall",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for slice_idx in range(reference.shape[2]):
            ref_slice = reference[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]

            slice_metrics = calculate_binary_metrics(pred_slice, ref_slice)

            # Empty-reference and empty-prediction slices are not informative for Dice.
            both_empty = ref_slice.sum() == 0 and pred_slice.sum() == 0

            writer.writerow(
                {
                    "slice_index": slice_idx,
                    "reference_pixels": int(ref_slice.sum()),
                    "predicted_pixels": int(pred_slice.sum()),
                    "dice": "" if both_empty else f"{slice_metrics['dice']:.6f}",
                    "iou": "" if both_empty else f"{slice_metrics['iou']:.6f}",
                    "precision": "" if both_empty else f"{slice_metrics['precision']:.6f}",
                    "recall": "" if both_empty else f"{slice_metrics['recall']:.6f}",
                }
            )


# =========================================================
# 3. LOAD CT, REFERENCE MASK, AND PREDICTION
# =========================================================
print("Loading files...")

ct_nifti, ct_volume = load_nifti(CT_PATH)
reference_nifti, reference_raw = load_nifti(REFERENCE_MASK_PATH)
prediction_nifti, prediction_raw = load_nifti(PREDICTION_MASK_PATH)

reference_mask = binarize(reference_raw, threshold=0.0)
prediction_mask = binarize(prediction_raw, threshold=PREDICTION_THRESHOLD)

print(f"CT shape:         {ct_volume.shape}")
print(f"Reference shape:  {reference_mask.shape}")
print(f"Prediction shape: {prediction_mask.shape}")


# =========================================================
# 4. VERIFY GEOMETRY
# =========================================================
if ct_volume.shape != reference_mask.shape:
    raise ValueError(
        "CT and reference-mask shapes do not match:\n"
        f"CT={ct_volume.shape}, reference={reference_mask.shape}"
    )

if ct_volume.shape != prediction_mask.shape:
    raise ValueError(
        "CT and prediction-mask shapes do not match:\n"
        f"CT={ct_volume.shape}, prediction={prediction_mask.shape}"
    )

ct_reference_affine_match = np.allclose(
    ct_nifti.affine,
    reference_nifti.affine,
    atol=1e-4,
)

ct_prediction_affine_match = np.allclose(
    ct_nifti.affine,
    prediction_nifti.affine,
    atol=1e-4,
)

print(f"CT/reference affine match:  {ct_reference_affine_match}")
print(f"CT/prediction affine match: {ct_prediction_affine_match}")

if not ct_reference_affine_match:
    print(
        "WARNING: CT and reference affine matrices differ. "
        "Review their alignment in 3D Slicer before trusting the metrics."
    )

if not ct_prediction_affine_match:
    print(
        "WARNING: CT and prediction affine matrices differ. "
        "Review their alignment in 3D Slicer before trusting the metrics."
    )


# =========================================================
# 5. CALCULATE WHOLE-VOLUME METRICS
# =========================================================
metrics = calculate_binary_metrics(prediction_mask, reference_mask)

reference_positive_slices = positive_slice_indices(reference_mask)
prediction_positive_slices = positive_slice_indices(prediction_mask)

false_positive_slice_indices = np.setdiff1d(
    prediction_positive_slices,
    reference_positive_slices,
)

missed_reference_slice_indices = np.setdiff1d(
    reference_positive_slices,
    prediction_positive_slices,
)

metrics.update(
    {
        "reference_positive_slice_count": int(len(reference_positive_slices)),
        "prediction_positive_slice_count": int(len(prediction_positive_slices)),
        "reference_positive_slice_range": safe_slice_range(reference_positive_slices),
        "prediction_positive_slice_range": safe_slice_range(prediction_positive_slices),
        "false_positive_slice_count": int(len(false_positive_slice_indices)),
        "missed_reference_slice_count": int(len(missed_reference_slice_indices)),
        "ct_reference_affine_match": ct_reference_affine_match,
        "ct_prediction_affine_match": ct_prediction_affine_match,
    }
)

print("\n" + "=" * 55)
print("PATIENT 06: WHOLE-VOLUME SEGMENTATION RESULTS")
print("=" * 55)
print(f"Dice coefficient: {metrics['dice']:.4f}")
print(f"IoU:              {metrics['iou']:.4f}")
print(f"Precision:        {metrics['precision']:.4f}")
print(f"Recall:           {metrics['recall']:.4f}")
print(f"Specificity:      {metrics['specificity']:.4f}")
print("-" * 55)
print(f"Reference voxels: {metrics['reference_voxels']}")
print(f"Predicted voxels: {metrics['predicted_voxels']}")
print(f"True positives:   {metrics['tp_voxels']}")
print(f"False positives:  {metrics['fp_voxels']}")
print(f"False negatives:  {metrics['fn_voxels']}")
print("-" * 55)
print(
    "Reference positive slices: "
    f"{metrics['reference_positive_slice_count']} "
    f"(range {metrics['reference_positive_slice_range']})"
)
print(
    "Predicted positive slices: "
    f"{metrics['prediction_positive_slice_count']} "
    f"(range {metrics['prediction_positive_slice_range']})"
)
print(f"False-positive slices: {metrics['false_positive_slice_count']}")
print(f"Missed reference slices: {metrics['missed_reference_slice_count']}")
print("=" * 55)


# =========================================================
# 6. SAVE METRICS
# =========================================================
metrics_csv_path = OUTPUT_DIR / "patient06_volume_metrics.csv"
per_slice_csv_path = OUTPUT_DIR / "patient06_per_slice_metrics.csv"

save_metrics_csv(metrics, metrics_csv_path)
save_per_slice_metrics(prediction_mask, reference_mask, per_slice_csv_path)

print(f"\nSaved volume metrics to:\n{metrics_csv_path}")
print(f"Saved per-slice metrics to:\n{per_slice_csv_path}")


# =========================================================
# 7. EXPORT ERROR MASKS FOR 3D SLICER
# =========================================================
false_positive_mask = np.logical_and(
    prediction_mask == 1,
    reference_mask == 0,
).astype(np.uint8)

false_negative_mask = np.logical_and(
    prediction_mask == 0,
    reference_mask == 1,
).astype(np.uint8)

true_positive_mask = np.logical_and(
    prediction_mask == 1,
    reference_mask == 1,
).astype(np.uint8)

fp_nifti = nib.Nifti1Image(
    false_positive_mask,
    ct_nifti.affine,
    ct_nifti.header,
)
fn_nifti = nib.Nifti1Image(
    false_negative_mask,
    ct_nifti.affine,
    ct_nifti.header,
)
tp_nifti = nib.Nifti1Image(
    true_positive_mask,
    ct_nifti.affine,
    ct_nifti.header,
)

fp_output_path = OUTPUT_DIR / "patient06_false_positive_mask.nii.gz"
fn_output_path = OUTPUT_DIR / "patient06_false_negative_mask.nii.gz"
tp_output_path = OUTPUT_DIR / "patient06_true_positive_mask.nii.gz"

nib.save(fp_nifti, str(fp_output_path))
nib.save(fn_nifti, str(fn_output_path))
nib.save(tp_nifti, str(tp_output_path))

print(f"Saved false-positive mask to:\n{fp_output_path}")
print(f"Saved false-negative mask to:\n{fn_output_path}")
print(f"Saved true-positive mask to:\n{tp_output_path}")


# =========================================================
# 8. CREATE AN OBJECTIVE COMPARISON FIGURE
# =========================================================
if reference_mask.sum() == 0:
    raise ValueError("Reference mask is empty; cannot select a comparison slice.")

# Select the slice with the largest reference-mask area.
comparison_slice = int(
    np.argmax(np.sum(reference_mask, axis=(0, 1)))
)

ct_slice = ct_volume[:, :, comparison_slice]
reference_slice = reference_mask[:, :, comparison_slice]
prediction_slice = prediction_mask[:, :, comparison_slice]

# Combined error coding:
# 0 = background
# 1 = correct overlap
# 2 = false positive
# 3 = false negative
error_map = np.zeros_like(reference_slice, dtype=np.uint8)
error_map[(reference_slice == 1) & (prediction_slice == 1)] = 1
error_map[(reference_slice == 0) & (prediction_slice == 1)] = 2
error_map[(reference_slice == 1) & (prediction_slice == 0)] = 3

reference_overlay = np.ma.masked_where(reference_slice == 0, reference_slice)
prediction_overlay = np.ma.masked_where(prediction_slice == 0, prediction_slice)
error_overlay = np.ma.masked_where(error_map == 0, error_map)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(ct_slice, cmap="gray")
axes[0].set_title(f"Original CT\nSlice {comparison_slice}")

axes[1].imshow(ct_slice, cmap="gray")
axes[1].imshow(reference_overlay, alpha=0.55)
axes[1].set_title("Reference Mask")

axes[2].imshow(ct_slice, cmap="gray")
axes[2].imshow(prediction_overlay, alpha=0.55)
axes[2].set_title("U-Net Prediction")

axes[3].imshow(ct_slice, cmap="gray")
axes[3].imshow(error_overlay, alpha=0.65)
axes[3].set_title(
    "Overlap / Error Map\n"
    "1=overlap, 2=false positive, 3=false negative"
)

for axis in axes:
    axis.axis("off")

fig.suptitle(
    "Patient 06 Quantitative Segmentation Evaluation\n"
    f"Dice={metrics['dice']:.4f}, "
    f"IoU={metrics['iou']:.4f}, "
    f"Precision={metrics['precision']:.4f}, "
    f"Recall={metrics['recall']:.4f}",
    fontsize=12,
)

plt.tight_layout()

figure_output_path = OUTPUT_DIR / "patient06_quantitative_comparison.png"
plt.savefig(figure_output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved comparison figure to:\n{figure_output_path}")


# =========================================================
# 9. FINAL MESSAGE
# =========================================================
print("\nEvaluation completed successfully.")
print(
    "Important: these metrics compare the U-Net prediction with the "
    "TotalSegmentator-generated reference mask. They are not yet equivalent "
    "to evaluation against a radiologist-validated clinical ground truth."
)