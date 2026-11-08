""""
Experiment 05B - Validation visualization for Pat_11 and Pat_12

Purpose
-------
Load the BEST Experiment 05B U-Net checkpoint and evaluate the two validation
patients (Pat_11 and Pat_12) without retraining.

For each patient this script:
1. Loads the original CT and TotalSegmentator-generated reference mask
2. Applies the SAME CT preprocessing used during training
3. Runs slice-wise 2D U-Net inference over the complete 3D volume
4. Reconstructs a 3D predicted pancreas mask
5. Calculates whole-volume Dice, IoU, Precision, and Recall
6. Saves the 3D prediction as NIfTI for 3D Slicer
7. Saves TP / FP / FN masks for error inspection in 3D Slicer
8. Saves per-slice metrics
9. Creates three comparison figures:
   - Largest reference pancreas slice
   - Best Dice slice
   - Worst pancreas-containing slice
10. Creates one combined summary CSV for Pat_11 and Pat_12

IMPORTANT
---------
This script is for VALIDATION analysis only.
Do not use Pat_13, Pat_14, or Pat_15 here if they are your final held-out test set.

Install if needed:
    pip install torch monai nibabel numpy matplotlib

Run:
    python validate_pat11_pat12_05B.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import UNet


# ============================================================
# 1. EDIT THESE PATHS IF NEEDED
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")

BEST_MODEL_PATH = Path(
    r"D:/Thesis Folder/Results/Experiment_05B_results/"
    r"Experiment_05B_uNET_2D_Improved_without_Pat_02_08/"
    r"Experiment_05B_best_pancreas_unet_v2.pth"
)

VALIDATION_PATIENTS = ["Pat_11", "Pat_12"]

OUTPUT_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_05B_results/"
    r"Experiment_05B_uNET_2D_Improved_without_Pat_02_08/"
    r"Validation_Visualization"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MUST MATCH EXPERIMENT 05B TRAINING SETTINGS
# ============================================================

HU_MIN = -100.0
HU_MAX = 250.0
PREDICTION_THRESHOLD = 0.5
INFERENCE_BATCH_SIZE = 8


# ============================================================
# 3. FILE HELPERS
# ============================================================

def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def find_patient_file(patient_dir: Path, kind: str) -> Path:
    """
    Robust file lookup for your existing folder naming conventions.
    """
    files = [
        p for p in patient_dir.iterdir()
        if p.is_file() and is_nifti(p)
    ]

    if kind == "ct":
        preferred = [
            "origional_images.nii",
            "origional_images.nii.gz",
            "original_images.nii",
            "original_images.nii.gz",
        ]

        for candidate in preferred:
            for p in files:
                if p.name.lower() == candidate:
                    return p

        for p in files:
            name = p.name.lower()
            if (
                ("origional" in name or "original" in name)
                and "mask" not in name
                and "prediction" not in name
            ):
                return p

    elif kind == "mask":
        preferred = [
            "masked_images.nii",
            "masked_images.nii.gz",
            "mask_images.nii",
            "mask_images.nii.gz",
        ]

        for candidate in preferred:
            for p in files:
                if p.name.lower() == candidate:
                    return p

        for p in files:
            name = p.name.lower()
            if (
                ("masked_images" in name or "mask_images" in name)
                and "prediction" not in name
                and "false_positive" not in name
                and "false_negative" not in name
                and "true_positive" not in name
            ):
                return p

    raise FileNotFoundError(
        f"Could not find {kind} NIfTI file inside:\n{patient_dir}"
    )


# ============================================================
# 4. PREPROCESSING - SAME AS EXPERIMENT 05B
# ============================================================

def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct, HU_MIN, HU_MAX)
    ct = (ct - HU_MIN) / (HU_MAX - HU_MIN)
    return ct.astype(np.float32)


# ============================================================
# 5. METRICS
# ============================================================

def calculate_metrics(prediction: np.ndarray, reference: np.ndarray):
    prediction = prediction.astype(bool)
    reference = reference.astype(bool)

    tp = int(np.logical_and(prediction, reference).sum())
    fp = int(
        np.logical_and(
            prediction,
            np.logical_not(reference),
        ).sum()
    )
    fn = int(
        np.logical_and(
            np.logical_not(prediction),
            reference,
        ).sum()
    )

    eps = 1e-8

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reference_voxels": int(reference.sum()),
        "predicted_voxels": int(prediction.sum()),
    }


def per_slice_metrics(prediction: np.ndarray, reference: np.ndarray):
    rows = []

    for slice_idx in range(reference.shape[2]):
        ref_slice = reference[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]

        ref_pixels = int(ref_slice.sum())
        pred_pixels = int(pred_slice.sum())

        metrics = calculate_metrics(pred_slice, ref_slice)

        both_empty = ref_pixels == 0 and pred_pixels == 0

        rows.append(
            {
                "slice_index": slice_idx,
                "reference_pixels": ref_pixels,
                "predicted_pixels": pred_pixels,
                "dice": np.nan if both_empty else metrics["dice"],
                "iou": np.nan if both_empty else metrics["iou"],
                "precision": np.nan if both_empty else metrics["precision"],
                "recall": np.nan if both_empty else metrics["recall"],
            }
        )

    return rows


# ============================================================
# 6. LOAD MODEL
# ============================================================

if not BEST_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Best Experiment 05B model not found:\n{BEST_MODEL_PATH}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 75)
print("EXPERIMENT 05B VALIDATION VISUALIZATION")
print("=" * 75)
print(f"Device: {device}")
print(f"Model:  {BEST_MODEL_PATH}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )
)

model.eval()
print("Best model loaded successfully.")


# ============================================================
# 7. INFERENCE FUNCTION
# ============================================================

def predict_full_volume(ct: np.ndarray) -> np.ndarray:
    prediction = np.zeros_like(ct, dtype=np.uint8)

    with torch.no_grad():
        for start in range(0, ct.shape[2], INFERENCE_BATCH_SIZE):
            end = min(start + INFERENCE_BATCH_SIZE, ct.shape[2])

            # [H, W, B] -> [B, H, W]
            batch_np = np.moveaxis(
                ct[:, :, start:end],
                2,
                0,
            )

            # -> [B, 1, H, W]
            batch_tensor = torch.from_numpy(
                batch_np.astype(np.float32)
            ).unsqueeze(1).to(device)

            logits = model(batch_tensor)
            probabilities = torch.sigmoid(logits)

            batch_prediction = (
                probabilities[:, 0]
                .cpu()
                .numpy()
                > PREDICTION_THRESHOLD
            ).astype(np.uint8)

            # [B, H, W] -> [H, W, B]
            prediction[:, :, start:end] = np.moveaxis(
                batch_prediction,
                0,
                2,
            )

    return prediction


# ============================================================
# 8. VISUALIZATION
# ============================================================

def save_comparison_figure(
    patient_name: str,
    ct: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    slice_index: int,
    title_suffix: str,
    output_path: Path,
    whole_volume_metrics: dict,
):
    ct_slice = ct[:, :, slice_index]
    ref_slice = reference[:, :, slice_index]
    pred_slice = prediction[:, :, slice_index]

    ref_overlay = np.ma.masked_where(ref_slice == 0, ref_slice)
    pred_overlay = np.ma.masked_where(pred_slice == 0, pred_slice)

    error_map = np.zeros_like(ref_slice, dtype=np.uint8)

    # 1 = true positive / overlap
    error_map[(ref_slice == 1) & (pred_slice == 1)] = 1

    # 2 = false positive
    error_map[(ref_slice == 0) & (pred_slice == 1)] = 2

    # 3 = false negative
    error_map[(ref_slice == 1) & (pred_slice == 0)] = 3

    error_overlay = np.ma.masked_where(error_map == 0, error_map)

    slice_metrics = calculate_metrics(pred_slice, ref_slice)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title(
        f"Original CT\nSlice {slice_index}"
    )

    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(ref_overlay, alpha=0.55)
    axes[1].set_title("Reference Mask")

    axes[2].imshow(ct_slice, cmap="gray")
    axes[2].imshow(pred_overlay, alpha=0.55)
    axes[2].set_title("U-Net Prediction")

    axes[3].imshow(ct_slice, cmap="gray")
    axes[3].imshow(error_overlay, alpha=0.65)
    axes[3].set_title(
        "Error Map\n"
        "1=overlap, 2=FP, 3=FN"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{patient_name} - {title_suffix}\n"
        f"Slice Dice={slice_metrics['dice']:.4f} | "
        f"Whole-volume Dice={whole_volume_metrics['dice']:.4f} | "
        f"Whole-volume IoU={whole_volume_metrics['iou']:.4f}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 9. VALIDATE PAT_11 AND PAT_12
# ============================================================

summary_rows = []

for patient_name in VALIDATION_PATIENTS:

    print("\n" + "=" * 75)
    print(f"VALIDATION PATIENT: {patient_name}")
    print("=" * 75)

    patient_dir = BASE_DIR / patient_name

    ct_path = find_patient_file(patient_dir, "ct")
    mask_path = find_patient_file(patient_dir, "mask")

    print(f"CT:   {ct_path}")
    print(f"Mask: {mask_path}")

    ct_nifti = nib.load(str(ct_path))
    mask_nifti = nib.load(str(mask_path))

    raw_ct = ct_nifti.get_fdata(dtype=np.float32)
    reference = (
        mask_nifti.get_fdata() > 0
    ).astype(np.uint8)

    if raw_ct.shape != reference.shape:
        raise ValueError(
            f"{patient_name}: CT/mask shape mismatch: "
            f"{raw_ct.shape} vs {reference.shape}"
        )

    if not np.allclose(
        ct_nifti.affine,
        mask_nifti.affine,
        atol=1e-4,
    ):
        raise ValueError(
            f"{patient_name}: CT/mask affine mismatch."
        )

    orientation = nib.aff2axcodes(ct_nifti.affine)
    spacing = ct_nifti.header.get_zooms()[:3]

    print(f"Shape:       {raw_ct.shape}")
    print(f"Orientation: {orientation}")
    print(
        "Spacing:     "
        f"{spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f} mm"
    )

    ct = preprocess_ct(raw_ct)

    print("Running whole-volume inference...")
    prediction = predict_full_volume(ct)

    metrics = calculate_metrics(
        prediction,
        reference,
    )

    print("\nWHOLE-VOLUME RESULTS")
    print(f"Dice:      {metrics['dice']:.4f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Reference voxels: {metrics['reference_voxels']}")
    print(f"Predicted voxels: {metrics['predicted_voxels']}")
    print(f"True positives:   {metrics['tp']}")
    print(f"False positives:  {metrics['fp']}")
    print(f"False negatives:  {metrics['fn']}")

    patient_output_dir = OUTPUT_DIR / patient_name
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Save 3D prediction
    # --------------------------------------------------------
    prediction_path = (
        patient_output_dir
        / f"{patient_name}_Experiment05B_prediction.nii.gz"
    )

    prediction_nifti = nib.Nifti1Image(
        prediction.astype(np.uint8),
        ct_nifti.affine,
        ct_nifti.header,
    )

    nib.save(
        prediction_nifti,
        str(prediction_path),
    )

    # --------------------------------------------------------
    # Save TP / FP / FN masks
    # --------------------------------------------------------
    tp_mask = np.logical_and(
        prediction == 1,
        reference == 1,
    ).astype(np.uint8)

    fp_mask = np.logical_and(
        prediction == 1,
        reference == 0,
    ).astype(np.uint8)

    fn_mask = np.logical_and(
        prediction == 0,
        reference == 1,
    ).astype(np.uint8)

    for name, array in [
        ("true_positive", tp_mask),
        ("false_positive", fp_mask),
        ("false_negative", fn_mask),
    ]:
        output_nifti = nib.Nifti1Image(
            array,
            ct_nifti.affine,
            ct_nifti.header,
        )

        nib.save(
            output_nifti,
            str(
                patient_output_dir
                / f"{patient_name}_{name}.nii.gz"
            ),
        )

    # --------------------------------------------------------
    # Slice ranges
    # --------------------------------------------------------
    reference_counts = np.sum(
        reference,
        axis=(0, 1),
    )

    prediction_counts = np.sum(
        prediction,
        axis=(0, 1),
    )

    reference_positive = np.where(
        reference_counts > 0
    )[0]

    prediction_positive = np.where(
        prediction_counts > 0
    )[0]

    def range_string(indices):
        if len(indices) == 0:
            return "None"
        return (
            f"{int(indices.min())}-"
            f"{int(indices.max())}"
        )

    print(
        "Reference positive slice range: "
        f"{range_string(reference_positive)}"
    )

    print(
        "Predicted positive slice range: "
        f"{range_string(prediction_positive)}"
    )

    # --------------------------------------------------------
    # Per-slice metrics
    # --------------------------------------------------------
    slice_rows = per_slice_metrics(
        prediction,
        reference,
    )

    per_slice_csv = (
        patient_output_dir
        / f"{patient_name}_per_slice_metrics.csv"
    )

    with per_slice_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice_index",
                "reference_pixels",
                "predicted_pixels",
                "dice",
                "iou",
                "precision",
                "recall",
            ],
        )

        writer.writeheader()
        writer.writerows(slice_rows)

    # --------------------------------------------------------
    # Identify important slices
    # --------------------------------------------------------

    # A. Largest reference pancreas area
    largest_reference_slice = int(
        np.argmax(reference_counts)
    )

    # B. Best Dice slice among slices where reference pancreas exists
    valid_rows = [
        row for row in slice_rows
        if row["reference_pixels"] > 0
        and not np.isnan(row["dice"])
    ]

    best_slice_row = max(
        valid_rows,
        key=lambda row: row["dice"],
    )

    best_dice_slice = int(
        best_slice_row["slice_index"]
    )

    # C. Worst slice among meaningful pancreas slices.
    # Avoid tiny edge slices by requiring at least 10% of max reference area.
    meaningful_threshold = max(
        1,
        int(reference_counts.max() * 0.10),
    )

    meaningful_rows = [
        row for row in valid_rows
        if row["reference_pixels"] >= meaningful_threshold
    ]

    if meaningful_rows:
        worst_slice_row = min(
            meaningful_rows,
            key=lambda row: row["dice"],
        )
    else:
        worst_slice_row = min(
            valid_rows,
            key=lambda row: row["dice"],
        )

    worst_dice_slice = int(
        worst_slice_row["slice_index"]
    )

    print(
        f"Largest reference slice: {largest_reference_slice}"
    )
    print(
        f"Best Dice slice:          "
        f"{best_dice_slice} "
        f"(Dice={best_slice_row['dice']:.4f})"
    )
    print(
        f"Worst meaningful slice:   "
        f"{worst_dice_slice} "
        f"(Dice={worst_slice_row['dice']:.4f})"
    )

    # --------------------------------------------------------
    # Save figures
    # --------------------------------------------------------
    save_comparison_figure(
        patient_name,
        ct,
        reference,
        prediction,
        largest_reference_slice,
        "Largest Reference Pancreas Slice",
        patient_output_dir
        / f"{patient_name}_largest_reference_slice.png",
        metrics,
    )

    save_comparison_figure(
        patient_name,
        ct,
        reference,
        prediction,
        best_dice_slice,
        "Best Dice Slice",
        patient_output_dir
        / f"{patient_name}_best_dice_slice.png",
        metrics,
    )

    save_comparison_figure(
        patient_name,
        ct,
        reference,
        prediction,
        worst_dice_slice,
        "Worst Meaningful Pancreas Slice",
        patient_output_dir
        / f"{patient_name}_worst_dice_slice.png",
        metrics,
    )

    summary_rows.append(
        {
            "patient": patient_name,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "reference_voxels":
                metrics["reference_voxels"],
            "predicted_voxels":
                metrics["predicted_voxels"],
            "true_positive_voxels":
                metrics["tp"],
            "false_positive_voxels":
                metrics["fp"],
            "false_negative_voxels":
                metrics["fn"],
            "reference_slice_range":
                range_string(reference_positive),
            "predicted_slice_range":
                range_string(prediction_positive),
            "largest_reference_slice":
                largest_reference_slice,
            "best_dice_slice":
                best_dice_slice,
            "best_slice_dice":
                best_slice_row["dice"],
            "worst_meaningful_slice":
                worst_dice_slice,
            "worst_slice_dice":
                worst_slice_row["dice"],
            "prediction_path":
                str(prediction_path),
        }
    )


# ============================================================
# 10. SAVE COMBINED VALIDATION SUMMARY
# ============================================================

summary_csv = (
    OUTPUT_DIR
    / "Experiment05B_validation_summary.csv"
)

with summary_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# 11. PRINT FINAL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT 05B VALIDATION SUMMARY")
print("=" * 75)

for row in summary_rows:
    print(
        f"{row['patient']}: "
        f"Dice={row['dice']:.4f}, "
        f"IoU={row['iou']:.4f}, "
        f"Precision={row['precision']:.4f}, "
        f"Recall={row['recall']:.4f}"
    )

mean_dice = float(
    np.mean([
        row["dice"]
        for row in summary_rows
    ])
)

print("-" * 75)
print(f"Mean validation Dice: {mean_dice:.4f}")
print(f"Results saved to:\n{OUTPUT_DIR}")

print(
    "\nNext step: inspect Pat_11 and Pat_12 prediction/error figures "
    "before changing preprocessing or moving to Experiment 06."
)