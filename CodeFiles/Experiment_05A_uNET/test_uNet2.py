
"""
Experiment 04 - FINAL held-out test evaluation.

Run this ONLY after train_unet_v2.py has finished and you have frozen the
model/hyperparameters.

The script:
- loads the BEST validation checkpoint
- evaluates Pat_10 and Pat_11
- predicts every axial slice
- reconstructs a complete 3D NIfTI mask
- computes whole-volume Dice, IoU, precision, recall
- saves one prediction per patient
- saves one visual comparison per patient
- writes a combined final_test_metrics.csv

IMPORTANT:
Pat_10 / Pat_11 must NOT have been used to tune learning rate, epochs,
threshold, architecture, or other model choices.

Install if needed:
    pip install torch monai nibabel numpy matplotlib

Run:
    python test_unet_v2.py
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
# 1. CONFIGURATION - MUST MATCH TRAINING
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")
Results = Path (r"D:/Thesis Folder/Results/Experiment_4_results")
EXPERIMENT_DIR = Results/ "Experiment_04_UNet_v2"

MODEL_PATH = EXPERIMENT_DIR / "best_pancreas_unet_v2.pth"

TEST_PATIENTS = [
    "Pat_13",
    "Pat_14",
    "Pat_15",
]

HU_MIN = -100.0
HU_MAX = 250.0
PREDICTION_THRESHOLD = 0.5
INFERENCE_BATCH_SIZE = 8

TEST_OUTPUT_DIR = EXPERIMENT_DIR / "Final_Test_Results"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = TEST_OUTPUT_DIR / "final_test_metrics.csv"


# ============================================================
# 2. HELPERS
# ============================================================

def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def find_patient_file(patient_dir: Path, kind: str) -> Path:
    files = [
        p for p in patient_dir.iterdir()
        if p.is_file() and is_nifti(p)
    ]

    if kind == "ct":
        exact_candidates = [
            "origional_images.nii",
            "origional_images.nii.gz",
            "original_images.nii",
            "original_images.nii.gz",
        ]

        for candidate in exact_candidates:
            for p in files:
                if p.name.lower() == candidate:
                    return p

        for p in files:
            name = p.name.lower()
            if (
                ("origional" in name or "original" in name)
                and "prediction" not in name
                and "mask" not in name
            ):
                return p

    elif kind == "mask":
        exact_candidates = [
            "masked_images.nii",
            "masked_images.nii.gz",
            "mask_images.nii",
            "mask_images.nii.gz",
        ]

        for candidate in exact_candidates:
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
        f"Could not find {kind} NIfTI inside {patient_dir}"
    )


def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct, HU_MIN, HU_MAX)
    ct = (ct - HU_MIN) / (HU_MAX - HU_MIN)
    return ct.astype(np.float32)


def calculate_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
):
    prediction = prediction.astype(bool)
    reference = reference.astype(bool)

    tp = int(
        np.logical_and(
            prediction,
            reference,
        ).sum()
    )

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

    dice = (
        2.0 * tp + eps
    ) / (
        2.0 * tp + fp + fn + eps
    )

    iou = (
        tp + eps
    ) / (
        tp + fp + fn + eps
    )

    precision = (
        tp + eps
    ) / (
        tp + fp + eps
    )

    recall = (
        tp + eps
    ) / (
        tp + fn + eps
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive_voxels": tp,
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "reference_voxels": int(reference.sum()),
        "predicted_voxels": int(prediction.sum()),
    }


# ============================================================
# 3. LOAD BEST MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Best model not found:\n{MODEL_PATH}\n"
        "Run train_unet_v2.py first."
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Testing device: {device}")

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
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()

print(f"Loaded BEST model:\n{MODEL_PATH}")


# ============================================================
# 4. TEST EACH HELD-OUT PATIENT
# ============================================================

summary_rows = []

for patient_name in TEST_PATIENTS:

    print("\n" + "=" * 70)
    print(f"FINAL TEST PATIENT: {patient_name}")
    print("=" * 70)

    patient_dir = BASE_DIR / patient_name

    ct_path = find_patient_file(
        patient_dir,
        "ct",
    )

    reference_path = find_patient_file(
        patient_dir,
        "mask",
    )

    ct_nifti = nib.load(str(ct_path))
    reference_nifti = nib.load(str(reference_path))

    raw_ct = ct_nifti.get_fdata(dtype=np.float32)
    reference = (
        reference_nifti.get_fdata() > 0
    ).astype(np.uint8)

    if raw_ct.shape != reference.shape:
        raise ValueError(
            f"{patient_name}: CT/mask shape mismatch."
        )

    if not np.allclose(
        ct_nifti.affine,
        reference_nifti.affine,
        atol=1e-4,
    ):
        raise ValueError(
            f"{patient_name}: CT/mask affine mismatch."
        )

    ct = preprocess_ct(raw_ct)

    prediction = np.zeros_like(
        reference,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Slice-wise 2D inference across FULL patient volume
    # --------------------------------------------------------
    with torch.no_grad():

        for start in range(
            0,
            ct.shape[2],
            INFERENCE_BATCH_SIZE,
        ):
            end = min(
                start + INFERENCE_BATCH_SIZE,
                ct.shape[2],
            )

            batch_np = np.moveaxis(
                ct[:, :, start:end],
                2,
                0,
            )

            batch_tensor = torch.from_numpy(
                batch_np.astype(np.float32)
            ).unsqueeze(1).to(device)

            logits = model(batch_tensor)
            probability = torch.sigmoid(logits)

            predicted_batch = (
                probability[:, 0]
                .cpu()
                .numpy()
                > PREDICTION_THRESHOLD
            ).astype(np.uint8)

            prediction[:, :, start:end] = np.moveaxis(
                predicted_batch,
                0,
                2,
            )

    # --------------------------------------------------------
    # Quantitative metrics
    # --------------------------------------------------------
    metrics = calculate_metrics(
        prediction,
        reference,
    )

    print(f"Dice:      {metrics['dice']:.4f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(
        f"Reference voxels: "
        f"{metrics['reference_voxels']}"
    )
    print(
        f"Predicted voxels: "
        f"{metrics['predicted_voxels']}"
    )

    # --------------------------------------------------------
    # Save 3D prediction with original CT geometry
    # --------------------------------------------------------
    patient_output_dir = (
        TEST_OUTPUT_DIR / patient_name
    )

    patient_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        patient_output_dir
        / f"{patient_name}_UNet_v2_prediction.nii.gz"
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
    # Slice ranges
    # --------------------------------------------------------
    ref_counts = np.sum(
        reference,
        axis=(0, 1),
    )

    pred_counts = np.sum(
        prediction,
        axis=(0, 1),
    )

    ref_positive = np.where(
        ref_counts > 0
    )[0]

    pred_positive = np.where(
        pred_counts > 0
    )[0]

    def range_string(indices):
        if len(indices) == 0:
            return "None"
        return (
            f"{int(indices.min())}-"
            f"{int(indices.max())}"
        )

    # --------------------------------------------------------
    # Objective visualization:
    # choose largest REFERENCE pancreas slice
    # --------------------------------------------------------
    comparison_slice = int(
        np.argmax(ref_counts)
    )

    ct_slice = ct[:, :, comparison_slice]
    ref_slice = reference[:, :, comparison_slice]
    pred_slice = prediction[:, :, comparison_slice]

    ref_overlay = np.ma.masked_where(
        ref_slice == 0,
        ref_slice,
    )

    pred_overlay = np.ma.masked_where(
        pred_slice == 0,
        pred_slice,
    )

    error_map = np.zeros_like(
        ref_slice,
        dtype=np.uint8,
    )

    # 1 = overlap
    # 2 = false positive
    # 3 = false negative
    error_map[
        (ref_slice == 1)
        & (pred_slice == 1)
    ] = 1

    error_map[
        (ref_slice == 0)
        & (pred_slice == 1)
    ] = 2

    error_map[
        (ref_slice == 1)
        & (pred_slice == 0)
    ] = 3

    error_overlay = np.ma.masked_where(
        error_map == 0,
        error_map,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
    )

    axes[0].imshow(
        ct_slice,
        cmap="gray",
    )
    axes[0].set_title(
        f"Original CT\nSlice {comparison_slice}"
    )

    axes[1].imshow(
        ct_slice,
        cmap="gray",
    )
    axes[1].imshow(
        ref_overlay,
        alpha=0.55,
    )
    axes[1].set_title(
        "Reference Mask"
    )

    axes[2].imshow(
        ct_slice,
        cmap="gray",
    )
    axes[2].imshow(
        pred_overlay,
        alpha=0.55,
    )
    axes[2].set_title(
        "U-Net v2 Prediction"
    )

    axes[3].imshow(
        ct_slice,
        cmap="gray",
    )
    axes[3].imshow(
        error_overlay,
        alpha=0.65,
    )
    axes[3].set_title(
        "Error Map\n"
        "1 overlap, 2 FP, 3 FN"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{patient_name} Final Held-Out Test\n"
        f"Dice={metrics['dice']:.4f}, "
        f"IoU={metrics['iou']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}"
    )

    plt.tight_layout()

    figure_path = (
        patient_output_dir
        / f"{patient_name}_comparison.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Summary row
    # --------------------------------------------------------
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
                metrics["true_positive_voxels"],
            "false_positive_voxels":
                metrics["false_positive_voxels"],
            "false_negative_voxels":
                metrics["false_negative_voxels"],
            "reference_slice_range":
                range_string(ref_positive),
            "predicted_slice_range":
                range_string(pred_positive),
            "comparison_slice":
                comparison_slice,
            "prediction_file":
                str(prediction_path),
        }
    )


# ============================================================
# 5. SAVE COMBINED TEST RESULTS
# ============================================================

with SUMMARY_CSV.open(
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
# 6. FINAL SUMMARY
# ============================================================

mean_dice = float(
    np.mean([
        row["dice"]
        for row in summary_rows
    ])
)

mean_iou = float(
    np.mean([
        row["iou"]
        for row in summary_rows
    ])
)

mean_precision = float(
    np.mean([
        row["precision"]
        for row in summary_rows
    ])
)

mean_recall = float(
    np.mean([
        row["recall"]
        for row in summary_rows
    ])
)

print("\n" + "=" * 70)
print("FINAL HELD-OUT TEST SUMMARY")
print("=" * 70)

for row in summary_rows:
    print(
        f"{row['patient']}: "
        f"Dice={row['dice']:.4f}, "
        f"IoU={row['iou']:.4f}, "
        f"Precision={row['precision']:.4f}, "
        f"Recall={row['recall']:.4f}"
    )

print("-" * 70)
print(f"Mean Dice:      {mean_dice:.4f}")
print(f"Mean IoU:       {mean_iou:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall:    {mean_recall:.4f}")
print(f"\nFull results saved to:\n{SUMMARY_CSV}")

print(
    "\nRemember: these scores evaluate agreement with the "
    "TotalSegmentator-generated reference masks unless those masks "
    "have subsequently been validated by a radiologist."
)