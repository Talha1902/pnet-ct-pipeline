from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import UNet
from scipy import ndimage

# ============================================================
# Experiment 06B — 3D Largest Connected Component Post-processing
# Uses the BEST Experiment 06A checkpoint. NO RETRAINING.
# Validation only: Pat_11 and Pat_12.
# Final test patients Pat_13–Pat_15 remain untouched.
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")

EXPERIMENT_06A_MODEL = Path(
    r"D:/Thesis Folder/Results/Experiment_06A_results/"
    r"Experiment_06A_1to1_Negative_Sampling/"
    r"Experiment_06A_best_model.pth"
)

OUTPUT_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_06B_results/"
    r"Experiment_06B_Largest_Connected_Component"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_PATIENTS = ["Pat_11", "Pat_12"]

# Must match Experiment 06A
HU_MIN = -100.0
HU_MAX = 250.0
PREDICTION_THRESHOLD = 0.5
INFERENCE_BATCH_SIZE = 8

SUMMARY_CSV = OUTPUT_DIR / "Experiment_06B_raw_vs_lcc_summary.csv"
COMPONENT_CSV = OUTPUT_DIR / "Experiment_06B_component_analysis.csv"


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def find_patient_file(patient_dir: Path, kind: str) -> Path:
    files = [p for p in patient_dir.iterdir() if p.is_file() and is_nifti(p)]

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
            if (("origional" in name or "original" in name)
                    and "mask" not in name
                    and "prediction" not in name):
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
            if (("masked_images" in name or "mask_images" in name)
                    and "prediction" not in name
                    and "false_positive" not in name
                    and "false_negative" not in name
                    and "true_positive" not in name):
                return p

    raise FileNotFoundError(f"Could not find {kind} NIfTI file inside:\n{patient_dir}")


def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct, HU_MIN, HU_MAX)
    ct = (ct - HU_MIN) / (HU_MAX - HU_MIN)
    return ct.astype(np.float32)


def calculate_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict:
    pb = prediction.astype(bool)
    rb = reference.astype(bool)

    tp = int(np.logical_and(pb, rb).sum())
    fp = int(np.logical_and(pb, ~rb).sum())
    fn = int(np.logical_and(~pb, rb).sum())

    eps = 1e-8
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    ref_counts = reference.sum(axis=(0, 1))
    pred_counts = prediction.sum(axis=(0, 1))

    ref_pos = np.where(ref_counts > 0)[0]
    pred_pos = np.where(pred_counts > 0)[0]

    fp_slices = np.setdiff1d(pred_pos, ref_pos)
    missed_slices = np.setdiff1d(ref_pos, pred_pos)

    def range_string(indices):
        if len(indices) == 0:
            return "None"
        return f"{int(indices.min())}-{int(indices.max())}"

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "reference_voxels": int(reference.sum()),
        "predicted_voxels": int(prediction.sum()),
        "true_positive_voxels": tp,
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "reference_slice_range": range_string(ref_pos),
        "predicted_slice_range": range_string(pred_pos),
        "reference_positive_slices": int(len(ref_pos)),
        "predicted_positive_slices": int(len(pred_pos)),
        "false_positive_slice_count": int(len(fp_slices)),
        "missed_reference_slice_count": int(len(missed_slices)),
    }


def largest_connected_component(binary_mask: np.ndarray):
    """Keep only the largest 3D foreground component using 26-connectivity."""
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_components = ndimage.label(binary_mask.astype(bool), structure=structure)

    if num_components == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8), []

    sizes = np.bincount(labeled.ravel())[1:]
    largest_label = int(np.argmax(sizes) + 1)
    lcc = (labeled == largest_label).astype(np.uint8)
    sorted_sizes = sorted([int(x) for x in sizes], reverse=True)
    return lcc, sorted_sizes


if not EXPERIMENT_06A_MODEL.exists():
    raise FileNotFoundError(
        f"Experiment 06A best model not found:\n{EXPERIMENT_06A_MODEL}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("EXPERIMENT 06B - 3D LARGEST CONNECTED COMPONENT")
print("=" * 80)
print(f"Device: {device}")
print(f"Model:  {EXPERIMENT_06A_MODEL}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load(EXPERIMENT_06A_MODEL, map_location=device))
model.eval()


def predict_full_volume(ct: np.ndarray) -> np.ndarray:
    prediction = np.zeros(ct.shape, dtype=np.uint8)

    with torch.no_grad():
        for start in range(0, ct.shape[2], INFERENCE_BATCH_SIZE):
            end = min(start + INFERENCE_BATCH_SIZE, ct.shape[2])
            batch_np = np.moveaxis(ct[:, :, start:end], 2, 0)
            batch_tensor = torch.from_numpy(batch_np.astype(np.float32)).unsqueeze(1).to(device)
            probs = torch.sigmoid(model(batch_tensor))
            batch_pred = (probs[:, 0].cpu().numpy() > PREDICTION_THRESHOLD).astype(np.uint8)
            prediction[:, :, start:end] = np.moveaxis(batch_pred, 0, 2)

    return prediction


def save_comparison_figure(
    patient_name,
    ct,
    reference,
    raw_prediction,
    lcc_prediction,
    raw_metrics,
    lcc_metrics,
    output_path,
):
    ref_counts = reference.sum(axis=(0, 1))
    slice_idx = int(np.argmax(ref_counts))

    ct_slice = ct[:, :, slice_idx]
    ref_slice = reference[:, :, slice_idx]
    raw_slice = raw_prediction[:, :, slice_idx]
    lcc_slice = lcc_prediction[:, :, slice_idx]

    ref_overlay = np.ma.masked_where(ref_slice == 0, ref_slice)
    raw_overlay = np.ma.masked_where(raw_slice == 0, raw_slice)
    lcc_overlay = np.ma.masked_where(lcc_slice == 0, lcc_slice)

    error_map = np.zeros_like(ref_slice, dtype=np.uint8)
    error_map[(ref_slice == 1) & (lcc_slice == 1)] = 1
    error_map[(ref_slice == 0) & (lcc_slice == 1)] = 2
    error_map[(ref_slice == 1) & (lcc_slice == 0)] = 3
    error_overlay = np.ma.masked_where(error_map == 0, error_map)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title(f"Original CT\nSlice {slice_idx}")

    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(ref_overlay, alpha=0.55)
    axes[1].set_title("Reference Mask")

    axes[2].imshow(ct_slice, cmap="gray")
    axes[2].imshow(raw_overlay, alpha=0.55)
    axes[2].set_title(f"Raw U-Net\nDice={raw_metrics['dice']:.4f}")

    axes[3].imshow(ct_slice, cmap="gray")
    axes[3].imshow(lcc_overlay, alpha=0.55)
    axes[3].set_title(f"LCC Prediction\nDice={lcc_metrics['dice']:.4f}")

    axes[4].imshow(ct_slice, cmap="gray")
    axes[4].imshow(error_overlay, alpha=0.65)
    axes[4].set_title("LCC Error Map\n1=overlap, 2=FP, 3=FN")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{patient_name} - Experiment 06B\n"
        f"Raw Dice={raw_metrics['dice']:.4f} → LCC Dice={lcc_metrics['dice']:.4f} | "
        f"Raw Precision={raw_metrics['precision']:.4f} → LCC Precision={lcc_metrics['precision']:.4f}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


summary_rows = []
component_rows = []

for patient_name in VALIDATION_PATIENTS:
    print("\n" + "=" * 80)
    print(f"VALIDATION PATIENT: {patient_name}")
    print("=" * 80)

    patient_dir = BASE_DIR / patient_name
    ct_path = find_patient_file(patient_dir, "ct")
    ref_path = find_patient_file(patient_dir, "mask")

    ct_img = nib.load(str(ct_path))
    ref_img = nib.load(str(ref_path))

    raw_ct = ct_img.get_fdata(dtype=np.float32)
    reference = (ref_img.get_fdata() > 0).astype(np.uint8)

    if raw_ct.shape != reference.shape:
        raise ValueError(f"{patient_name}: CT/mask shape mismatch")
    if not np.allclose(ct_img.affine, ref_img.affine, atol=1e-4):
        raise ValueError(f"{patient_name}: CT/mask affine mismatch")

    ct = preprocess_ct(raw_ct)
    print(f"Shape: {ct.shape}")
    print("Running Experiment 06A inference...")

    raw_prediction = predict_full_volume(ct)
    raw_metrics = calculate_metrics(raw_prediction, reference)

    print("\nRAW U-NET RESULTS")
    print(f"Dice:      {raw_metrics['dice']:.4f}")
    print(f"Precision: {raw_metrics['precision']:.4f}")
    print(f"Recall:    {raw_metrics['recall']:.4f}")
    print(f"FP voxels: {raw_metrics['false_positive_voxels']}")
    print(f"FP slices: {raw_metrics['false_positive_slice_count']}")
    print(f"Predicted slice range: {raw_metrics['predicted_slice_range']}")

    lcc_prediction, component_sizes = largest_connected_component(raw_prediction)
    lcc_metrics = calculate_metrics(lcc_prediction, reference)

    print("\nLARGEST CONNECTED COMPONENT RESULTS")
    print(f"Dice:      {lcc_metrics['dice']:.4f}")
    print(f"Precision: {lcc_metrics['precision']:.4f}")
    print(f"Recall:    {lcc_metrics['recall']:.4f}")
    print(f"FP voxels: {lcc_metrics['false_positive_voxels']}")
    print(f"FP slices: {lcc_metrics['false_positive_slice_count']}")
    print(f"Predicted slice range: {lcc_metrics['predicted_slice_range']}")
    print(f"Connected components found: {len(component_sizes)}")
    if component_sizes:
        print(f"Largest component size: {component_sizes[0]} voxels")
        print(f"Top 10 component sizes: {component_sizes[:10]}")

    patient_output_dir = OUTPUT_DIR / patient_name
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = patient_output_dir / f"{patient_name}_Experiment06A_raw_prediction.nii.gz"
    lcc_path = patient_output_dir / f"{patient_name}_Experiment06B_LCC_prediction.nii.gz"

    nib.save(
        nib.Nifti1Image(raw_prediction.astype(np.uint8), ct_img.affine, ct_img.header),
        str(raw_path),
    )
    nib.save(
        nib.Nifti1Image(lcc_prediction.astype(np.uint8), ct_img.affine, ct_img.header),
        str(lcc_path),
    )

    lcc_fp = np.logical_and(lcc_prediction == 1, reference == 0).astype(np.uint8)
    lcc_fn = np.logical_and(lcc_prediction == 0, reference == 1).astype(np.uint8)

    nib.save(
        nib.Nifti1Image(lcc_fp, ct_img.affine, ct_img.header),
        str(patient_output_dir / f"{patient_name}_Experiment06B_LCC_false_positive.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(lcc_fn, ct_img.affine, ct_img.header),
        str(patient_output_dir / f"{patient_name}_Experiment06B_LCC_false_negative.nii.gz"),
    )

    save_comparison_figure(
        patient_name,
        ct,
        reference,
        raw_prediction,
        lcc_prediction,
        raw_metrics,
        lcc_metrics,
        patient_output_dir / f"{patient_name}_Experiment06B_raw_vs_LCC.png",
    )

    for method_name, metrics in [
        ("RAW_UNET", raw_metrics),
        ("LARGEST_CONNECTED_COMPONENT", lcc_metrics),
    ]:
        summary_rows.append({"patient": patient_name, "method": method_name, **metrics})

    for rank, size in enumerate(component_sizes, start=1):
        component_rows.append({
            "patient": patient_name,
            "component_rank_by_size": rank,
            "component_size_voxels": size,
            "is_largest_component": rank == 1,
        })


summary_fields = [
    "patient", "method", "dice", "iou", "precision", "recall",
    "reference_voxels", "predicted_voxels", "true_positive_voxels",
    "false_positive_voxels", "false_negative_voxels",
    "reference_slice_range", "predicted_slice_range",
    "reference_positive_slices", "predicted_positive_slices",
    "false_positive_slice_count", "missed_reference_slice_count",
]

with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

component_fields = [
    "patient", "component_rank_by_size", "component_size_voxels", "is_largest_component"
]

with COMPONENT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=component_fields)
    writer.writeheader()
    writer.writerows(component_rows)

print("\n" + "=" * 80)
print("EXPERIMENT 06B SUMMARY")
print("=" * 80)

for patient_name in VALIDATION_PATIENTS:
    raw_row = next(
        row for row in summary_rows
        if row["patient"] == patient_name and row["method"] == "RAW_UNET"
    )
    lcc_row = next(
        row for row in summary_rows
        if row["patient"] == patient_name and row["method"] == "LARGEST_CONNECTED_COMPONENT"
    )

    print(f"\n{patient_name}")
    print(f"  Dice:       {raw_row['dice']:.4f} → {lcc_row['dice']:.4f}")
    print(f"  Precision:  {raw_row['precision']:.4f} → {lcc_row['precision']:.4f}")
    print(f"  Recall:     {raw_row['recall']:.4f} → {lcc_row['recall']:.4f}")
    print(f"  FP voxels:  {raw_row['false_positive_voxels']} → {lcc_row['false_positive_voxels']}")
    print(f"  FP slices:  {raw_row['false_positive_slice_count']} → {lcc_row['false_positive_slice_count']}")
    print(f"  Pred range: {raw_row['predicted_slice_range']} → {lcc_row['predicted_slice_range']}")

raw_mean = float(np.mean([
    row["dice"] for row in summary_rows if row["method"] == "RAW_UNET"
]))
lcc_mean = float(np.mean([
    row["dice"] for row in summary_rows if row["method"] == "LARGEST_CONNECTED_COMPONENT"
]))

print("\n" + "-" * 80)
print(f"Mean RAW validation Dice: {raw_mean:.4f}")
print(f"Mean LCC validation Dice: {lcc_mean:.4f}")
print(f"Absolute Dice change:     {lcc_mean - raw_mean:+.4f}")
print(f"\nSummary CSV:\n{SUMMARY_CSV}")
print(f"\nComponent analysis CSV:\n{COMPONENT_CSV}")
print("\nIMPORTANT: Pat_13, Pat_14, and Pat_15 remain untouched.")