
"""
Experiment 06C — Minimum 3D Component-Size Filtering
====================================================

Research question:
Can we reduce small disconnected false-positive pancreas predictions while
preserving multiple potentially correct pancreas fragments?

This experiment uses the BEST Experiment 06A checkpoint and evaluates only
Pat_11 and Pat_12. NO RETRAINING occurs.

Thresholds:
RAW, 25, 50, 100, 250, 500, 1000 voxels.

Protected final test patients:
Pat_13, Pat_14, Pat_15 — DO NOT use them here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import UNet
from scipy import ndimage


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")

MODEL_PATH = Path(
    r"D:/Thesis Folder/Results/Experiment_06A_results/"
    r"Experiment_06A_1to1_Negative_Sampling/"
    r"Experiment_06A_best_model.pth"
)

OUTPUT_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_06C_results/"
    r"Experiment_06C_Minimum_Component_Filtering"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_PATIENTS = ["Pat_11", "Pat_12"]
FINAL_TEST_PATIENTS = ["Pat_13", "Pat_14", "Pat_15"]

HU_MIN = -100.0
HU_MAX = 250.0
PRED_THRESHOLD = 0.5
INFERENCE_BATCH_SIZE = 8

THRESHOLDS = [0, 25, 50, 100, 250, 500, 1000]

SUMMARY_CSV = OUTPUT_DIR / "Experiment_06C_threshold_summary.csv"
MEAN_CSV = OUTPUT_DIR / "Experiment_06C_mean_results.csv"
COMPONENT_CSV = OUTPUT_DIR / "Experiment_06C_component_analysis.csv"
BEST_JSON = OUTPUT_DIR / "Experiment_06C_best_threshold.json"
PLOT_PATH = OUTPUT_DIR / "Experiment_06C_threshold_performance.png"


# ============================================================
# 2. FILE HELPERS
# ============================================================

def is_nifti(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def find_patient_file(patient_dir: Path, kind: str) -> Path:
    files = [p for p in patient_dir.iterdir() if p.is_file() and is_nifti(p)]

    if kind == "ct":
        preferred = [
            "origional_images.nii", "origional_images.nii.gz",
            "original_images.nii", "original_images.nii.gz",
        ]
        for candidate in preferred:
            for p in files:
                if p.name.lower() == candidate:
                    return p
        for p in files:
            n = p.name.lower()
            if ("origional" in n or "original" in n) and "mask" not in n and "prediction" not in n:
                return p

    if kind == "mask":
        preferred = [
            "masked_images.nii", "masked_images.nii.gz",
            "mask_images.nii", "mask_images.nii.gz",
        ]
        for candidate in preferred:
            for p in files:
                if p.name.lower() == candidate:
                    return p
        for p in files:
            n = p.name.lower()
            if ("masked_images" in n or "mask_images" in n) and "prediction" not in n:
                return p

    raise FileNotFoundError(f"Could not find {kind} file in {patient_dir}")


# ============================================================
# 3. PREPROCESSING / METRICS
# ============================================================

def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct, HU_MIN, HU_MAX)
    return ((ct - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def range_text(indices: np.ndarray) -> str:
    if len(indices) == 0:
        return "None"
    return f"{int(indices.min())}-{int(indices.max())}"


def calculate_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict:
    pb = prediction.astype(bool)
    rb = reference.astype(bool)

    tp = int(np.logical_and(pb, rb).sum())
    fp = int(np.logical_and(pb, ~rb).sum())
    fn = int(np.logical_and(~pb, rb).sum())

    eps = 1e-8
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    ref_counts = reference.sum(axis=(0, 1))
    pred_counts = prediction.sum(axis=(0, 1))
    ref_pos = np.where(ref_counts > 0)[0]
    pred_pos = np.where(pred_counts > 0)[0]

    fp_slices = np.setdiff1d(pred_pos, ref_pos)
    missed = np.setdiff1d(ref_pos, pred_pos)

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
        "reference_slice_range": range_text(ref_pos),
        "predicted_slice_range": range_text(pred_pos),
        "reference_positive_slices": int(len(ref_pos)),
        "predicted_positive_slices": int(len(pred_pos)),
        "false_positive_slice_count": int(len(fp_slices)),
        "missed_reference_slice_count": int(len(missed)),
    }


# ============================================================
# 4. CONNECTED COMPONENT FILTERING
# ============================================================

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def label_components(mask: np.ndarray):
    labeled, n = ndimage.label(mask.astype(bool), structure=CONNECTIVITY_26)
    if n == 0:
        return labeled, np.array([], dtype=np.int64), 0
    sizes = np.bincount(labeled.ravel())[1:].astype(np.int64)
    return labeled, sizes, int(n)


def filter_by_min_size(mask: np.ndarray, threshold: int):
    if threshold == 0:
        _, sizes, n = label_components(mask)
        return mask.astype(np.uint8).copy(), n, n, sizes

    labeled, sizes, n = label_components(mask)

    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8), 0, 0, sizes

    keep_labels = np.where(sizes >= threshold)[0] + 1
    filtered = np.isin(labeled, keep_labels).astype(np.uint8)

    return filtered, n, int(len(keep_labels)), sizes


# ============================================================
# 5. LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Experiment 06A model not found:\n{MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("EXPERIMENT 06C — MINIMUM COMPONENT-SIZE FILTERING")
print("=" * 80)
print(f"Device: {device}")
print(f"Model:  {MODEL_PATH}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ============================================================
# 6. FULL-VOLUME INFERENCE
# ============================================================

def predict_volume(ct: np.ndarray) -> np.ndarray:
    pred = np.zeros(ct.shape, dtype=np.uint8)

    with torch.no_grad():
        for start in range(0, ct.shape[2], INFERENCE_BATCH_SIZE):
            end = min(start + INFERENCE_BATCH_SIZE, ct.shape[2])

            x_np = np.moveaxis(ct[:, :, start:end], 2, 0)
            x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(1).to(device)

            probs = torch.sigmoid(model(x))
            batch_pred = (
                probs[:, 0].cpu().numpy() > PRED_THRESHOLD
            ).astype(np.uint8)

            pred[:, :, start:end] = np.moveaxis(batch_pred, 0, 2)

    return pred


# ============================================================
# 7. GENERATE RAW VALIDATION PREDICTIONS ONCE
# ============================================================

patient_data = {}
component_rows = []

for patient in VALIDATION_PATIENTS:
    print(f"\nGenerating raw prediction for {patient}...")

    folder = BASE_DIR / patient
    ct_img = nib.load(str(find_patient_file(folder, "ct")))
    mask_img = nib.load(str(find_patient_file(folder, "mask")))

    raw_ct = ct_img.get_fdata(dtype=np.float32)
    ref = (mask_img.get_fdata() > 0).astype(np.uint8)

    if raw_ct.shape != ref.shape:
        raise ValueError(f"{patient}: shape mismatch")

    if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError(f"{patient}: affine mismatch")

    ct = preprocess_ct(raw_ct)
    raw_pred = predict_volume(ct)

    _, sizes, n_components = label_components(raw_pred)
    sorted_sizes = sorted(sizes.tolist(), reverse=True)

    spacing = ct_img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(spacing))

    print(f"  Components: {n_components}")
    print(f"  Top 10 sizes: {sorted_sizes[:10]}")

    for rank, size in enumerate(sorted_sizes, start=1):
        component_rows.append({
            "patient": patient,
            "rank": rank,
            "size_voxels": int(size),
            "volume_mm3": float(size * voxel_volume_mm3),
            "volume_ml": float(size * voxel_volume_mm3 / 1000.0),
        })

    patient_data[patient] = {
        "ct": ct,
        "reference": ref,
        "raw_prediction": raw_pred,
        "ct_img": ct_img,
    }

with COMPONENT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["patient", "rank", "size_voxels", "volume_mm3", "volume_ml"],
    )
    writer.writeheader()
    writer.writerows(component_rows)


# ============================================================
# 8. TEST ALL THRESHOLDS
# ============================================================

summary_rows = []

print("\n" + "=" * 80)
print("THRESHOLD TESTING")
print("=" * 80)

for threshold in THRESHOLDS:
    label = "RAW" if threshold == 0 else f">={threshold}_voxels"
    print(f"\nThreshold: {label}")

    for patient in VALIDATION_PATIENTS:
        ref = patient_data[patient]["reference"]
        raw_pred = patient_data[patient]["raw_prediction"]

        filtered, raw_n, kept_n, _ = filter_by_min_size(raw_pred, threshold)
        metrics = calculate_metrics(filtered, ref)

        row = {
            "threshold_voxels": threshold,
            "threshold_label": label,
            "patient": patient,
            "raw_component_count": raw_n,
            "retained_component_count": kept_n,
            **metrics,
        }
        summary_rows.append(row)

        print(
            f"  {patient}: Dice={metrics['dice']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"FP voxels={metrics['false_positive_voxels']}, "
            f"FP slices={metrics['false_positive_slice_count']}, "
            f"Pred range={metrics['predicted_slice_range']}, "
            f"Components kept={kept_n}"
        )

summary_fields = [
    "threshold_voxels", "threshold_label", "patient",
    "raw_component_count", "retained_component_count",
    "dice", "iou", "precision", "recall",
    "reference_voxels", "predicted_voxels",
    "true_positive_voxels", "false_positive_voxels",
    "false_negative_voxels", "reference_slice_range",
    "predicted_slice_range", "reference_positive_slices",
    "predicted_positive_slices", "false_positive_slice_count",
    "missed_reference_slice_count",
]

with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# 9. MEAN VALIDATION RESULTS
# ============================================================

mean_rows = []

for threshold in THRESHOLDS:
    rows = [r for r in summary_rows if r["threshold_voxels"] == threshold]

    mean_rows.append({
        "threshold_voxels": threshold,
        "threshold_label": "RAW" if threshold == 0 else f">={threshold}_voxels",
        "mean_dice": float(np.mean([r["dice"] for r in rows])),
        "mean_iou": float(np.mean([r["iou"] for r in rows])),
        "mean_precision": float(np.mean([r["precision"] for r in rows])),
        "mean_recall": float(np.mean([r["recall"] for r in rows])),
        "total_false_positive_voxels": int(sum(r["false_positive_voxels"] for r in rows)),
        "total_false_positive_slices": int(sum(r["false_positive_slice_count"] for r in rows)),
    })

with MEAN_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "threshold_voxels", "threshold_label",
            "mean_dice", "mean_iou", "mean_precision", "mean_recall",
            "total_false_positive_voxels", "total_false_positive_slices",
        ],
    )
    writer.writeheader()
    writer.writerows(mean_rows)

best = max(mean_rows, key=lambda r: r["mean_dice"])
best_threshold = int(best["threshold_voxels"])

with BEST_JSON.open("w", encoding="utf-8") as f:
    json.dump({
        "experiment": "06C",
        "selection_metric": "mean validation Dice",
        "best_threshold_voxels": best_threshold,
        "best_threshold_label": best["threshold_label"],
        "mean_validation_dice": best["mean_dice"],
        "mean_validation_precision": best["mean_precision"],
        "mean_validation_recall": best["mean_recall"],
        "validation_patients": VALIDATION_PATIENTS,
        "protected_final_test_patients": FINAL_TEST_PATIENTS,
        "note": "Freeze this threshold before evaluating Pat_13-Pat_15.",
    }, f, indent=2)


# ============================================================
# 10. SAVE BEST-THRESHOLD PREDICTIONS
# ============================================================

for patient in VALIDATION_PATIENTS:
    out_dir = OUTPUT_DIR / patient
    out_dir.mkdir(parents=True, exist_ok=True)

    ct = patient_data[patient]["ct"]
    ref = patient_data[patient]["reference"]
    raw_pred = patient_data[patient]["raw_prediction"]
    ct_img = patient_data[patient]["ct_img"]

    filtered, _, _, _ = filter_by_min_size(raw_pred, best_threshold)

    raw_metrics = calculate_metrics(raw_pred, ref)
    filt_metrics = calculate_metrics(filtered, ref)

    nib.save(
        nib.Nifti1Image(raw_pred.astype(np.uint8), ct_img.affine, ct_img.header),
        str(out_dir / f"{patient}_Experiment06A_raw_prediction.nii.gz"),
    )

    nib.save(
        nib.Nifti1Image(filtered.astype(np.uint8), ct_img.affine, ct_img.header),
        str(out_dir / f"{patient}_Experiment06C_best_filtered_prediction.nii.gz"),
    )

    ref_counts = ref.sum(axis=(0, 1))
    slice_idx = int(np.argmax(ref_counts))

    ct_slice = ct[:, :, slice_idx]
    ref_slice = ref[:, :, slice_idx]
    raw_slice = raw_pred[:, :, slice_idx]
    filt_slice = filtered[:, :, slice_idx]

    ref_overlay = np.ma.masked_where(ref_slice == 0, ref_slice)
    raw_overlay = np.ma.masked_where(raw_slice == 0, raw_slice)
    filt_overlay = np.ma.masked_where(filt_slice == 0, filt_slice)

    error_map = np.zeros_like(ref_slice, dtype=np.uint8)
    error_map[(ref_slice == 1) & (filt_slice == 1)] = 1
    error_map[(ref_slice == 0) & (filt_slice == 1)] = 2
    error_map[(ref_slice == 1) & (filt_slice == 0)] = 3
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
    axes[3].imshow(filt_overlay, alpha=0.55)
    axes[3].set_title(
        f"Filtered >= {best_threshold} voxels\n"
        f"Dice={filt_metrics['dice']:.4f}"
    )

    axes[4].imshow(ct_slice, cmap="gray")
    axes[4].imshow(error_overlay, alpha=0.65)
    axes[4].set_title("Filtered Error Map\n1=overlap, 2=FP, 3=FN")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{patient} - Experiment 06C\n"
        f"Raw Dice={raw_metrics['dice']:.4f} -> "
        f"Filtered Dice={filt_metrics['dice']:.4f}"
    )

    plt.tight_layout()
    plt.savefig(
        out_dir / f"{patient}_Experiment06C_best_threshold.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 11. PLOT THRESHOLD PERFORMANCE
# ============================================================

x = [r["threshold_voxels"] for r in mean_rows]
dice_y = [r["mean_dice"] for r in mean_rows]
prec_y = [r["mean_precision"] for r in mean_rows]
rec_y = [r["mean_recall"] for r in mean_rows]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, dice_y, marker="o", label="Mean Dice")
ax.plot(x, prec_y, marker="s", label="Mean Precision")
ax.plot(x, rec_y, marker="^", label="Mean Recall")
ax.set_xlabel("Minimum component size (voxels)\n0 = raw prediction")
ax.set_ylabel("Mean validation metric")
ax.set_title("Experiment 06C - Minimum Component Size Filtering")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 12. FINAL TABLE
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 06C MEAN VALIDATION RESULTS")
print("=" * 80)

print(
    f"{'Threshold':>14} | {'Mean Dice':>10} | "
    f"{'Precision':>10} | {'Recall':>10} | "
    f"{'FP voxels':>10} | {'FP slices':>9}"
)
print("-" * 80)

for row in mean_rows:
    print(
        f"{row['threshold_label']:>14} | "
        f"{row['mean_dice']:10.4f} | "
        f"{row['mean_precision']:10.4f} | "
        f"{row['mean_recall']:10.4f} | "
        f"{row['total_false_positive_voxels']:10d} | "
        f"{row['total_false_positive_slices']:9d}"
    )

print("\nBest threshold:", best["threshold_label"])
print(f"Best mean validation Dice: {best['mean_dice']:.4f}")
print(f"Summary CSV: {SUMMARY_CSV}")
print(f"Mean-results CSV: {MEAN_CSV}")
print(f"Component analysis: {COMPONENT_CSV}")
print(f"Best-threshold config: {BEST_JSON}")
print(f"Plot: {PLOT_PATH}")
