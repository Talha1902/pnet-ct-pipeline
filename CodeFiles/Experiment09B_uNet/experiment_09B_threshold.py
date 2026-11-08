"""
Experiment 09B-T — Validation Threshold Sensitivity Analysis
============================================================

Purpose
-------
Experiment 09B reduced false-positive predictions but also strongly reduced
recall at the default threshold of 0.50.

Before retraining another model, this script checks whether Experiment 09B
mainly changed probability calibration. It loads the BEST 09B checkpoint,
runs whole-volume inference ONCE for Pat_11 and Pat_12, then evaluates a range
of probability thresholds.

NO retraining is performed.
Pat_13, Pat_14, and Pat_15 remain untouched.

Outputs
-------
- Experiment_09B_threshold_results.csv
- Experiment_09B_threshold_mean_summary.csv
- Experiment_09B_threshold_sweep.png
- Experiment_09B_threshold_summary.txt

Run:
    python experiment_09B_threshold_sweep.py
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
# CONFIG
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")

MODEL_PATH = Path(
    r"D:/Thesis Folder/Results/Experiment_09B_results/"
    r"Experiment_09B_2p5D_Hard_Negative_Mining/"
    r"Experiment_09B_best_model.pth"
)

OUTPUT_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_09B_results/"
    r"Experiment_09B_2p5D_Hard_Negative_Mining/"
    r"Threshold_Sensitivity"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_PATIENTS = ["Pat_11", "Pat_12"]

HU_MIN = -100.0
HU_MAX = 250.0
INFERENCE_BATCH_SIZE = 8

THRESHOLDS = np.round(
    np.arange(0.05, 0.751, 0.05),
    2
)

DETAIL_CSV = OUTPUT_DIR / "Experiment_09B_threshold_results.csv"
MEAN_CSV = OUTPUT_DIR / "Experiment_09B_threshold_mean_summary.csv"
PLOT_PNG = OUTPUT_DIR / "Experiment_09B_threshold_sweep.png"
SUMMARY_TXT = OUTPUT_DIR / "Experiment_09B_threshold_summary.txt"


# ============================================================
# FILE HELPERS
# ============================================================

def is_nifti(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def find_patient_file(patient_dir: Path, kind: str) -> Path:
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
            n = p.name.lower()
            if (
                ("origional" in n or "original" in n)
                and "mask" not in n
                and "prediction" not in n
            ):
                return p

    if kind == "mask":
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
            n = p.name.lower()
            if (
                ("masked_images" in n or "mask_images" in n)
                and "prediction" not in n
                and "false_positive" not in n
                and "false_negative" not in n
                and "true_positive" not in n
            ):
                return p

    raise FileNotFoundError(
        f"Could not find {kind} file inside:\n{patient_dir}"
    )


def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct, HU_MIN, HU_MAX)
    return ((ct - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def load_patient(patient_name: str):
    patient_dir = BASE_DIR / patient_name

    ct_img = nib.load(
        str(find_patient_file(patient_dir, "ct"))
    )
    mask_img = nib.load(
        str(find_patient_file(patient_dir, "mask"))
    )

    raw_ct = ct_img.get_fdata(dtype=np.float32)
    raw_mask = mask_img.get_fdata()

    if raw_ct.shape != raw_mask.shape:
        raise ValueError(
            f"{patient_name}: shape mismatch "
            f"{raw_ct.shape} vs {raw_mask.shape}"
        )

    if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError(
            f"{patient_name}: affine mismatch."
        )

    orientation = nib.aff2axcodes(ct_img.affine)

    if orientation[2] not in ("S", "I"):
        raise ValueError(
            f"{patient_name}: expected axial array axis 2. "
            f"Orientation={orientation}"
        )

    return (
        preprocess_ct(raw_ct),
        (raw_mask > 0).astype(np.uint8),
    )


# ============================================================
# 2.5D INPUT + MODEL
# ============================================================

def build_three_slice_input(ct, z):
    last = ct.shape[2] - 1
    return np.stack(
        [
            ct[:, :, max(0, z - 1)],
            ct[:, :, z],
            ct[:, :, min(last, z + 1)],
        ],
        axis=0,
    ).astype(np.float32)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Best Experiment 09B model not found:\n{MODEL_PATH}"
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
model.eval()

print("=" * 84)
print("EXPERIMENT 09B-T — THRESHOLD SENSITIVITY")
print("=" * 84)
print(f"Device: {device}")
print(f"Model:  {MODEL_PATH}")


# ============================================================
# PROBABILITY INFERENCE
# ============================================================

def predict_probability_volume(ct):
    probability = np.zeros(
        ct.shape,
        dtype=np.float32,
    )

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

            batch = np.stack(
                [
                    build_three_slice_input(ct, z)
                    for z in range(start, end)
                ],
                axis=0,
            )

            tensor = torch.from_numpy(
                batch
            ).to(device)

            probs = torch.sigmoid(
                model(tensor)
            )[:, 0].cpu().numpy()

            probability[:, :, start:end] = np.moveaxis(
                probs,
                0,
                2,
            )

    return probability


# ============================================================
# METRICS
# ============================================================

def range_text(indices):
    if len(indices) == 0:
        return "None"
    return f"{int(indices.min())}-{int(indices.max())}"


def evaluate(prediction, reference):
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

    fp_slices = np.setdiff1d(
        pred_pos,
        ref_pos,
    )

    missed = np.setdiff1d(
        ref_pos,
        pred_pos,
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_voxels": int(prediction.sum()),
        "true_positive_voxels": tp,
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "predicted_slice_range": range_text(pred_pos),
        "false_positive_slices": int(len(fp_slices)),
        "missed_reference_slices": int(len(missed)),
    }


# ============================================================
# RUN
# ============================================================

patient_data = {}

for patient_name in VAL_PATIENTS:
    print(f"\nRunning probability inference for {patient_name}...")

    ct, reference = load_patient(patient_name)
    probability = predict_probability_volume(ct)

    patient_data[patient_name] = {
        "reference": reference,
        "probability": probability,
    }

    print(
        f"  Probability range: "
        f"{probability.min():.4f} to {probability.max():.4f}"
    )

detail_rows = []
mean_rows = []

for threshold in THRESHOLDS:

    threshold_patient_results = []

    for patient_name in VAL_PATIENTS:

        reference = patient_data[
            patient_name
        ]["reference"]

        probability = patient_data[
            patient_name
        ]["probability"]

        prediction = (
            probability >= threshold
        ).astype(np.uint8)

        result = evaluate(
            prediction,
            reference,
        )

        detail_rows.append(
            {
                "threshold": threshold,
                "patient": patient_name,
                **result,
            }
        )

        threshold_patient_results.append(
            result
        )

    mean_row = {
        "threshold":
            threshold,

        "mean_dice":
            float(
                np.mean(
                    [r["dice"] for r in threshold_patient_results]
                )
            ),

        "mean_precision":
            float(
                np.mean(
                    [r["precision"] for r in threshold_patient_results]
                )
            ),

        "mean_recall":
            float(
                np.mean(
                    [r["recall"] for r in threshold_patient_results]
                )
            ),

        "total_false_positive_voxels":
            int(
                sum(
                    r["false_positive_voxels"]
                    for r in threshold_patient_results
                )
            ),

        "total_false_positive_slices":
            int(
                sum(
                    r["false_positive_slices"]
                    for r in threshold_patient_results
                )
            ),

        "total_missed_reference_slices":
            int(
                sum(
                    r["missed_reference_slices"]
                    for r in threshold_patient_results
                )
            ),
    }

    mean_rows.append(mean_row)


# ============================================================
# SAVE TABLES
# ============================================================

with DETAIL_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(detail_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(detail_rows)


with MEAN_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(mean_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(mean_rows)


# ============================================================
# BEST THRESHOLD + CONSOLE
# ============================================================

best_mean = max(
    mean_rows,
    key=lambda row: row["mean_dice"],
)

best_threshold = float(
    best_mean["threshold"]
)

print("\n" + "=" * 84)
print("THRESHOLD RESULTS")
print("=" * 84)

for row in mean_rows:
    print(
        f"Threshold {row['threshold']:.2f}: "
        f"Mean Dice={row['mean_dice']:.4f}, "
        f"Precision={row['mean_precision']:.4f}, "
        f"Recall={row['mean_recall']:.4f}, "
        f"FP slices={row['total_false_positive_slices']}, "
        f"Missed ref slices={row['total_missed_reference_slices']}"
    )

print("\nBEST MEAN-DICE THRESHOLD")
print(
    f"Threshold:  {best_threshold:.2f}"
)
print(
    f"Mean Dice:  {best_mean['mean_dice']:.4f}"
)
print(
    f"Precision:  {best_mean['mean_precision']:.4f}"
)
print(
    f"Recall:     {best_mean['mean_recall']:.4f}"
)

print("\nPer-patient results at best threshold:")

for row in detail_rows:
    if abs(
        float(row["threshold"])
        - best_threshold
    ) < 1e-9:

        print(
            f"  {row['patient']}: "
            f"Dice={row['dice']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"FP voxels={row['false_positive_voxels']}, "
            f"FP slices={row['false_positive_slices']}, "
            f"Pred range={row['predicted_slice_range']}, "
            f"Missed ref={row['missed_reference_slices']}"
        )


# ============================================================
# PLOT
# ============================================================

x = [row["threshold"] for row in mean_rows]
dice_y = [row["mean_dice"] for row in mean_rows]
precision_y = [row["mean_precision"] for row in mean_rows]
recall_y = [row["mean_recall"] for row in mean_rows]

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    x,
    dice_y,
    marker="o",
    label="Mean Dice",
)

ax.plot(
    x,
    precision_y,
    marker="s",
    label="Mean Precision",
)

ax.plot(
    x,
    recall_y,
    marker="^",
    label="Mean Recall",
)

ax.axvline(
    0.50,
    linestyle="--",
    label="Original threshold 0.50",
)

ax.axvline(
    best_threshold,
    linestyle=":",
    label=f"Best mean-Dice threshold {best_threshold:.2f}",
)

ax.set_xlabel(
    "Prediction threshold"
)

ax.set_ylabel(
    "Mean validation metric"
)

ax.set_title(
    "Experiment 09B-T — Validation Threshold Sensitivity"
)

ax.grid(True)
ax.legend()

plt.tight_layout()

plt.savefig(
    PLOT_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# SUMMARY TXT
# ============================================================

best_detail = [
    row
    for row in detail_rows
    if abs(
        float(row["threshold"])
        - best_threshold
    ) < 1e-9
]

lines = [
    "=" * 84,
    "EXPERIMENT 09B-T — THRESHOLD SENSITIVITY SUMMARY",
    "=" * 84,
    "",
    f"Best threshold by mean validation Dice: {best_threshold:.2f}",
    f"Mean Dice: {best_mean['mean_dice']:.4f}",
    f"Mean Precision: {best_mean['mean_precision']:.4f}",
    f"Mean Recall: {best_mean['mean_recall']:.4f}",
    "",
    "Per-patient results:",
]

for row in best_detail:
    lines.append(
        f"{row['patient']}: "
        f"Dice={row['dice']:.4f}, "
        f"Precision={row['precision']:.4f}, "
        f"Recall={row['recall']:.4f}, "
        f"FP voxels={row['false_positive_voxels']}, "
        f"FP slices={row['false_positive_slices']}, "
        f"missed reference slices={row['missed_reference_slices']}"
    )

lines += [
    "",
    "Reference benchmarks:",
    "Experiment 08 @ threshold 0.50: mean Dice=0.3079",
    "  Pat_11 Dice=0.0823, Precision=0.0792, Recall=0.0856, FP slices=228",
    "  Pat_12 Dice=0.5336, Precision=0.6058, Recall=0.4768, FP slices=111",
    "",
    "Experiment 09B @ threshold 0.50: mean Dice=0.2314",
    "  Pat_11 Dice=0.0043, Precision=0.0114, Recall=0.0027, FP slices=108",
    "  Pat_12 Dice=0.4585, Precision=0.6306, Recall=0.3602, FP slices=108",
    "",
    "Pat_13/Pat_14/Pat_15 remain untouched.",
]

SUMMARY_TXT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print(f"\nDetail CSV:\n{DETAIL_CSV}")
print(f"\nMean summary CSV:\n{MEAN_CSV}")
print(f"\nPlot:\n{PLOT_PNG}")
print(f"\nSummary:\n{SUMMARY_TXT}")
