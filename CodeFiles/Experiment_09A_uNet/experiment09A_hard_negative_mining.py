# Experiment 09A — Offline Hard-Negative Mining
# Full script generated for the user's pancreas segmentation workflow.

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import UNet
from scipy import ndimage

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")
MODEL_PATH = Path(
    r"D:/Thesis Folder/Results/Experiment_08_results/"
    r"Experiment_08_2p5D_Three_Slice_UNet/"
    r"Experiment_08_best_model.pth"
)
OUTPUT_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_09A_results/"
    r"Experiment_09A_Hard_Negative_Mining"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATIENTS = ["Pat_01","Pat_03","Pat_04","Pat_05","Pat_06","Pat_07","Pat_09","Pat_10"]
VALIDATION_PATIENTS = ["Pat_11","Pat_12"]
FINAL_TEST_PATIENTS = ["Pat_13","Pat_14","Pat_15"]
EXCLUDED_PATIENTS = ["Pat_02","Pat_08"]

HU_MIN = -100.0
HU_MAX = 250.0
PREDICTION_THRESHOLD = 0.5
INFERENCE_BATCH_SIZE = 8
MIN_POSITIVE_PIXELS = 50
HARD_NEGATIVE_FRACTION_OF_TOTAL_NEGATIVES = 0.50
TOP_MONTAGE_SLICES_PER_PATIENT = 16
GLOBAL_TOP_MONTAGE_SLICES = 24

ALL_CANDIDATES_CSV = OUTPUT_DIR / "Experiment_09A_all_negative_candidates.csv"
SELECTED_HARD_NEGATIVES_CSV = OUTPUT_DIR / "Experiment_09A_selected_hard_negatives.csv"
MINING_SUMMARY_CSV = OUTPUT_DIR / "Experiment_09A_patient_summary.csv"
CONFIG_JSON = OUTPUT_DIR / "Experiment_09A_config.json"
TEXT_SUMMARY = OUTPUT_DIR / "Experiment_09A_summary.txt"

assert not (set(TRAIN_PATIENTS) & set(VALIDATION_PATIENTS))
assert not (set(TRAIN_PATIENTS) & set(FINAL_TEST_PATIENTS))
assert not (set(VALIDATION_PATIENTS) & set(FINAL_TEST_PATIENTS))
assert not (set(TRAIN_PATIENTS) & set(EXCLUDED_PATIENTS))

def is_nifti(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def find_patient_file(patient_dir: Path, kind: str) -> Path:
    files = [p for p in patient_dir.iterdir() if p.is_file() and is_nifti(p)]
    if kind == "ct":
        preferred = ["origional_images.nii","origional_images.nii.gz","original_images.nii","original_images.nii.gz"]
        for candidate in preferred:
            for p in files:
                if p.name.lower() == candidate:
                    return p
        for p in files:
            n = p.name.lower()
            if ("origional" in n or "original" in n) and "mask" not in n and "prediction" not in n:
                return p
    if kind == "mask":
        preferred = ["masked_images.nii","masked_images.nii.gz","mask_images.nii","mask_images.nii.gz"]
        for candidate in preferred:
            for p in files:
                if p.name.lower() == candidate:
                    return p
        for p in files:
            n = p.name.lower()
            if ("masked_images" in n or "mask_images" in n) and "prediction" not in n and "false_positive" not in n and "false_negative" not in n and "true_positive" not in n:
                return p
    raise FileNotFoundError(f"Could not find {kind} NIfTI file in:\n{patient_dir}")

def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct, HU_MIN, HU_MAX)
    return ((ct - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)

def load_patient(patient_name: str):
    patient_dir = BASE_DIR / patient_name
    ct_path = find_patient_file(patient_dir, "ct")
    mask_path = find_patient_file(patient_dir, "mask")
    ct_img = nib.load(str(ct_path))
    mask_img = nib.load(str(mask_path))
    raw_ct = ct_img.get_fdata(dtype=np.float32)
    raw_mask = mask_img.get_fdata()
    if raw_ct.shape != raw_mask.shape:
        raise ValueError(f"{patient_name}: CT/mask shape mismatch {raw_ct.shape} vs {raw_mask.shape}")
    if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError(f"{patient_name}: CT/mask affine mismatch.")
    orientation = nib.aff2axcodes(ct_img.affine)
    if orientation[2] not in ("S","I"):
        raise ValueError(f"{patient_name}: array axis 2 is not axial. Orientation={orientation}")
    return preprocess_ct(raw_ct), (raw_mask > 0).astype(np.uint8), ct_img, orientation

def build_three_slice_input(ct_volume: np.ndarray, center_index: int) -> np.ndarray:
    last = ct_volume.shape[2] - 1
    prev_i = max(0, center_index - 1)
    next_i = min(last, center_index + 1)
    return np.stack(
        [ct_volume[:,:,prev_i], ct_volume[:,:,center_index], ct_volume[:,:,next_i]],
        axis=0
    ).astype(np.float32)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Experiment 08 best model not found:\n{MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*88)
print("EXPERIMENT 09A — OFFLINE HARD-NEGATIVE MINING")
print("="*88)
print(f"Device: {device}")
print(f"Model:  {MODEL_PATH}")

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16,32,64,128),
    strides=(2,2,2),
    num_res_units=2,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Experiment 08 best model loaded successfully.")

CONNECTIVITY_8 = np.ones((3,3), dtype=np.uint8)

def two_dimensional_component_metrics(binary_prediction: np.ndarray):
    labeled, n = ndimage.label(binary_prediction.astype(bool), structure=CONNECTIVITY_8)
    if n == 0:
        return 0, 0
    sizes = np.bincount(labeled.ravel())[1:]
    return int(n), int(sizes.max())

def score_patient_negatives(patient_name: str):
    ct, mask, _, orientation = load_patient(patient_name)
    foreground_per_slice = np.sum(mask, axis=(0,1))
    positive_indices = np.where(foreground_per_slice > MIN_POSITIVE_PIXELS)[0]
    empty_indices = np.where(foreground_per_slice == 0)[0]
    if len(positive_indices) == 0:
        raise ValueError(f"{patient_name}: no meaningful positive slices.")
    desired_total_negatives = int(round(len(positive_indices)))
    desired_hard = int(round(desired_total_negatives * HARD_NEGATIVE_FRACTION_OF_TOTAL_NEGATIVES))
    rows = []
    with torch.no_grad():
        for start in range(0, len(empty_indices), INFERENCE_BATCH_SIZE):
            batch_slice_indices = empty_indices[start:start+INFERENCE_BATCH_SIZE]
            batch_inputs = np.stack(
                [build_three_slice_input(ct, int(idx)) for idx in batch_slice_indices],
                axis=0
            )
            batch_tensor = torch.from_numpy(batch_inputs.astype(np.float32)).to(device)
            probs = torch.sigmoid(model(batch_tensor))[:,0].cpu().numpy()
            for local_idx, center_index in enumerate(batch_slice_indices):
                pmap = probs[local_idx]
                binary = (pmap > PREDICTION_THRESHOLD).astype(np.uint8)
                predicted_pixels = int(binary.sum())
                n_comp, largest_comp = two_dimensional_component_metrics(binary)
                rows.append({
                    "patient": patient_name,
                    "slice_index": int(center_index),
                    "orientation": "".join(orientation),
                    "reference_pixels": 0,
                    "predicted_pixels": predicted_pixels,
                    "predicted_fraction": float(predicted_pixels / binary.size),
                    "max_probability": float(pmap.max()),
                    "mean_probability": float(pmap.mean()),
                    "sum_probability": float(pmap.sum()),
                    "num_2d_components": n_comp,
                    "largest_2d_component_pixels": largest_comp,
                    "meaningful_positive_slices_in_patient": int(len(positive_indices)),
                    "desired_hard_negatives_for_09B": int(desired_hard),
                })
    rows.sort(
        key=lambda r: (r["predicted_pixels"], r["largest_2d_component_pixels"], r["sum_probability"], r["max_probability"]),
        reverse=True
    )
    for rank, row in enumerate(rows, start=1):
        row["hardness_rank_within_patient"] = rank

    nonzero = [r for r in rows if r["predicted_pixels"] > 0]
    selected = nonzero[:desired_hard]

    if len(selected) < desired_hard:
        selected_keys = {(r["patient"], r["slice_index"]) for r in selected}
        remaining = [r for r in rows if (r["patient"], r["slice_index"]) not in selected_keys]
        remaining.sort(
            key=lambda r: (r["max_probability"], r["sum_probability"], r["mean_probability"]),
            reverse=True
        )
        selected.extend(remaining[:desired_hard-len(selected)])

    selected_keys = {(r["patient"], r["slice_index"]) for r in selected}
    for selection_rank, row in enumerate(selected, start=1):
        row["selected_for_09B"] = 1
        row["selected_hard_rank"] = selection_rank
    for row in rows:
        if (row["patient"], row["slice_index"]) not in selected_keys:
            row["selected_for_09B"] = 0
            row["selected_hard_rank"] = ""

    summary = {
        "patient": patient_name,
        "meaningful_positive_slices": int(len(positive_indices)),
        "strictly_empty_negative_candidates": int(len(empty_indices)),
        "negative_candidates_with_nonzero_prediction": int(sum(r["predicted_pixels"] > 0 for r in rows)),
        "desired_total_negatives_for_09B": int(desired_total_negatives),
        "selected_hard_negatives_for_09B": int(len(selected)),
        "largest_false_positive_area_pixels": int(rows[0]["predicted_pixels"] if rows else 0),
        "largest_false_positive_component_pixels": int(max((r["largest_2d_component_pixels"] for r in rows), default=0)),
        "maximum_negative_slice_probability": float(max((r["max_probability"] for r in rows), default=0.0)),
        "mean_predicted_pixels_across_negative_candidates": float(np.mean([r["predicted_pixels"] for r in rows]) if rows else 0.0),
        "median_predicted_pixels_across_negative_candidates": float(np.median([r["predicted_pixels"] for r in rows]) if rows else 0.0),
        "mean_predicted_pixels_selected_hard_negatives": float(np.mean([r["predicted_pixels"] for r in selected]) if selected else 0.0),
    }
    return ct, rows, selected, summary

def save_patient_montage(patient_name, ct, selected_rows, output_path):
    rows_to_show = selected_rows[:TOP_MONTAGE_SLICES_PER_PATIENT]
    if not rows_to_show:
        return
    cols = 4
    rows_n = math.ceil(len(rows_to_show)/cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4*cols, 4*rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, row in zip(axes, rows_to_show):
        s = int(row["slice_index"])
        ax.imshow(ct[:,:,s], cmap="gray")
        ax.set_title(
            f"#{row['selected_hard_rank']} slice {s}\n"
            f"FP={row['predicted_pixels']} | comp={row['largest_2d_component_pixels']}\n"
            f"max p={row['max_probability']:.3f}"
        )
        ax.axis("off")
    fig.suptitle(f"{patient_name} — Experiment 09A Top Mined Hard Negatives", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

def save_patient_score_plot(patient_name, rows, selected_rows, output_path):
    x = np.arange(1, len(rows)+1)
    y = np.array([r["predicted_pixels"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(11,6))
    ax.plot(x, y, marker="o", markersize=2)
    if selected_rows:
        ax.axvline(len(selected_rows), linestyle="--", label="09B hard-negative selection cutoff")
        ax.legend()
    ax.set_xlabel("Negative slice hardness rank")
    ax.set_ylabel("False-positive predicted pixels")
    ax.set_title(f"{patient_name} — Experiment 08 Errors on Training Negative Slices")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_global_montage(global_selected, patient_ct_cache, output_path):
    rows_to_show = sorted(
        global_selected,
        key=lambda r: (r["predicted_pixels"], r["largest_2d_component_pixels"], r["sum_probability"]),
        reverse=True
    )[:GLOBAL_TOP_MONTAGE_SLICES]
    if not rows_to_show:
        return
    cols = 4
    rows_n = math.ceil(len(rows_to_show)/cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4*cols, 4*rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for global_rank, (ax, row) in enumerate(zip(axes, rows_to_show), start=1):
        p = row["patient"]
        s = int(row["slice_index"])
        ax.imshow(patient_ct_cache[p][:,:,s], cmap="gray")
        ax.set_title(
            f"Global #{global_rank}\n{p}, slice {s}\n"
            f"FP={row['predicted_pixels']} | comp={row['largest_2d_component_pixels']} | "
            f"max p={row['max_probability']:.3f}"
        )
        ax.axis("off")
    fig.suptitle("Experiment 09A — Most Confusing Negative Training Slices", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

all_candidate_rows = []
all_selected_rows = []
patient_summary_rows = []
patient_ct_cache = {}

print("\n" + "="*88)
print("MINING TRAINING-COHORT HARD NEGATIVES")
print("="*88)

for patient_name in TRAIN_PATIENTS:
    print(f"\nMining {patient_name}...")
    ct, candidate_rows, selected_rows, patient_summary = score_patient_negatives(patient_name)
    patient_ct_cache[patient_name] = ct
    all_candidate_rows.extend(candidate_rows)
    all_selected_rows.extend(selected_rows)
    patient_summary_rows.append(patient_summary)

    print(f"  Positive slices: {patient_summary['meaningful_positive_slices']}")
    print(f"  Empty negative candidates: {patient_summary['strictly_empty_negative_candidates']}")
    print(f"  Negative slices producing FP masks: {patient_summary['negative_candidates_with_nonzero_prediction']}")
    print(f"  Selected hard negatives: {patient_summary['selected_hard_negatives_for_09B']}")
    print(f"  Largest FP area: {patient_summary['largest_false_positive_area_pixels']} pixels")
    if selected_rows:
        print("  Top 5 selected:")
        for row in selected_rows[:5]:
            print(
                f"    slice {row['slice_index']}: FP={row['predicted_pixels']}, "
                f"largest component={row['largest_2d_component_pixels']}, "
                f"max p={row['max_probability']:.3f}"
            )

    patient_dir = OUTPUT_DIR / patient_name
    patient_dir.mkdir(parents=True, exist_ok=True)
    save_patient_montage(
        patient_name, ct, selected_rows,
        patient_dir / f"{patient_name}_top_hard_negatives.png"
    )
    save_patient_score_plot(
        patient_name, candidate_rows, selected_rows,
        patient_dir / f"{patient_name}_hardness_rank_plot.png"
    )

all_candidate_fields = [
    "patient","slice_index","orientation","reference_pixels","predicted_pixels",
    "predicted_fraction","max_probability","mean_probability","sum_probability",
    "num_2d_components","largest_2d_component_pixels",
    "meaningful_positive_slices_in_patient","desired_hard_negatives_for_09B",
    "hardness_rank_within_patient","selected_for_09B","selected_hard_rank"
]

with ALL_CANDIDATES_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_candidate_fields)
    writer.writeheader()
    writer.writerows(all_candidate_rows)

with SELECTED_HARD_NEGATIVES_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_candidate_fields)
    writer.writeheader()
    writer.writerows(sorted(all_selected_rows, key=lambda r: (r["patient"], r["selected_hard_rank"])))

summary_fields = list(patient_summary_rows[0].keys())
with MINING_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(patient_summary_rows)

save_global_montage(
    all_selected_rows,
    patient_ct_cache,
    OUTPUT_DIR / "Experiment_09A_global_top_hard_negatives.png"
)

with CONFIG_JSON.open("w", encoding="utf-8") as f:
    json.dump({
        "experiment": "09A",
        "name": "Offline hard-negative mining",
        "source_model": str(MODEL_PATH),
        "training_patients_used_for_mining": TRAIN_PATIENTS,
        "validation_patients_not_used_for_mining": VALIDATION_PATIENTS,
        "final_test_patients_untouched": FINAL_TEST_PATIENTS,
        "excluded_patients": EXCLUDED_PATIENTS,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "minimum_positive_pixels": MIN_POSITIVE_PIXELS,
        "future_09B_hard_negative_fraction": HARD_NEGATIVE_FRACTION_OF_TOTAL_NEGATIVES,
        "primary_hardness_metric": "predicted_pixels",
        "tie_break_metrics": [
            "largest_2d_component_pixels",
            "sum_probability",
            "max_probability"
        ],
        "note": "Experiment 09A performs mining only. No retraining and no validation/test leakage."
    }, f, indent=2)

total_positive_slices = int(sum(r["meaningful_positive_slices"] for r in patient_summary_rows))
total_negative_candidates = int(sum(r["strictly_empty_negative_candidates"] for r in patient_summary_rows))
total_fp_negative_slices = int(sum(r["negative_candidates_with_nonzero_prediction"] for r in patient_summary_rows))
total_selected_hard_negatives = len(all_selected_rows)

summary_lines = [
    "="*88,
    "EXPERIMENT 09A — OFFLINE HARD-NEGATIVE MINING SUMMARY",
    "="*88,
    "",
    f"Source model: {MODEL_PATH}",
    "",
    "Mining cohort:",
    ", ".join(TRAIN_PATIENTS),
    "",
    "Validation patients NOT used:",
    ", ".join(VALIDATION_PATIENTS),
    "",
    "Final test patients untouched:",
    ", ".join(FINAL_TEST_PATIENTS),
    "",
    f"Total meaningful positive slices: {total_positive_slices}",
    f"Total strictly empty negative candidates: {total_negative_candidates}",
    f"Negative candidates producing non-zero false-positive predictions: {total_fp_negative_slices}",
    f"Selected model-mined hard negatives for future 09B: {total_selected_hard_negatives}",
    "",
    "Per-patient summary:"
]
for row in patient_summary_rows:
    summary_lines.append(
        f"{row['patient']}: positive={row['meaningful_positive_slices']}, "
        f"negative candidates={row['strictly_empty_negative_candidates']}, "
        f"FP-producing negatives={row['negative_candidates_with_nonzero_prediction']}, "
        f"selected hard={row['selected_hard_negatives_for_09B']}, "
        f"largest FP area={row['largest_false_positive_area_pixels']}"
    )
summary_lines += [
    "",
    "NEXT:",
    "Inspect the patient montages and global montage BEFORE training Experiment 09B.",
    "If selected slices are sensible hard negatives, freeze this CSV and use it for 09B.",
    "",
    "Pat_11/Pat_12 were not used for mining.",
    "Pat_13/Pat_14/Pat_15 remain untouched."
]
TEXT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n" + "="*88)
print("EXPERIMENT 09A MINING COMPLETE")
print("="*88)
print(f"Total meaningful positive slices: {total_positive_slices}")
print(f"Total empty negative candidates: {total_negative_candidates}")
print(f"Negative candidates producing non-zero FP predictions: {total_fp_negative_slices}")
print(f"Selected hard negatives for 09B: {total_selected_hard_negatives}")
print(f"\nAll candidates:\n{ALL_CANDIDATES_CSV}")
print(f"\nSelected hard negatives:\n{SELECTED_HARD_NEGATIVES_CSV}")
print(f"\nPatient summary:\n{MINING_SUMMARY_CSV}")
print(f"\nGlobal montage:\n{OUTPUT_DIR / 'Experiment_09A_global_top_hard_negatives.png'}")
print(f"\nReadable summary:\n{TEXT_SUMMARY}")