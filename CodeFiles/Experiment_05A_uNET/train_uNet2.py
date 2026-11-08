
"""
Experiment 05A - Improved 2D U-Net pancreas segmentation training without Pat_02.

Main changes compared with the first experiment:
1. Patient-level split (NO slice leakage)
2. Train / validation / test patients are fixed and saved
3. Training includes pancreas-positive slices + selected negative slices
4. Uses fixed HU normalization instead of per-patient min/max normalization
5. Uses Dice + BCE loss (better when negative/background slices are included)
6. Computes WHOLE-VOLUME validation Dice on validation patients
7. Saves the BEST model according to validation Dice, not simply the last epoch
8. Saves training history and split information for thesis/reproducibility
9. Uses deterministic random seeds

IMPORTANT:
- Pat_11 and Pat_12 are reserved for FINAL TESTING and are never loaded here.
- Do not look at final test performance while tuning this model.
- These masks are TotalSegmentator-generated reference masks unless radiologist validated.

Install if needed:
    pip install torch monai nibabel numpy matplotlib

Run:
    python train_unet_v2.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path (r"D:/Thesis Folder/Training_dataset")
Results = Path (r"D:/Thesis Folder/Results/Experiment_05A_results")
EXPERIMENT_DIR = Results / "Experiment_05A_uNET_2D_Improved_without_Pat_02"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed patient-level split.
TRAIN_PATIENTS = [
    "Pat_01",
    "Pat_03",
    "Pat_04",
    "Pat_05",
    "Pat_06",
    "Pat_07",
    "Pat_08",
    "Pat_09",
    "Pat_10",
]
VAL_PATIENTS = ["Pat_11", "Pat_12"]

# These are NOT used during training.
FINAL_TEST_PATIENTS = ["Pat_13", "Pat_14", "Pat_15"]

# If you later decide that a patient's mask is unreliable, add it here
# AND remove it from the relevant split above.
EXCLUDED_PATIENTS = ["Pat_02"]   #due to poor quality of the masks, these patients are excluded from training and validation.  

SEED = 42
EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 1e-3

# Validate whole 3D volumes every N epochs.
VAL_EVERY = 2

# Threshold used to convert probability -> binary mask.
PREDICTION_THRESHOLD = 0.5

# CT soft-tissue window used in your first experiment.
HU_MIN = -100.0
HU_MAX = 250.0

# Positive training slice definition.
# This avoids very tiny edge masks.
MIN_POSITIVE_PIXELS = 50

# Number of negative slices relative to positive slices.
# 0.50 means ~1 negative for every 2 positive slices.
NEGATIVE_TO_POSITIVE_RATIO = 0.50

# We prefer some negatives close to the pancreas because they are harder and
# more useful than completely unrelated superior/inferior slices.
NEARBY_NEGATIVE_MARGIN = 20

# Early stopping is based on WHOLE-VOLUME validation Dice.
# Patience counts validation checks, not epochs.
EARLY_STOPPING_PATIENCE = 8

# Windows + CPU friendly.
NUM_WORKERS = 0

BEST_MODEL_PATH = EXPERIMENT_DIR / "best_pancreas_unet_v2.pth"
LAST_MODEL_PATH = EXPERIMENT_DIR / "last_pancreas_unet_v2.pth"
HISTORY_CSV = EXPERIMENT_DIR / "training_history.csv"
SPLIT_JSON = EXPERIMENT_DIR / "dataset_split.json"
CURVE_PNG = EXPERIMENT_DIR / "training_validation_curve.png"
CONFIG_JSON = EXPERIMENT_DIR / "experiment_config.json"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. FILE FINDING
# ============================================================

def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def find_patient_file(patient_dir: Path, kind: str) -> Path:
    """
    Robustly find the original CT or reference mask.

    Preferred standardized names:
        Origional_images.nii / .nii.gz
        masked_images.nii / .nii.gz

    It also tolerates capitalization and the common 'Original' spelling.
    """
    files = [p for p in patient_dir.iterdir() if p.is_file() and is_nifti(p)]

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

        # Fallback: find a NIfTI containing original/origional,
        # but avoid prediction/error files.
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
        f"Could not find {kind} NIfTI file inside:\n{patient_dir}"
    )


# ============================================================
# 4. PREPROCESSING
# ============================================================

def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    """
    Fixed CT normalization.

    -100 HU -> 0
     250 HU -> 1

    This is more consistent across patients than calculating each patient's
    own minimum and maximum after clipping.
    """
    ct = np.clip(ct, HU_MIN, HU_MAX)
    ct = (ct - HU_MIN) / (HU_MAX - HU_MIN)
    return ct.astype(np.float32)


def load_patient(patient_name: str):
    patient_dir = BASE_DIR / patient_name

    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient folder not found: {patient_dir}")

    ct_path = find_patient_file(patient_dir, "ct")
    mask_path = find_patient_file(patient_dir, "mask")

    ct_img = nib.load(str(ct_path))
    mask_img = nib.load(str(mask_path))

    ct = ct_img.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata()

    if ct.shape != mask.shape:
        raise ValueError(
            f"{patient_name}: CT/mask shape mismatch: "
            f"{ct.shape} vs {mask.shape}"
        )

    if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError(f"{patient_name}: CT/mask affine mismatch.")

    mask = (mask > 0).astype(np.uint8)
    ct = preprocess_ct(ct)

    return ct, mask, ct_img.affine


# ============================================================
# 5. SELECT TRAINING SLICES
# ============================================================

def select_training_indices(mask: np.ndarray, rng: np.random.Generator):
    """
    Use:
    - all meaningful pancreas-positive slices
    - selected negative slices

    Negative slices teach the network when NOT to predict pancreas.
    """
    foreground_per_slice = np.sum(mask, axis=(0, 1))

    positive_indices = np.where(
        foreground_per_slice > MIN_POSITIVE_PIXELS
    )[0]

    if len(positive_indices) == 0:
        raise ValueError("No positive pancreas slices found.")

    all_negative_indices = np.where(foreground_per_slice == 0)[0]

    desired_negative_count = int(
        round(len(positive_indices) * NEGATIVE_TO_POSITIVE_RATIO)
    )

    if desired_negative_count == 0 or len(all_negative_indices) == 0:
        return positive_indices.tolist(), []

    first_positive = int(positive_indices.min())
    last_positive = int(positive_indices.max())

    nearby_low = max(0, first_positive - NEARBY_NEGATIVE_MARGIN)
    nearby_high = min(mask.shape[2] - 1, last_positive + NEARBY_NEGATIVE_MARGIN)

    nearby_negative_indices = all_negative_indices[
        (all_negative_indices >= nearby_low)
        & (all_negative_indices <= nearby_high)
    ]

    far_negative_indices = np.setdiff1d(
        all_negative_indices,
        nearby_negative_indices,
    )

    # Try to obtain about half the negatives from nearby slices.
    desired_nearby = min(
        len(nearby_negative_indices),
        max(1, desired_negative_count // 2),
    )

    selected_nearby = (
        rng.choice(
            nearby_negative_indices,
            size=desired_nearby,
            replace=False,
        )
        if desired_nearby > 0
        else np.array([], dtype=int)
    )

    remaining = desired_negative_count - len(selected_nearby)

    available_pool = np.setdiff1d(
        all_negative_indices,
        selected_nearby,
    )

    remaining = min(remaining, len(available_pool))

    selected_remaining = (
        rng.choice(
            available_pool,
            size=remaining,
            replace=False,
        )
        if remaining > 0
        else np.array([], dtype=int)
    )

    negative_indices = np.sort(
        np.concatenate([selected_nearby, selected_remaining])
    )

    return positive_indices.tolist(), negative_indices.astype(int).tolist()


# ============================================================
# 6. BUILD MEMORY-EFFICIENT 2D TRAINING DATASET
# ============================================================

class PancreasSliceDataset(Dataset):
    def __init__(self, images, masks, metadata):
        # Keep images in float16 and masks uint8 to reduce RAM.
        self.images = np.asarray(images, dtype=np.float16)
        self.masks = np.asarray(masks, dtype=np.uint8)
        self.metadata = metadata

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(
            self.images[index].astype(np.float32)
        ).unsqueeze(0)

        mask = torch.from_numpy(
            self.masks[index].astype(np.float32)
        ).unsqueeze(0)

        return image, mask


def build_training_dataset(patient_names):
    images = []
    masks = []
    metadata = []

    rng = np.random.default_rng(SEED)

    print("\nBuilding TRAINING slice dataset...")

    total_positive = 0
    total_negative = 0

    for patient_name in patient_names:
        if patient_name in EXCLUDED_PATIENTS:
            print(f"Skipping excluded patient: {patient_name}")
            continue

        ct, mask, _ = load_patient(patient_name)

        positive_indices, negative_indices = select_training_indices(
            mask,
            rng,
        )

        print(
            f"{patient_name}: "
            f"{len(positive_indices)} positive + "
            f"{len(negative_indices)} negative slices"
        )

        total_positive += len(positive_indices)
        total_negative += len(negative_indices)

        selected_indices = [
            (idx, 1) for idx in positive_indices
        ] + [
            (idx, 0) for idx in negative_indices
        ]

        for slice_idx, contains_pancreas in selected_indices:
            images.append(ct[:, :, slice_idx])
            masks.append(mask[:, :, slice_idx])
            metadata.append(
                {
                    "patient": patient_name,
                    "slice": int(slice_idx),
                    "positive": bool(contains_pancreas),
                }
            )

    if not images:
        raise RuntimeError("No training slices were created.")

    print("\nTRAINING DATA SUMMARY")
    print(f"Positive slices: {total_positive}")
    print(f"Negative slices: {total_negative}")
    print(f"Total slices:    {len(images)}")

    return PancreasSliceDataset(images, masks, metadata)


# ============================================================
# 7. MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nTraining device: {device}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

# Important change:
# Dice + BCE is more appropriate than Dice alone once empty/negative slices
# are deliberately included.
loss_function = DiceCELoss(
    sigmoid=True,
    lambda_dice=1.0,
    lambda_ce=1.0,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 8. WHOLE-VOLUME VALIDATION
# ============================================================

def validate_patient_volume(
    patient_name: str,
    model: torch.nn.Module,
    inference_batch_size: int = 8,
):
    """
    Process every axial slice of one patient and calculate whole-volume Dice.
    """
    ct, reference, _ = load_patient(patient_name)

    predicted = np.zeros_like(reference, dtype=np.uint8)

    model.eval()

    with torch.no_grad():
        for start in range(0, ct.shape[2], inference_batch_size):
            end = min(start + inference_batch_size, ct.shape[2])

            # [H, W, B] -> [B, H, W] -> [B, 1, H, W]
            batch_np = np.moveaxis(
                ct[:, :, start:end],
                2,
                0,
            )

            batch_tensor = torch.from_numpy(
                batch_np.astype(np.float32)
            ).unsqueeze(1).to(device)

            logits = model(batch_tensor)
            probabilities = torch.sigmoid(logits)

            batch_prediction = (
                probabilities[:, 0].cpu().numpy()
                > PREDICTION_THRESHOLD
            ).astype(np.uint8)

            # [B, H, W] -> [H, W, B]
            predicted[:, :, start:end] = np.moveaxis(
                batch_prediction,
                0,
                2,
            )

    prediction_bool = predicted.astype(bool)
    reference_bool = reference.astype(bool)

    tp = np.logical_and(prediction_bool, reference_bool).sum()
    fp = np.logical_and(
        prediction_bool,
        np.logical_not(reference_bool),
    ).sum()
    fn = np.logical_and(
        np.logical_not(prediction_bool),
        reference_bool,
    ).sum()

    eps = 1e-8

    dice = (2.0 * tp + eps) / (
        2.0 * tp + fp + fn + eps
    )

    iou = (tp + eps) / (
        tp + fp + fn + eps
    )

    precision = (tp + eps) / (
        tp + fp + eps
    )

    recall = (tp + eps) / (
        tp + fn + eps
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def validate_all_patients(patient_names, model):
    patient_results = {}

    for patient_name in patient_names:
        if patient_name in EXCLUDED_PATIENTS:
            continue

        result = validate_patient_volume(
            patient_name,
            model,
        )

        patient_results[patient_name] = result

    if not patient_results:
        raise RuntimeError("No validation patients evaluated.")

    mean_dice = float(
        np.mean([
            result["dice"]
            for result in patient_results.values()
        ])
    )

    return mean_dice, patient_results


# ============================================================
# 9. SAVE EXPERIMENT INFORMATION
# ============================================================

split_info = {
    "train_patients": TRAIN_PATIENTS,
    "validation_patients": VAL_PATIENTS,
    "final_test_patients_not_used_in_training": FINAL_TEST_PATIENTS,
    "excluded_patients": EXCLUDED_PATIENTS,
}

with SPLIT_JSON.open("w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=2)

experiment_config = {
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "loss": "MONAI DiceCELoss(sigmoid=True)",
    "hu_min": HU_MIN,
    "hu_max": HU_MAX,
    "min_positive_pixels": MIN_POSITIVE_PIXELS,
    "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
    "nearby_negative_margin": NEARBY_NEGATIVE_MARGIN,
    "prediction_threshold": PREDICTION_THRESHOLD,
    "validation_every_epochs": VAL_EVERY,
    "early_stopping_patience_validation_checks":
        EARLY_STOPPING_PATIENCE,
    "architecture": {
        "name": "MONAI UNet",
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": [16, 32, 64, 128],
        "strides": [2, 2, 2],
        "num_res_units": 2,
    },
}

with CONFIG_JSON.open("w", encoding="utf-8") as f:
    json.dump(experiment_config, f, indent=2)


# ============================================================
# 10. BUILD TRAINING DATALOADER
# ============================================================

train_dataset = build_training_dataset(TRAIN_PATIENTS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 11. TRAINING LOOP
# ============================================================

history = []

best_validation_dice = -1.0
best_epoch = None
validation_checks_without_improvement = 0

print("\n" + "=" * 70)
print("STARTING IMPROVED TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    model.train()
    epoch_loss = 0.0

    for batch_images, batch_masks in train_loader:
        batch_images = batch_images.to(device)
        batch_masks = batch_masks.to(device)

        optimizer.zero_grad()

        logits = model(batch_images)

        loss = loss_function(
            logits,
            batch_masks,
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_training_loss = (
        epoch_loss / len(train_loader)
    )

    row = {
        "epoch": epoch,
        "training_loss": average_training_loss,
        "mean_validation_dice": "",
        "validation_details": "",
    }

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"- Train Loss: {average_training_loss:.4f}",
        end="",
    )

    if epoch % VAL_EVERY == 0 or epoch == EPOCHS:

        mean_val_dice, val_results = validate_all_patients(
            VAL_PATIENTS,
            model,
        )

        validation_details = "; ".join(
            [
                (
                    f"{patient}:"
                    f"Dice={metrics['dice']:.4f},"
                    f"IoU={metrics['iou']:.4f},"
                    f"P={metrics['precision']:.4f},"
                    f"R={metrics['recall']:.4f}"
                )
                for patient, metrics in val_results.items()
            ]
        )

        row["mean_validation_dice"] = mean_val_dice
        row["validation_details"] = validation_details

        print(
            f" - Mean Val Dice: {mean_val_dice:.4f}"
        )

        for patient, metrics in val_results.items():
            print(
                f"    {patient}: "
                f"Dice={metrics['dice']:.4f}, "
                f"IoU={metrics['iou']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}"
            )

        # Save BEST model based only on validation patients.
        if mean_val_dice > best_validation_dice:
            best_validation_dice = mean_val_dice
            best_epoch = epoch
            validation_checks_without_improvement = 0

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH,
            )

            print(
                f"    NEW BEST MODEL SAVED "
                f"(Val Dice={best_validation_dice:.4f})"
            )

        else:
            validation_checks_without_improvement += 1

        history.append(row)

        if (
            validation_checks_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered because validation Dice "
                "did not improve."
            )
            break

    else:
        print()
        history.append(row)


# Save final state separately.
torch.save(
    model.state_dict(),
    LAST_MODEL_PATH,
)


# ============================================================
# 12. SAVE TRAINING HISTORY
# ============================================================

with HISTORY_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "training_loss",
            "mean_validation_dice",
            "validation_details",
        ],
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# 13. CREATE TRAINING / VALIDATION CURVE
# ============================================================

epochs_history = [
    row["epoch"] for row in history
]

training_losses = [
    row["training_loss"] for row in history
]

validation_epochs = [
    row["epoch"]
    for row in history
    if row["mean_validation_dice"] != ""
]

validation_dices = [
    float(row["mean_validation_dice"])
    for row in history
    if row["mean_validation_dice"] != ""
]

fig, ax1 = plt.subplots(
    figsize=(10, 6)
)

ax1.plot(
    epochs_history,
    training_losses,
    marker="o",
    markersize=3,
    label="Training Loss",
)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Dice + BCE Training Loss")
ax1.grid(True)

ax2 = ax1.twinx()

if validation_epochs:
    ax2.plot(
        validation_epochs,
        validation_dices,
        marker="s",
        markersize=5,
        label="Whole-volume Validation Dice",
    )

ax2.set_ylabel("Validation Dice")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="center right",
)

plt.title(
    "Experiment 04 - Training Loss and Validation Dice"
)

plt.tight_layout()
plt.savefig(
    CURVE_PNG,
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Best epoch:           {best_epoch}")
print(
    f"Best mean Val Dice:   "
    f"{best_validation_dice:.4f}"
)
print(f"Best model:           {BEST_MODEL_PATH}")
print(f"Last model:           {LAST_MODEL_PATH}")
print(f"Training history:     {HISTORY_CSV}")
print(f"Dataset split:        {SPLIT_JSON}")
print(f"Experiment config:    {CONFIG_JSON}")
print(f"Curve:                {CURVE_PNG}")

print(
    "\nIMPORTANT: Do not evaluate Pat_10 / Pat_11 until you are satisfied "
    "with the training/validation pipeline. They are the final held-out "
    "test patients for this experiment."
)