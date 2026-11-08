"""
Experiment 07 — Spatial Standardization / Voxel-Spacing Resampling
=================================================================

Research question
-----------------
Does resampling all CT volumes and masks to a common physical voxel spacing
improve inter-patient generalization of the 2D U-Net?

Controlled comparison
---------------------
Experiment 07 keeps the successful Experiment 06A setup:

UNCHANGED
---------
- Clean training patients:
  Pat_01, Pat_03, Pat_04, Pat_05, Pat_06, Pat_07, Pat_09, Pat_10
- Validation:
  Pat_11, Pat_12
- Final untouched test:
  Pat_13, Pat_14, Pat_15
- 2D MONAI U-Net
- Dice + BCE loss
- Adam optimizer
- Learning rate = 0.001
- Batch size = 8
- 1:1 negative:positive slice sampling
- 50% nearby/hard negatives + 50% farther/random negatives
- HU window [-100, 250]
- Prediction threshold = 0.5
- Validation every 2 epochs
- Early stopping by mean whole-volume validation Dice

MAIN CHANGE
-----------
Before extracting axial slices:
1. Read original NIfTI voxel spacing.
2. Compute a TARGET spacing using ONLY the training patients:
       median(training spacings)
3. Resample CT with LINEAR interpolation (order=1).
4. Resample binary mask with NEAREST-NEIGHBOR interpolation (order=0).
5. Center crop/pad the in-plane dimensions to 512 x 512.
6. Normalize CT using the same fixed HU window as earlier experiments.

Why center crop/pad?
--------------------
After physical resampling, patients naturally have different matrix sizes.
The U-Net batch needs a common H x W size. Center crop/pad preserves the
physical scale created by resampling, unlike resizing every image back to
512 x 512.

IMPORTANT
---------
- No connected-component post-processing is used during validation.
  This isolates the effect of spatial standardization.
- Pat_13, Pat_14, Pat_15 remain completely untouched.
- Target spacing is estimated from TRAINING patients only, avoiding leakage.

Install if needed:
    pip install torch monai nibabel numpy matplotlib scipy

Run:
    python train_experiment_07.py
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
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")

RESULTS_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_07_results/"
    r"Experiment_07_Spatial_Standardization"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATIENTS = [
    "Pat_01",
    "Pat_03",
    "Pat_04",
    "Pat_05",
    "Pat_06",
    "Pat_07",
    "Pat_09",
    "Pat_10",
]

VAL_PATIENTS = [
    "Pat_11",
    "Pat_12",
]

FINAL_TEST_PATIENTS = [
    "Pat_13",
    "Pat_14",
    "Pat_15",
]

EXCLUDED_PATIENTS = [
    "Pat_02",
    "Pat_08",
]

SEED = 42

EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 1e-3

VAL_EVERY = 2
EARLY_STOPPING_PATIENCE = 8  # validation checks, not epochs

HU_MIN = -100.0
HU_MAX = 250.0

PREDICTION_THRESHOLD = 0.5

MIN_POSITIVE_PIXELS = 50

NEGATIVE_TO_POSITIVE_RATIO = 1.0
HARD_NEGATIVE_FRACTION = 0.50

# Because z spacing is standardized, 30 axial slices now correspond
# to a consistent physical distance determined by target z spacing.
NEARBY_NEGATIVE_MARGIN_SLICES = 30

# Common U-Net input size after physical resampling.
TARGET_HEIGHT = 512
TARGET_WIDTH = 512

NUM_WORKERS = 0

BEST_MODEL_PATH = RESULTS_DIR / "Experiment_07_best_model.pth"
LAST_MODEL_PATH = RESULTS_DIR / "Experiment_07_last_model.pth"

HISTORY_CSV = RESULTS_DIR / "Experiment_07_training_history.csv"
VAL_DETAILS_CSV = RESULTS_DIR / "Experiment_07_validation_details.csv"
SLICE_METADATA_CSV = RESULTS_DIR / "Experiment_07_selected_training_slices.csv"
RESAMPLING_AUDIT_CSV = RESULTS_DIR / "Experiment_07_resampling_audit.csv"

CONFIG_JSON = RESULTS_DIR / "Experiment_07_config.json"
SPLIT_JSON = RESULTS_DIR / "Experiment_07_split.json"
CURVE_PNG = RESULTS_DIR / "Experiment_07_training_validation_curve.png"


# ============================================================
# 2. REPRODUCIBILITY / SPLIT SAFETY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

assert not (set(TRAIN_PATIENTS) & set(VAL_PATIENTS))
assert not (set(TRAIN_PATIENTS) & set(FINAL_TEST_PATIENTS))
assert not (set(VAL_PATIENTS) & set(FINAL_TEST_PATIENTS))
assert not (set(TRAIN_PATIENTS) & set(EXCLUDED_PATIENTS))


# ============================================================
# 3. FILE HELPERS
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
        preferred = [
            "origional_images.nii",
            "origional_images.nii.gz",
            "original_images.nii",
            "original_images.nii.gz",
        ]

        for candidate in preferred:
            for path in files:
                if path.name.lower() == candidate:
                    return path

        for path in files:
            name = path.name.lower()
            if (
                ("origional" in name or "original" in name)
                and "mask" not in name
                and "prediction" not in name
            ):
                return path

    elif kind == "mask":
        preferred = [
            "masked_images.nii",
            "masked_images.nii.gz",
            "mask_images.nii",
            "mask_images.nii.gz",
        ]

        for candidate in preferred:
            for path in files:
                if path.name.lower() == candidate:
                    return path

        for path in files:
            name = path.name.lower()
            if (
                ("masked_images" in name or "mask_images" in name)
                and "prediction" not in name
                and "false_positive" not in name
                and "false_negative" not in name
                and "true_positive" not in name
            ):
                return path

    raise FileNotFoundError(
        f"Could not find {kind} NIfTI file inside:\n{patient_dir}"
    )


def inspect_patient_header(patient_name: str):
    patient_dir = BASE_DIR / patient_name

    ct_path = find_patient_file(
        patient_dir,
        "ct",
    )

    ct_img = nib.load(
        str(ct_path)
    )

    spacing = np.array(
        ct_img.header.get_zooms()[:3],
        dtype=np.float64,
    )

    orientation = nib.aff2axcodes(
        ct_img.affine
    )

    return {
        "patient": patient_name,
        "shape": tuple(ct_img.shape[:3]),
        "spacing": spacing,
        "orientation": orientation,
    }


# ============================================================
# 4. COMPUTE TARGET SPACING FROM TRAINING PATIENTS ONLY
# ============================================================

training_header_info = [
    inspect_patient_header(patient)
    for patient in TRAIN_PATIENTS
]

training_spacings = np.stack(
    [
        info["spacing"]
        for info in training_header_info
    ],
    axis=0,
)

TARGET_SPACING = np.median(
    training_spacings,
    axis=0,
)

print("=" * 78)
print("EXPERIMENT 07 — SPATIAL STANDARDIZATION")
print("=" * 78)

print("\nTraining-patient voxel spacings:")

for info in training_header_info:
    sx, sy, sz = info["spacing"]

    print(
        f"{info['patient']}: "
        f"{sx:.4f} x {sy:.4f} x {sz:.4f} mm | "
        f"shape={info['shape']} | "
        f"orientation={info['orientation']}"
    )

print(
    "\nTARGET SPACING "
    "(median of TRAINING patients only): "
    f"{TARGET_SPACING[0]:.4f} x "
    f"{TARGET_SPACING[1]:.4f} x "
    f"{TARGET_SPACING[2]:.4f} mm"
)


# ============================================================
# 5. SPATIAL STANDARDIZATION FUNCTIONS
# ============================================================

def resample_array(
    array: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: np.ndarray,
    interpolation_order: int,
) -> np.ndarray:
    """
    Resample array so each voxel has target physical spacing.

    zoom factor = original spacing / target spacing.

    CT:
        order=1 (linear interpolation)

    Mask:
        order=0 (nearest-neighbor interpolation)
    """

    zoom_factors = (
        original_spacing
        / target_spacing
    )

    resampled = ndimage.zoom(
        array,
        zoom=zoom_factors,
        order=interpolation_order,
        mode="nearest",
        prefilter=False,
    )

    return resampled


def center_crop_or_pad_xy(
    volume: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float,
):
    """
    Center crop or pad ONLY array axes 0 and 1.
    The axial/z dimension is kept unchanged.

    Returns:
        standardized volume
        transform information for audit
    """

    original_h, original_w, depth = volume.shape

    # --------------------------
    # Crop axis 0 if necessary.
    # --------------------------
    if original_h > target_h:
        start_h = (original_h - target_h) // 2
        end_h = start_h + target_h

        volume = volume[
            start_h:end_h,
            :,
            :,
        ]

        crop_h_before = start_h
        crop_h_after = original_h - end_h
    else:
        crop_h_before = 0
        crop_h_after = 0

    # --------------------------
    # Crop axis 1 if necessary.
    # --------------------------
    current_h, current_w, _ = volume.shape

    if current_w > target_w:
        start_w = (current_w - target_w) // 2
        end_w = start_w + target_w

        volume = volume[
            :,
            start_w:end_w,
            :,
        ]

        crop_w_before = start_w
        crop_w_after = current_w - end_w
    else:
        crop_w_before = 0
        crop_w_after = 0

    # --------------------------
    # Pad if necessary.
    # --------------------------
    current_h, current_w, _ = volume.shape

    pad_h_total = max(
        0,
        target_h - current_h,
    )

    pad_w_total = max(
        0,
        target_w - current_w,
    )

    pad_h_before = (
        pad_h_total // 2
    )

    pad_h_after = (
        pad_h_total
        - pad_h_before
    )

    pad_w_before = (
        pad_w_total // 2
    )

    pad_w_after = (
        pad_w_total
        - pad_w_before
    )

    volume = np.pad(
        volume,
        pad_width=(
            (pad_h_before, pad_h_after),
            (pad_w_before, pad_w_after),
            (0, 0),
        ),
        mode="constant",
        constant_values=pad_value,
    )

    transform_info = {
        "crop_axis0_before": int(crop_h_before),
        "crop_axis0_after": int(crop_h_after),
        "crop_axis1_before": int(crop_w_before),
        "crop_axis1_after": int(crop_w_after),
        "pad_axis0_before": int(pad_h_before),
        "pad_axis0_after": int(pad_h_after),
        "pad_axis1_before": int(pad_w_before),
        "pad_axis1_after": int(pad_w_after),
    }

    return volume, transform_info


def preprocess_ct(
    ct: np.ndarray,
) -> np.ndarray:
    """
    Same fixed HU preprocessing used in earlier experiments.
    """
    ct = np.clip(
        ct,
        HU_MIN,
        HU_MAX,
    )

    ct = (
        ct - HU_MIN
    ) / (
        HU_MAX - HU_MIN
    )

    return ct.astype(
        np.float32
    )


# ============================================================
# 6. LOAD + RESAMPLE + STANDARDIZE A PATIENT
# ============================================================

resampling_audit_rows = []


def load_standardized_patient(
    patient_name: str,
    record_audit: bool = False,
):
    patient_dir = (
        BASE_DIR
        / patient_name
    )

    if not patient_dir.exists():
        raise FileNotFoundError(
            f"Patient folder not found:\n{patient_dir}"
        )

    ct_path = find_patient_file(
        patient_dir,
        "ct",
    )

    mask_path = find_patient_file(
        patient_dir,
        "mask",
    )

    ct_img = nib.load(
        str(ct_path)
    )

    mask_img = nib.load(
        str(mask_path)
    )

    raw_ct = ct_img.get_fdata(
        dtype=np.float32
    )

    raw_mask = mask_img.get_fdata()

    if raw_ct.shape != raw_mask.shape:
        raise ValueError(
            f"{patient_name}: CT/mask shape mismatch "
            f"{raw_ct.shape} vs {raw_mask.shape}"
        )

    if not np.allclose(
        ct_img.affine,
        mask_img.affine,
        atol=1e-4,
    ):
        raise ValueError(
            f"{patient_name}: CT/mask affine mismatch."
        )

    orientation = nib.aff2axcodes(
        ct_img.affine
    )

    if orientation[2] not in (
        "S",
        "I",
    ):
        raise ValueError(
            f"{patient_name}: array axis 2 is not axial. "
            f"Orientation={orientation}"
        )

    original_spacing = np.array(
        ct_img.header.get_zooms()[:3],
        dtype=np.float64,
    )

    original_shape = tuple(
        raw_ct.shape
    )

    binary_mask = (
        raw_mask > 0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Resample RAW HU CT first, then normalize later.
    # --------------------------------------------------------
    resampled_ct = resample_array(
        raw_ct,
        original_spacing,
        TARGET_SPACING,
        interpolation_order=1,
    )

    resampled_mask = resample_array(
        binary_mask,
        original_spacing,
        TARGET_SPACING,
        interpolation_order=0,
    )

    resampled_mask = (
        resampled_mask > 0.5
    ).astype(
        np.uint8
    )

    if (
        resampled_ct.shape
        != resampled_mask.shape
    ):
        raise ValueError(
            f"{patient_name}: resampled CT/mask shape mismatch "
            f"{resampled_ct.shape} vs {resampled_mask.shape}"
        )

    resampled_shape_before_crop_pad = tuple(
        resampled_ct.shape
    )

    # --------------------------------------------------------
    # Standardize H x W while preserving physical scale.
    # CT padding value is HU_MIN because values below HU_MIN
    # would be clipped to HU_MIN anyway.
    # --------------------------------------------------------
    standardized_ct, transform_info = center_crop_or_pad_xy(
        resampled_ct,
        TARGET_HEIGHT,
        TARGET_WIDTH,
        pad_value=HU_MIN,
    )

    standardized_mask, _ = center_crop_or_pad_xy(
        resampled_mask,
        TARGET_HEIGHT,
        TARGET_WIDTH,
        pad_value=0,
    )

    standardized_mask = (
        standardized_mask > 0
    ).astype(
        np.uint8
    )

    if (
        standardized_ct.shape
        != standardized_mask.shape
    ):
        raise ValueError(
            f"{patient_name}: standardized CT/mask shape mismatch."
        )

    standardized_ct = preprocess_ct(
        standardized_ct
    )

    if record_audit:
        audit_row = {
            "patient": patient_name,

            "original_shape_axis0":
                original_shape[0],
            "original_shape_axis1":
                original_shape[1],
            "original_shape_axis2":
                original_shape[2],

            "original_spacing_axis0_mm":
                float(original_spacing[0]),
            "original_spacing_axis1_mm":
                float(original_spacing[1]),
            "original_spacing_axis2_mm":
                float(original_spacing[2]),

            "target_spacing_axis0_mm":
                float(TARGET_SPACING[0]),
            "target_spacing_axis1_mm":
                float(TARGET_SPACING[1]),
            "target_spacing_axis2_mm":
                float(TARGET_SPACING[2]),

            "resampled_shape_axis0":
                resampled_shape_before_crop_pad[0],
            "resampled_shape_axis1":
                resampled_shape_before_crop_pad[1],
            "resampled_shape_axis2":
                resampled_shape_before_crop_pad[2],

            "final_shape_axis0":
                standardized_ct.shape[0],
            "final_shape_axis1":
                standardized_ct.shape[1],
            "final_shape_axis2":
                standardized_ct.shape[2],

            "orientation":
                "".join(orientation),

            **transform_info,

            "reference_foreground_voxels_after_standardization":
                int(standardized_mask.sum()),
        }

        resampling_audit_rows.append(
            audit_row
        )

    return (
        standardized_ct,
        standardized_mask,
    )


# ============================================================
# 7. CREATE RESAMPLING AUDIT FOR TRAIN + VALIDATION
# ============================================================

print("\nCreating resampling audit...")

for patient_name in (
    TRAIN_PATIENTS
    + VAL_PATIENTS
):
    ct_tmp, mask_tmp = load_standardized_patient(
        patient_name,
        record_audit=True,
    )

    print(
        f"{patient_name}: standardized shape="
        f"{ct_tmp.shape}"
    )

audit_fields = list(
    resampling_audit_rows[0].keys()
)

with RESAMPLING_AUDIT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=audit_fields,
    )

    writer.writeheader()
    writer.writerows(
        resampling_audit_rows
    )


# ============================================================
# 8. NEGATIVE-SLICE SAMPLING
# ============================================================

def select_training_indices(
    mask: np.ndarray,
    rng: np.random.Generator,
):
    foreground_per_slice = np.sum(
        mask,
        axis=(0, 1),
    )

    positive_indices = np.where(
        foreground_per_slice
        > MIN_POSITIVE_PIXELS
    )[0]

    if len(positive_indices) == 0:
        raise ValueError(
            "No meaningful positive pancreas slices found."
        )

    empty_indices = np.where(
        foreground_per_slice == 0
    )[0]

    desired_total_negatives = min(
        int(
            round(
                len(positive_indices)
                * NEGATIVE_TO_POSITIVE_RATIO
            )
        ),
        len(empty_indices),
    )

    first_positive = int(
        positive_indices.min()
    )

    last_positive = int(
        positive_indices.max()
    )

    nearby_low = max(
        0,
        first_positive
        - NEARBY_NEGATIVE_MARGIN_SLICES,
    )

    nearby_high = min(
        mask.shape[2] - 1,
        last_positive
        + NEARBY_NEGATIVE_MARGIN_SLICES,
    )

    nearby_candidates = empty_indices[
        (empty_indices >= nearby_low)
        & (empty_indices <= nearby_high)
    ]

    far_candidates = np.setdiff1d(
        empty_indices,
        nearby_candidates,
    )

    desired_hard = int(
        round(
            desired_total_negatives
            * HARD_NEGATIVE_FRACTION
        )
    )

    hard_count = min(
        desired_hard,
        len(nearby_candidates),
    )

    if hard_count > 0:
        hard_negatives = np.sort(
            rng.choice(
                nearby_candidates,
                size=hard_count,
                replace=False,
            )
        )
    else:
        hard_negatives = np.array(
            [],
            dtype=int,
        )

    remaining = (
        desired_total_negatives
        - len(hard_negatives)
    )

    random_count = min(
        remaining,
        len(far_candidates),
    )

    if random_count > 0:
        random_negatives = np.sort(
            rng.choice(
                far_candidates,
                size=random_count,
                replace=False,
            )
        )
    else:
        random_negatives = np.array(
            [],
            dtype=int,
        )

    remaining = (
        desired_total_negatives
        - len(hard_negatives)
        - len(random_negatives)
    )

    if remaining > 0:
        unused_nearby = np.setdiff1d(
            nearby_candidates,
            hard_negatives,
        )

        extra_count = min(
            remaining,
            len(unused_nearby),
        )

        if extra_count > 0:
            extra = np.sort(
                rng.choice(
                    unused_nearby,
                    size=extra_count,
                    replace=False,
                )
            )

            random_negatives = np.sort(
                np.concatenate(
                    [
                        random_negatives,
                        extra,
                    ]
                )
            )

    return (
        positive_indices.astype(int),
        hard_negatives.astype(int),
        random_negatives.astype(int),
    )


# ============================================================
# 9. DATASET
# ============================================================

class SliceDataset(Dataset):

    def __init__(
        self,
        images,
        masks,
    ):
        # Memory-conscious storage.
        self.images = np.asarray(
            images,
            dtype=np.float16,
        )

        self.masks = np.asarray(
            masks,
            dtype=np.uint8,
        )

    def __len__(self):
        return len(
            self.images
        )

    def __getitem__(
        self,
        index,
    ):
        image = torch.from_numpy(
            self.images[
                index
            ].astype(
                np.float32
            )
        ).unsqueeze(0)

        mask = torch.from_numpy(
            self.masks[
                index
            ].astype(
                np.float32
            )
        ).unsqueeze(0)

        return (
            image,
            mask,
        )


def build_training_dataset():
    rng = np.random.default_rng(
        SEED
    )

    images = []
    masks = []
    metadata_rows = []

    positive_total = 0
    hard_negative_total = 0
    random_negative_total = 0

    print("\nBuilding Experiment 07 training dataset...")

    for patient_name in TRAIN_PATIENTS:

        ct, mask = load_standardized_patient(
            patient_name,
            record_audit=False,
        )

        (
            positive_indices,
            hard_negative_indices,
            random_negative_indices,
        ) = select_training_indices(
            mask,
            rng,
        )

        positive_total += len(
            positive_indices
        )

        hard_negative_total += len(
            hard_negative_indices
        )

        random_negative_total += len(
            random_negative_indices
        )

        print(
            f"{patient_name}: "
            f"{len(positive_indices)} positive + "
            f"{len(hard_negative_indices)} hard negative + "
            f"{len(random_negative_indices)} random negative"
        )

        selected = (
            [
                (index, "positive")
                for index in positive_indices
            ]
            + [
                (index, "hard_negative")
                for index in hard_negative_indices
            ]
            + [
                (index, "random_negative")
                for index in random_negative_indices
            ]
        )

        for slice_index, slice_type in selected:

            images.append(
                ct[
                    :,
                    :,
                    slice_index,
                ]
            )

            masks.append(
                mask[
                    :,
                    :,
                    slice_index,
                ]
            )

            metadata_rows.append(
                {
                    "patient":
                        patient_name,

                    "slice_index_after_resampling":
                        int(slice_index),

                    "slice_type":
                        slice_type,

                    "foreground_pixels":
                        int(
                            mask[
                                :,
                                :,
                                slice_index,
                            ].sum()
                        ),
                }
            )

    negative_total = (
        hard_negative_total
        + random_negative_total
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 07 TRAINING DATA SUMMARY")
    print("=" * 78)

    print(
        f"Positive slices:        "
        f"{positive_total}"
    )

    print(
        f"Hard negative slices:   "
        f"{hard_negative_total}"
    )

    print(
        f"Random negative slices: "
        f"{random_negative_total}"
    )

    print(
        f"Total negative slices:  "
        f"{negative_total}"
    )

    print(
        f"Total training slices:  "
        f"{len(images)}"
    )

    print(
        "Actual negative:positive ratio: "
        f"{negative_total / positive_total:.3f}:1"
    )

    with SLICE_METADATA_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                metadata_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            metadata_rows
        )

    return SliceDataset(
        images,
        masks,
    )


# ============================================================
# 10. MODEL / LOSS / OPTIMIZER
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"\nTraining device: {device}"
)

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(
        16,
        32,
        64,
        128,
    ),
    strides=(
        2,
        2,
        2,
    ),
    num_res_units=2,
).to(device)

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
# 11. WHOLE-VOLUME VALIDATION
# ============================================================

def range_string(
    indices: np.ndarray,
) -> str:
    if len(indices) == 0:
        return "None"

    return (
        f"{int(indices.min())}-"
        f"{int(indices.max())}"
    )


def validate_patient_volume(
    patient_name: str,
):

    ct, reference = load_standardized_patient(
        patient_name,
        record_audit=False,
    )

    prediction = np.zeros_like(
        reference,
        dtype=np.uint8,
    )

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            ct.shape[2],
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                ct.shape[2],
            )

            batch_np = np.moveaxis(
                ct[
                    :,
                    :,
                    start:end,
                ],
                2,
                0,
            )

            batch_tensor = torch.from_numpy(
                batch_np.astype(
                    np.float32
                )
            ).unsqueeze(1).to(
                device
            )

            logits = model(
                batch_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            batch_prediction = (
                probabilities[
                    :,
                    0,
                ]
                .cpu()
                .numpy()
                > PREDICTION_THRESHOLD
            ).astype(
                np.uint8
            )

            prediction[
                :,
                :,
                start:end,
            ] = np.moveaxis(
                batch_prediction,
                0,
                2,
            )

    prediction_bool = prediction.astype(
        bool
    )

    reference_bool = reference.astype(
        bool
    )

    tp = int(
        np.logical_and(
            prediction_bool,
            reference_bool,
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction_bool,
            np.logical_not(
                reference_bool
            ),
        ).sum()
    )

    fn = int(
        np.logical_and(
            np.logical_not(
                prediction_bool
            ),
            reference_bool,
        ).sum()
    )

    eps = 1e-8

    dice = (
        2.0 * tp + eps
    ) / (
        2.0 * tp
        + fp
        + fn
        + eps
    )

    iou = (
        tp + eps
    ) / (
        tp
        + fp
        + fn
        + eps
    )

    precision = (
        tp + eps
    ) / (
        tp
        + fp
        + eps
    )

    recall = (
        tp + eps
    ) / (
        tp
        + fn
        + eps
    )

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

    false_positive_slices = np.setdiff1d(
        prediction_positive,
        reference_positive,
    )

    missed_reference_slices = np.setdiff1d(
        reference_positive,
        prediction_positive,
    )

    return {
        "dice":
            float(dice),

        "iou":
            float(iou),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "reference_voxels":
            int(
                reference.sum()
            ),

        "predicted_voxels":
            int(
                prediction.sum()
            ),

        "true_positive_voxels":
            tp,

        "false_positive_voxels":
            fp,

        "false_negative_voxels":
            fn,

        "reference_positive_slice_count":
            int(
                len(
                    reference_positive
                )
            ),

        "predicted_positive_slice_count":
            int(
                len(
                    prediction_positive
                )
            ),

        "reference_slice_range":
            range_string(
                reference_positive
            ),

        "predicted_slice_range":
            range_string(
                prediction_positive
            ),

        "false_positive_slice_count":
            int(
                len(
                    false_positive_slices
                )
            ),

        "missed_reference_slice_count":
            int(
                len(
                    missed_reference_slices
                )
            ),
    }


def validate_all_patients(
    epoch: int,
):

    patient_results = {
        patient:
            validate_patient_volume(
                patient
            )
        for patient
        in VAL_PATIENTS
    }

    mean_dice = float(
        np.mean(
            [
                result[
                    "dice"
                ]
                for result
                in patient_results.values()
            ]
        )
    )

    file_exists = (
        VAL_DETAILS_CSV.exists()
    )

    fields = [
        "epoch",
        "patient",
        "dice",
        "iou",
        "precision",
        "recall",
        "reference_voxels",
        "predicted_voxels",
        "true_positive_voxels",
        "false_positive_voxels",
        "false_negative_voxels",
        "reference_positive_slice_count",
        "predicted_positive_slice_count",
        "reference_slice_range",
        "predicted_slice_range",
        "false_positive_slice_count",
        "missed_reference_slice_count",
    ]

    with VAL_DETAILS_CSV.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        if not file_exists:
            writer.writeheader()

        for patient_name, result in patient_results.items():

            writer.writerow(
                {
                    "epoch":
                        epoch,

                    "patient":
                        patient_name,

                    **result,
                }
            )

    return (
        mean_dice,
        patient_results,
    )


# ============================================================
# 12. SAVE CONFIG / SPLIT
# ============================================================

with SPLIT_JSON.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "train_patients":
                TRAIN_PATIENTS,

            "validation_patients":
                VAL_PATIENTS,

            "final_test_patients_UNTOUCHED":
                FINAL_TEST_PATIENTS,

            "excluded_patients":
                EXCLUDED_PATIENTS,
        },
        file,
        indent=2,
    )


with CONFIG_JSON.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "experiment":
                "07",

            "research_question":
                "Does common physical voxel spacing improve "
                "inter-patient segmentation generalization?",

            "target_spacing_source":
                "Median of TRAINING patients only",

            "target_spacing_mm":
                [
                    float(
                        TARGET_SPACING[0]
                    ),
                    float(
                        TARGET_SPACING[1]
                    ),
                    float(
                        TARGET_SPACING[2]
                    ),
                ],

            "ct_interpolation":
                "linear (scipy.ndimage.zoom order=1)",

            "mask_interpolation":
                "nearest-neighbor (scipy.ndimage.zoom order=0)",

            "final_inplane_matrix":
                [
                    TARGET_HEIGHT,
                    TARGET_WIDTH,
                ],

            "seed":
                SEED,

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "loss":
                "MONAI DiceCELoss(sigmoid=True)",

            "hu_window":
                [
                    HU_MIN,
                    HU_MAX,
                ],

            "negative_to_positive_ratio":
                NEGATIVE_TO_POSITIVE_RATIO,

            "hard_negative_fraction":
                HARD_NEGATIVE_FRACTION,

            "nearby_negative_margin_slices":
                NEARBY_NEGATIVE_MARGIN_SLICES,

            "prediction_threshold":
                PREDICTION_THRESHOLD,

            "model":
                {
                    "name":
                        "MONAI UNet",

                    "spatial_dims":
                        2,

                    "in_channels":
                        1,

                    "out_channels":
                        1,

                    "channels":
                        [
                            16,
                            32,
                            64,
                            128,
                        ],

                    "strides":
                        [
                            2,
                            2,
                            2,
                        ],

                    "num_res_units":
                        2,
                },
        },
        file,
        indent=2,
    )


# ============================================================
# 13. BUILD TRAINING DATA
# ============================================================

train_dataset = build_training_dataset()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 14. TRAINING LOOP
# ============================================================

history = []

best_mean_validation_dice = -1.0
best_epoch = None

checks_without_improvement = 0

print("\n" + "=" * 78)
print("STARTING EXPERIMENT 07 TRAINING")
print("=" * 78)

for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_loss = 0.0

    for batch_images, batch_masks in train_loader:

        batch_images = batch_images.to(
            device
        )

        batch_masks = batch_masks.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            batch_images
        )

        loss = loss_function(
            logits,
            batch_masks,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

    average_loss = (
        total_loss
        / len(
            train_loader
        )
    )

    history_row = {
        "epoch":
            epoch,

        "training_loss":
            average_loss,

        "mean_validation_dice":
            "",

        "Pat_11_dice":
            "",

        "Pat_11_precision":
            "",

        "Pat_11_recall":
            "",

        "Pat_11_false_positive_slices":
            "",

        "Pat_11_predicted_slice_range":
            "",

        "Pat_12_dice":
            "",

        "Pat_12_precision":
            "",

        "Pat_12_recall":
            "",

        "Pat_12_false_positive_slices":
            "",

        "Pat_12_predicted_slice_range":
            "",
    }

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"- Train Loss: {average_loss:.4f}",
        end="",
    )

    if epoch % VAL_EVERY == 0:

        (
            mean_validation_dice,
            validation_results,
        ) = validate_all_patients(
            epoch
        )

        history_row[
            "mean_validation_dice"
        ] = mean_validation_dice

        print(
            f" - Mean Val Dice: "
            f"{mean_validation_dice:.4f}"
        )

        for patient_name, result in validation_results.items():

            history_row[
                f"{patient_name}_dice"
            ] = result["dice"]

            history_row[
                f"{patient_name}_precision"
            ] = result["precision"]

            history_row[
                f"{patient_name}_recall"
            ] = result["recall"]

            history_row[
                f"{patient_name}_false_positive_slices"
            ] = result[
                "false_positive_slice_count"
            ]

            history_row[
                f"{patient_name}_predicted_slice_range"
            ] = result[
                "predicted_slice_range"
            ]

            print(
                f"  {patient_name}: "
                f"Dice={result['dice']:.4f}, "
                f"IoU={result['iou']:.4f}, "
                f"Precision={result['precision']:.4f}, "
                f"Recall={result['recall']:.4f}"
            )

            print(
                f"     Ref range="
                f"{result['reference_slice_range']} | "
                f"Pred range="
                f"{result['predicted_slice_range']} | "
                f"FP slices="
                f"{result['false_positive_slice_count']} | "
                f"Missed ref slices="
                f"{result['missed_reference_slice_count']}"
            )

        if (
            mean_validation_dice
            > best_mean_validation_dice
        ):

            best_mean_validation_dice = (
                mean_validation_dice
            )

            best_epoch = epoch

            checks_without_improvement = 0

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH,
            )

            print(
                "  NEW BEST EXPERIMENT 07 MODEL SAVED: "
                f"mean Dice="
                f"{best_mean_validation_dice:.4f}"
            )

        else:
            checks_without_improvement += 1

        history.append(
            history_row
        )

        if (
            checks_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    else:
        print()

        history.append(
            history_row
        )


# ============================================================
# 15. SAVE LAST MODEL / HISTORY
# ============================================================

torch.save(
    model.state_dict(),
    LAST_MODEL_PATH,
)

history_fields = list(
    history[0].keys()
)

with HISTORY_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=history_fields,
    )

    writer.writeheader()
    writer.writerows(
        history
    )


# ============================================================
# 16. TRAINING / VALIDATION CURVE
# ============================================================

epochs_x = [
    row["epoch"]
    for row in history
]

training_loss_y = [
    row["training_loss"]
    for row in history
]

validation_x = [
    row["epoch"]
    for row in history
    if row[
        "mean_validation_dice"
    ] != ""
]

validation_y = [
    float(
        row[
            "mean_validation_dice"
        ]
    )
    for row in history
    if row[
        "mean_validation_dice"
    ] != ""
]

fig, ax1 = plt.subplots(
    figsize=(10, 6)
)

ax1.plot(
    epochs_x,
    training_loss_y,
    marker="o",
    markersize=3,
    label="Training Loss",
)

ax1.set_xlabel(
    "Epoch"
)

ax1.set_ylabel(
    "Dice + BCE Training Loss"
)

ax1.grid(
    True
)

ax2 = ax1.twinx()

ax2.plot(
    validation_x,
    validation_y,
    marker="s",
    markersize=5,
    label="Validation Dice",
)

ax2.set_ylabel(
    "Whole-volume Validation Dice"
)

lines1, labels1 = (
    ax1.get_legend_handles_labels()
)

lines2, labels2 = (
    ax2.get_legend_handles_labels()
)

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="center right",
)

plt.title(
    "Experiment 07 - Spatial Standardization\n"
    "Training Loss and Whole-volume Validation Dice"
)

plt.tight_layout()

plt.savefig(
    CURVE_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("EXPERIMENT 07 COMPLETE")
print("=" * 78)

print(
    "Target spacing: "
    f"{TARGET_SPACING[0]:.4f} x "
    f"{TARGET_SPACING[1]:.4f} x "
    f"{TARGET_SPACING[2]:.4f} mm"
)

print(
    f"Best epoch:         "
    f"{best_epoch}"
)

print(
    f"Best mean Val Dice: "
    f"{best_mean_validation_dice:.4f}"
)

print(
    f"Best model:         "
    f"{BEST_MODEL_PATH}"
)

print(
    f"Training history:   "
    f"{HISTORY_CSV}"
)

print(
    f"Validation details: "
    f"{VAL_DETAILS_CSV}"
)

print(
    f"Resampling audit:   "
    f"{RESAMPLING_AUDIT_CSV}"
)

print(
    f"Selected slices:    "
    f"{SLICE_METADATA_CSV}"
)

print(
    f"Config:             "
    f"{CONFIG_JSON}"
)

print(
    f"Curve:              "
    f"{CURVE_PNG}"
)

print(
    "\nCompare Experiment 07 directly against the RAW "
    "Experiment 06A validation baseline:"
)

print(
    "Pat_11 baseline: Dice=0.0313, "
    "Precision=0.0629, Recall=0.0208"
)

print(
    "Pat_12 baseline: Dice=0.5084, "
    "Precision=0.6098, Recall=0.4359"
)

print(
    "Mean baseline Dice=0.2698"
)

print(
    "\nIMPORTANT: Do NOT evaluate Pat_13, Pat_14, or Pat_15 yet."
)