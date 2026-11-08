
"""
Experiment 09B — 2.5D U-Net with Diversified Model-Mined Hard Negatives
========================================================================

Research question
-----------------
Can model-based hard-negative sampling reduce anatomically coherent
false-positive pancreas predictions while preserving the segmentation gains
of Experiment 08?

Controlled comparison
---------------------
Experiment 08:
    2.5D U-Net
    1:1 negative:positive ratio
    conventional nearby/random negatives

Experiment 09B:
    SAME 2.5D U-Net
    SAME 1:1 negative:positive ratio
    SAME patient split
    SAME HU preprocessing
    SAME optimizer / LR / loss / threshold
    SAME validation method

MAIN CHANGE:
    approximately half of all negative center slices are replaced by the
    diversified model-mined hard negatives produced by Experiment 09A-R.

Training is FROM SCRATCH.
Experiment 08 weights are NOT used for initialization.

Training cohort:
    Pat_01, Pat_03, Pat_04, Pat_05,
    Pat_06, Pat_07, Pat_09, Pat_10

Validation only:
    Pat_11, Pat_12

Final test — untouched:
    Pat_13, Pat_14, Pat_15

Excluded from clean cohort:
    Pat_02, Pat_08

Expected negative composition:
    621 positive center slices
    621 negative center slices total

    ~310 diversified model-mined hard negatives
    ~311 conventional negatives

Conventional negatives are selected from the remaining empty slices using:
    ~50% nearby negatives
    remainder random/far negatives

No validation/test information is used for negative selection.

Install if needed:
    pip install torch monai nibabel numpy pandas matplotlib

Run:
    python train_experiment_09B.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")

HARD_NEGATIVE_CSV = Path(
    r"D:/Thesis Folder/Results/Experiment_09A_results/"
    r"Experiment_09A_Hard_Negative_Mining/"
    r"Experiment_09A_diversified_hard_negatives.csv"
)

RESULTS_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_09B_results/"
    r"Experiment_09B_2p5D_Hard_Negative_Mining"
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
EARLY_STOPPING_PATIENCE = 8  # validation checks

HU_MIN = -100.0
HU_MAX = 250.0

PREDICTION_THRESHOLD = 0.5

MIN_POSITIVE_PIXELS = 50

NEGATIVE_TO_POSITIVE_RATIO = 1.0

# For the conventional half only:
CONVENTIONAL_NEARBY_FRACTION = 0.50
NEARBY_NEGATIVE_MARGIN_SLICES = 30

NUM_WORKERS = 0

BEST_MODEL_PATH = RESULTS_DIR / "Experiment_09B_best_model.pth"
LAST_MODEL_PATH = RESULTS_DIR / "Experiment_09B_last_model.pth"

HISTORY_CSV = RESULTS_DIR / "Experiment_09B_training_history.csv"
VAL_DETAILS_CSV = RESULTS_DIR / "Experiment_09B_validation_details.csv"
SELECTED_SLICES_CSV = RESULTS_DIR / "Experiment_09B_selected_training_slices.csv"

CONFIG_JSON = RESULTS_DIR / "Experiment_09B_config.json"
SPLIT_JSON = RESULTS_DIR / "Experiment_09B_split.json"
CURVE_PNG = RESULTS_DIR / "Experiment_09B_training_validation_curve.png"


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
# 4. LOAD / PREPROCESS
# ============================================================

def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct, HU_MIN, HU_MAX)
    ct = (ct - HU_MIN) / (HU_MAX - HU_MIN)
    return ct.astype(np.float32)


def load_patient(patient_name: str):
    patient_dir = BASE_DIR / patient_name

    ct_path = find_patient_file(patient_dir, "ct")
    mask_path = find_patient_file(patient_dir, "mask")

    ct_img = nib.load(str(ct_path))
    mask_img = nib.load(str(mask_path))

    raw_ct = ct_img.get_fdata(dtype=np.float32)
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

    orientation = nib.aff2axcodes(ct_img.affine)

    if orientation[2] not in ("S", "I"):
        raise ValueError(
            f"{patient_name}: array axis 2 is not axial. "
            f"Orientation={orientation}"
        )

    ct = preprocess_ct(raw_ct)
    mask = (raw_mask > 0).astype(np.uint8)

    return ct, mask, ct_img


# ============================================================
# 5. 2.5D INPUT
# ============================================================

def build_three_slice_input(
    ct_volume: np.ndarray,
    center_index: int,
) -> np.ndarray:
    last_index = ct_volume.shape[2] - 1

    previous_index = max(
        0,
        center_index - 1,
    )

    next_index = min(
        last_index,
        center_index + 1,
    )

    return np.stack(
        [
            ct_volume[:, :, previous_index],
            ct_volume[:, :, center_index],
            ct_volume[:, :, next_index],
        ],
        axis=0,
    ).astype(np.float32)


# ============================================================
# 6. LOAD + AUDIT HARD NEGATIVE CSV
# ============================================================

if not HARD_NEGATIVE_CSV.exists():
    raise FileNotFoundError(
        f"Diversified hard-negative CSV not found:\n"
        f"{HARD_NEGATIVE_CSV}"
    )

hard_df = pd.read_csv(HARD_NEGATIVE_CSV)

required_columns = {
    "patient",
    "slice_index",
    "predicted_pixels",
}

missing = required_columns - set(hard_df.columns)

if missing:
    raise ValueError(
        f"Hard-negative CSV missing columns: {sorted(missing)}"
    )

unexpected_patients = (
    set(hard_df["patient"].unique())
    - set(TRAIN_PATIENTS)
)

if unexpected_patients:
    raise ValueError(
        "Hard-negative CSV contains patients outside training cohort: "
        f"{sorted(unexpected_patients)}"
    )

if hard_df.duplicated(
    subset=["patient", "slice_index"]
).any():
    raise ValueError(
        "Duplicate patient/slice hard negatives found."
    )

hard_negative_indices_by_patient = {}

for patient_name in TRAIN_PATIENTS:
    patient_rows = hard_df[
        hard_df["patient"] == patient_name
    ]

    hard_negative_indices_by_patient[
        patient_name
    ] = sorted(
        patient_rows["slice_index"]
        .astype(int)
        .tolist()
    )


# ============================================================
# 7. NEGATIVE SELECTION
# ============================================================

def select_conventional_negatives(
    mask: np.ndarray,
    positive_indices: np.ndarray,
    mined_hard_indices: np.ndarray,
    desired_count: int,
    rng: np.random.Generator,
):
    """
    Select conventional negatives only from slices NOT already used as mined
    hard negatives.

    Prefer:
      ~50% nearby to pancreas range
      remainder farther/random

    Fill from the other pool if one category is insufficient.
    """

    foreground_per_slice = np.sum(
        mask,
        axis=(0, 1),
    )

    empty_indices = np.where(
        foreground_per_slice == 0
    )[0]

    available = np.setdiff1d(
        empty_indices,
        mined_hard_indices,
    )

    first_positive = int(
        positive_indices.min()
    )

    last_positive = int(
        positive_indices.max()
    )

    nearby_low = max(
        0,
        first_positive - NEARBY_NEGATIVE_MARGIN_SLICES,
    )

    nearby_high = min(
        mask.shape[2] - 1,
        last_positive + NEARBY_NEGATIVE_MARGIN_SLICES,
    )

    nearby_candidates = available[
        (available >= nearby_low)
        & (available <= nearby_high)
    ]

    far_candidates = np.setdiff1d(
        available,
        nearby_candidates,
    )

    desired_nearby = int(
        round(
            desired_count
            * CONVENTIONAL_NEARBY_FRACTION
        )
    )

    nearby_count = min(
        desired_nearby,
        len(nearby_candidates),
    )

    if nearby_count > 0:
        nearby = np.sort(
            rng.choice(
                nearby_candidates,
                size=nearby_count,
                replace=False,
            )
        )
    else:
        nearby = np.array([], dtype=int)

    remaining = desired_count - len(nearby)

    far_count = min(
        remaining,
        len(far_candidates),
    )

    if far_count > 0:
        random_far = np.sort(
            rng.choice(
                far_candidates,
                size=far_count,
                replace=False,
            )
        )
    else:
        random_far = np.array([], dtype=int)

    remaining = (
        desired_count
        - len(nearby)
        - len(random_far)
    )

    if remaining > 0:
        unused_nearby = np.setdiff1d(
            nearby_candidates,
            nearby,
        )

        extra_nearby_count = min(
            remaining,
            len(unused_nearby),
        )

        if extra_nearby_count > 0:
            extra_nearby = np.sort(
                rng.choice(
                    unused_nearby,
                    size=extra_nearby_count,
                    replace=False,
                )
            )

            nearby = np.sort(
                np.concatenate(
                    [nearby, extra_nearby]
                )
            )

    remaining = (
        desired_count
        - len(nearby)
        - len(random_far)
    )

    if remaining > 0:
        unused_far = np.setdiff1d(
            far_candidates,
            random_far,
        )

        extra_far_count = min(
            remaining,
            len(unused_far),
        )

        if extra_far_count > 0:
            extra_far = np.sort(
                rng.choice(
                    unused_far,
                    size=extra_far_count,
                    replace=False,
                )
            )

            random_far = np.sort(
                np.concatenate(
                    [random_far, extra_far]
                )
            )

    if (
        len(nearby)
        + len(random_far)
        != desired_count
    ):
        raise ValueError(
            "Could not construct requested conventional negative count."
        )

    return (
        nearby.astype(int),
        random_far.astype(int),
    )


# ============================================================
# 8. TRAINING DATASET
# ============================================================

class ThreeSliceDataset(Dataset):

    def __init__(
        self,
        inputs,
        masks,
    ):
        self.inputs = np.asarray(
            inputs,
            dtype=np.float16,
        )

        self.masks = np.asarray(
            masks,
            dtype=np.uint8,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(
        self,
        index,
    ):
        x = torch.from_numpy(
            self.inputs[index].astype(np.float32)
        )

        y = torch.from_numpy(
            self.masks[index].astype(np.float32)
        ).unsqueeze(0)

        return x, y


def build_training_dataset():

    rng = np.random.default_rng(SEED)

    inputs = []
    masks = []
    metadata_rows = []

    total_positive = 0
    total_mined_hard = 0
    total_nearby = 0
    total_random = 0

    print("\nBuilding Experiment 09B training dataset...")

    for patient_name in TRAIN_PATIENTS:

        ct, mask, _ = load_patient(
            patient_name
        )

        foreground_per_slice = np.sum(
            mask,
            axis=(0, 1),
        )

        positive_indices = np.where(
            foreground_per_slice > MIN_POSITIVE_PIXELS
        )[0].astype(int)

        if len(positive_indices) == 0:
            raise ValueError(
                f"{patient_name}: no meaningful positive slices."
            )

        mined_hard_indices = np.array(
            hard_negative_indices_by_patient[
                patient_name
            ],
            dtype=int,
        )

        # Audit every mined slice against the current mask.
        for idx in mined_hard_indices:
            if idx < 0 or idx >= mask.shape[2]:
                raise ValueError(
                    f"{patient_name}: mined slice {idx} outside volume."
                )

            if foreground_per_slice[idx] != 0:
                raise ValueError(
                    f"{patient_name}: mined hard-negative slice {idx} "
                    "is not empty in the reference mask."
                )

        desired_total_negatives = int(
            round(
                len(positive_indices)
                * NEGATIVE_TO_POSITIVE_RATIO
            )
        )

        if len(mined_hard_indices) > desired_total_negatives:
            raise ValueError(
                f"{patient_name}: too many mined negatives "
                f"({len(mined_hard_indices)}) for total negative quota "
                f"({desired_total_negatives})."
            )

        conventional_count = (
            desired_total_negatives
            - len(mined_hard_indices)
        )

        (
            nearby_indices,
            random_indices,
        ) = select_conventional_negatives(
            mask=mask,
            positive_indices=positive_indices,
            mined_hard_indices=mined_hard_indices,
            desired_count=conventional_count,
            rng=rng,
        )

        all_negative_indices = np.concatenate(
            [
                mined_hard_indices,
                nearby_indices,
                random_indices,
            ]
        )

        if len(np.unique(all_negative_indices)) != len(all_negative_indices):
            raise ValueError(
                f"{patient_name}: duplicate negative slices after selection."
            )

        total_positive += len(positive_indices)
        total_mined_hard += len(mined_hard_indices)
        total_nearby += len(nearby_indices)
        total_random += len(random_indices)

        print(
            f"{patient_name}: "
            f"{len(positive_indices)} positive + "
            f"{len(mined_hard_indices)} mined hard + "
            f"{len(nearby_indices)} nearby conventional + "
            f"{len(random_indices)} random conventional"
        )

        sample_specs = (
            [
                (idx, "positive")
                for idx in positive_indices
            ]
            + [
                (idx, "mined_hard_negative")
                for idx in mined_hard_indices
            ]
            + [
                (idx, "nearby_conventional_negative")
                for idx in nearby_indices
            ]
            + [
                (idx, "random_conventional_negative")
                for idx in random_indices
            ]
        )

        for center_index, sample_type in sample_specs:

            inputs.append(
                build_three_slice_input(
                    ct,
                    int(center_index),
                )
            )

            masks.append(
                mask[
                    :,
                    :,
                    int(center_index),
                ]
            )

            metadata_rows.append(
                {
                    "patient":
                        patient_name,

                    "center_slice_index":
                        int(center_index),

                    "previous_slice_index":
                        int(
                            max(
                                0,
                                int(center_index) - 1,
                            )
                        ),

                    "next_slice_index":
                        int(
                            min(
                                ct.shape[2] - 1,
                                int(center_index) + 1,
                            )
                        ),

                    "sample_type":
                        sample_type,

                    "reference_pixels":
                        int(
                            mask[
                                :,
                                :,
                                int(center_index),
                            ].sum()
                        ),
                }
            )

    total_negative = (
        total_mined_hard
        + total_nearby
        + total_random
    )

    print("\n" + "=" * 84)
    print("EXPERIMENT 09B TRAINING DATA SUMMARY")
    print("=" * 84)

    print(
        f"Positive center slices:             {total_positive}"
    )

    print(
        f"Model-mined hard negatives:         {total_mined_hard}"
    )

    print(
        f"Nearby conventional negatives:      {total_nearby}"
    )

    print(
        f"Random conventional negatives:      {total_random}"
    )

    print(
        f"Total negative center slices:       {total_negative}"
    )

    print(
        f"Total 2.5D training samples:        {len(inputs)}"
    )

    print(
        "Actual negative:positive ratio:      "
        f"{total_negative / total_positive:.3f}:1"
    )

    print(
        "Hard negatives as fraction negatives: "
        f"{total_mined_hard / total_negative:.3f}"
    )

    with SELECTED_SLICES_CSV.open(
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
        writer.writerows(metadata_rows)

    return ThreeSliceDataset(
        inputs,
        masks,
    )


# ============================================================
# 9. MODEL / LOSS / OPTIMIZER
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 84)
print("EXPERIMENT 09B — 2.5D U-NET + HARD-NEGATIVE MINING")
print("=" * 84)
print(f"Training device: {device}")
print("Model initialization: FROM SCRATCH")

model = UNet(
    spatial_dims=2,
    in_channels=3,
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
# 10. WHOLE-VOLUME INFERENCE
# ============================================================

def predict_full_volume(
    ct: np.ndarray,
) -> np.ndarray:

    prediction = np.zeros(
        ct.shape,
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

            batch_np = np.stack(
                [
                    build_three_slice_input(
                        ct,
                        center_index,
                    )
                    for center_index
                    in range(start, end)
                ],
                axis=0,
            )

            batch_tensor = torch.from_numpy(
                batch_np.astype(np.float32)
            ).to(device)

            logits = model(
                batch_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            batch_prediction = (
                probabilities[:, 0]
                .cpu()
                .numpy()
                > PREDICTION_THRESHOLD
            ).astype(np.uint8)

            for local_index, center_index in enumerate(
                range(start, end)
            ):
                prediction[
                    :,
                    :,
                    center_index,
                ] = batch_prediction[
                    local_index
                ]

    return prediction


# ============================================================
# 11. METRICS
# ============================================================

def range_string(indices: np.ndarray) -> str:
    if len(indices) == 0:
        return "None"

    return (
        f"{int(indices.min())}-"
        f"{int(indices.max())}"
    )


def calculate_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
):
    prediction_bool = prediction.astype(bool)
    reference_bool = reference.astype(bool)

    tp = int(
        np.logical_and(
            prediction_bool,
            reference_bool,
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction_bool,
            np.logical_not(reference_bool),
        ).sum()
    )

    fn = int(
        np.logical_and(
            np.logical_not(prediction_bool),
            reference_bool,
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
            int(reference.sum()),

        "predicted_voxels":
            int(prediction.sum()),

        "true_positive_voxels":
            tp,

        "false_positive_voxels":
            fp,

        "false_negative_voxels":
            fn,

        "reference_positive_slice_count":
            int(len(reference_positive)),

        "predicted_positive_slice_count":
            int(len(prediction_positive)),

        "reference_slice_range":
            range_string(reference_positive),

        "predicted_slice_range":
            range_string(prediction_positive),

        "false_positive_slice_count":
            int(len(false_positive_slices)),

        "missed_reference_slice_count":
            int(len(missed_reference_slices)),
    }


def validate_patient(
    patient_name: str,
):
    ct, reference, _ = load_patient(
        patient_name
    )

    prediction = predict_full_volume(
        ct
    )

    return calculate_metrics(
        prediction,
        reference,
    )


def validate_all(epoch: int):

    patient_results = {
        patient_name:
            validate_patient(patient_name)

        for patient_name
        in VAL_PATIENTS
    }

    mean_dice = float(
        np.mean(
            [
                result["dice"]
                for result
                in patient_results.values()
            ]
        )
    )

    file_exists = VAL_DETAILS_CSV.exists()

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
                    "epoch": epoch,
                    "patient": patient_name,
                    **result,
                }
            )

    return mean_dice, patient_results


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
                "09B",

            "name":
                "2.5D U-Net with diversified model-mined hard negatives",

            "source_of_hard_negative_selection":
                str(HARD_NEGATIVE_CSV),

            "initialization":
                "from scratch",

            "input":
                "[z-1, z, z+1] axial CT slices",

            "target":
                "pancreas mask of center slice z",

            "spatial_resampling":
                False,

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
                [HU_MIN, HU_MAX],

            "negative_to_positive_ratio":
                NEGATIVE_TO_POSITIVE_RATIO,

            "conventional_nearby_fraction":
                CONVENTIONAL_NEARBY_FRACTION,

            "nearby_negative_margin_slices":
                NEARBY_NEGATIVE_MARGIN_SLICES,

            "prediction_threshold":
                PREDICTION_THRESHOLD,

            "postprocessing":
                "none during validation",

            "model":
                {
                    "name":
                        "MONAI UNet",

                    "spatial_dims":
                        2,

                    "in_channels":
                        3,

                    "out_channels":
                        1,

                    "channels":
                        [16, 32, 64, 128],

                    "strides":
                        [2, 2, 2],

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

print("\n" + "=" * 84)
print("STARTING EXPERIMENT 09B TRAINING")
print("=" * 84)

for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_loss = 0.0

    for batch_inputs, batch_masks in train_loader:

        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)

        optimizer.zero_grad()

        logits = model(batch_inputs)

        loss = loss_function(
            logits,
            batch_masks,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(train_loader)
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

        "Pat_11_fp_slices":
            "",

        "Pat_11_pred_range":
            "",

        "Pat_12_dice":
            "",

        "Pat_12_precision":
            "",

        "Pat_12_recall":
            "",

        "Pat_12_fp_slices":
            "",

        "Pat_12_pred_range":
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
        ) = validate_all(epoch)

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
                f"{patient_name}_fp_slices"
            ] = result[
                "false_positive_slice_count"
            ]

            history_row[
                f"{patient_name}_pred_range"
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
                "  NEW BEST EXPERIMENT 09B MODEL SAVED: "
                f"mean Dice="
                f"{best_mean_validation_dice:.4f}"
            )

        else:
            checks_without_improvement += 1

        history.append(history_row)

        if (
            checks_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print("\nEarly stopping triggered.")
            break

    else:
        print()
        history.append(history_row)


# ============================================================
# 15. SAVE LAST MODEL + HISTORY
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
    writer.writerows(history)


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
    if row["mean_validation_dice"] != ""
]

validation_y = [
    float(row["mean_validation_dice"])
    for row in history
    if row["mean_validation_dice"] != ""
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

ax1.set_xlabel("Epoch")

ax1.set_ylabel(
    "Dice + BCE Training Loss"
)

ax1.grid(True)

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
    "Experiment 09B - 2.5D U-Net + Hard-Negative Mining\n"
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

print("\n" + "=" * 84)
print("EXPERIMENT 09B COMPLETE")
print("=" * 84)

print(
    f"Best epoch:         {best_epoch}"
)

print(
    f"Best mean Val Dice: "
    f"{best_mean_validation_dice:.4f}"
)

print(
    f"Best model:         {BEST_MODEL_PATH}"
)

print(
    f"Training history:   {HISTORY_CSV}"
)

print(
    f"Validation details: {VAL_DETAILS_CSV}"
)

print(
    f"Selected slices:    {SELECTED_SLICES_CSV}"
)

print(
    f"Config:             {CONFIG_JSON}"
)

print(
    f"Curve:              {CURVE_PNG}"
)

print("\nPRIMARY COMPARISON — EXPERIMENT 08")
print(
    "Experiment 08 best mean Dice = 0.3079"
)
print(
    "Pat_11: Dice=0.0823, Precision=0.0792, Recall=0.0856, FP slices=228"
)
print(
    "Pat_12: Dice=0.5336, Precision=0.6058, Recall=0.4768, FP slices=111"
)

print("\nSECONDARY REFERENCE — EXPERIMENT 06A")
print(
    "Experiment 06A mean Dice = 0.2698"
)