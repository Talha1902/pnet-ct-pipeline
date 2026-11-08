"""
Experiment 08 — 2.5D Three-Slice U-Net
======================================

Research question
-----------------
Does giving the 2D U-Net local through-plane context improve pancreas
segmentation generalization?

Experiment 08 keeps the strongest clean baseline setup from Experiment 06A
and changes only the model input:

Experiment 06A:
    input = current axial CT slice z

Experiment 08:
    input channel 0 = axial slice z-1
    input channel 1 = axial slice z
    input channel 2 = axial slice z+1

The target remains:
    pancreas mask for the CENTER slice z

This is a 2.5D approach:
- cheaper than a full 3D U-Net
- preserves the 2D model
- adds real neighboring anatomical context

Controlled design
-----------------
UNCHANGED from Experiment 06A:
- Training patients:
  Pat_01, Pat_03, Pat_04, Pat_05, Pat_06, Pat_07, Pat_09, Pat_10
- Validation:
  Pat_11, Pat_12
- Final untouched test:
  Pat_13, Pat_14, Pat_15
- No spatial resampling
- Original 512 x 512 matrix
- HU window [-100, 250]
- 1:1 negative:positive training slice ratio
- 50% nearby/hard negatives
- 50% farther/random negatives
- Dice + BCE loss
- Adam, LR = 0.001
- Batch size = 8
- Prediction threshold = 0.5
- Whole-volume validation every 2 epochs
- Early stopping by mean whole-volume validation Dice
- NO post-processing during validation

MAIN CHANGE:
    U-Net in_channels: 1 -> 3

IMPORTANT:
Pat_13, Pat_14, Pat_15 remain untouched.

Install if needed:
    pip install torch monai nibabel numpy matplotlib

Run:
    python train_experiment_08.py
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

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")

RESULTS_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_08_results/"
    r"Experiment_08_2p5D_Three_Slice_UNet"
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
HARD_NEGATIVE_FRACTION = 0.50

# Same style as Experiment 06A:
# nearby negatives are selected from around the true pancreas range.
NEARBY_NEGATIVE_MARGIN_SLICES = 30

NUM_WORKERS = 0

BEST_MODEL_PATH = RESULTS_DIR / "Experiment_08_best_model.pth"
LAST_MODEL_PATH = RESULTS_DIR / "Experiment_08_last_model.pth"

HISTORY_CSV = RESULTS_DIR / "Experiment_08_training_history.csv"
VAL_DETAILS_CSV = RESULTS_DIR / "Experiment_08_validation_details.csv"
SLICE_METADATA_CSV = RESULTS_DIR / "Experiment_08_selected_training_slices.csv"

CONFIG_JSON = RESULTS_DIR / "Experiment_08_config.json"
SPLIT_JSON = RESULTS_DIR / "Experiment_08_split.json"
CURVE_PNG = RESULTS_DIR / "Experiment_08_training_validation_curve.png"


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
# 4. LOAD ORIGINAL-PHYSICAL-GRID PATIENT
# ============================================================

def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    """
    Same preprocessing as Experiment 06A.
    No spatial resampling.
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

    return ct.astype(np.float32)


def load_patient(patient_name: str):
    patient_dir = BASE_DIR / patient_name

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

    if orientation[2] not in ("S", "I"):
        raise ValueError(
            f"{patient_name}: array axis 2 is not axial. "
            f"Orientation={orientation}"
        )

    ct = preprocess_ct(raw_ct)

    mask = (
        raw_mask > 0
    ).astype(np.uint8)

    return ct, mask, ct_img


# ============================================================
# 5. 2.5D THREE-SLICE INPUT CREATION
# ============================================================

def build_three_slice_input(
    ct_volume: np.ndarray,
    center_index: int,
) -> np.ndarray:
    """
    Build [3, H, W] input:
      channel 0 = z-1
      channel 1 = z
      channel 2 = z+1

    At volume boundaries, the nearest valid slice is repeated.

    Example:
      z = 0
        channels = [0, 0, 1]

      z = last
        channels = [last-1, last, last]
    """
    last_index = ct_volume.shape[2] - 1

    prev_index = max(
        0,
        center_index - 1,
    )

    next_index = min(
        last_index,
        center_index + 1,
    )

    three_slice = np.stack(
        [
            ct_volume[:, :, prev_index],
            ct_volume[:, :, center_index],
            ct_volume[:, :, next_index],
        ],
        axis=0,
    )

    return three_slice.astype(
        np.float32
    )


# ============================================================
# 6. NEGATIVE-SLICE SAMPLING
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
# 7. TRAINING DATASET
# ============================================================

class ThreeSliceDataset(Dataset):

    def __init__(
        self,
        inputs,
        masks,
    ):
        # [N,3,H,W]
        self.inputs = np.asarray(
            inputs,
            dtype=np.float16,
        )

        # [N,H,W]
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
            self.inputs[
                index
            ].astype(
                np.float32
            )
        )

        y = torch.from_numpy(
            self.masks[
                index
            ].astype(
                np.float32
            )
        ).unsqueeze(0)

        return x, y


def build_training_dataset():
    rng = np.random.default_rng(
        SEED
    )

    inputs = []
    masks = []
    metadata_rows = []

    positive_total = 0
    hard_negative_total = 0
    random_negative_total = 0

    print("\nBuilding Experiment 08 2.5D training dataset...")

    for patient_name in TRAIN_PATIENTS:

        ct, mask, _ = load_patient(
            patient_name
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

        for center_index, slice_type in selected:

            inputs.append(
                build_three_slice_input(
                    ct,
                    center_index,
                )
            )

            masks.append(
                mask[
                    :,
                    :,
                    center_index,
                ]
            )

            metadata_rows.append(
                {
                    "patient":
                        patient_name,

                    "previous_slice_index":
                        int(
                            max(
                                0,
                                center_index - 1,
                            )
                        ),

                    "center_slice_index":
                        int(center_index),

                    "next_slice_index":
                        int(
                            min(
                                ct.shape[2] - 1,
                                center_index + 1,
                            )
                        ),

                    "slice_type":
                        slice_type,

                    "center_mask_foreground_pixels":
                        int(
                            mask[
                                :,
                                :,
                                center_index,
                            ].sum()
                        ),
                }
            )

    negative_total = (
        hard_negative_total
        + random_negative_total
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 08 TRAINING DATA SUMMARY")
    print("=" * 78)

    print(
        f"Positive center slices:        "
        f"{positive_total}"
    )

    print(
        f"Hard negative center slices:   "
        f"{hard_negative_total}"
    )

    print(
        f"Random negative center slices: "
        f"{random_negative_total}"
    )

    print(
        f"Total negative center slices:  "
        f"{negative_total}"
    )

    print(
        f"Total 2.5D samples:            "
        f"{len(inputs)}"
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

    return ThreeSliceDataset(
        inputs,
        masks,
    )


# ============================================================
# 8. MODEL / LOSS / OPTIMIZER
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 78)
print("EXPERIMENT 08 — 2.5D THREE-SLICE U-NET")
print("=" * 78)
print(f"Training device: {device}")

model = UNet(
    spatial_dims=2,

    # MAIN EXPERIMENT 08 CHANGE
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
# 9. WHOLE-VOLUME 2.5D INFERENCE
# ============================================================

def predict_full_volume(
    ct: np.ndarray,
) -> np.ndarray:
    """
    Predict every center slice using:
        [z-1, z, z+1]
    """
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

            batch_inputs = []

            for center_index in range(
                start,
                end,
            ):
                batch_inputs.append(
                    build_three_slice_input(
                        ct,
                        center_index,
                    )
                )

            batch_np = np.stack(
                batch_inputs,
                axis=0,
            )

            batch_tensor = torch.from_numpy(
                batch_np.astype(
                    np.float32
                )
            ).to(device)

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

            for local_i, center_index in enumerate(
                range(
                    start,
                    end,
                )
            ):
                prediction[
                    :,
                    :,
                    center_index,
                ] = batch_prediction[
                    local_i
                ]

    return prediction


# ============================================================
# 10. WHOLE-VOLUME METRICS
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


def calculate_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
):
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


def validate_all(
    epoch: int,
):
    patient_results = {
        patient_name:
            validate_patient(
                patient_name
            )

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
# 11. SAVE CONFIG / SPLIT
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
                "08",

            "name":
                "2.5D Three-Slice U-Net",

            "research_question":
                "Does neighboring axial-slice context improve "
                "inter-patient pancreas segmentation generalization?",

            "input_channels":
                {
                    "channel_0":
                        "z-1 axial CT slice",

                    "channel_1":
                        "z center axial CT slice",

                    "channel_2":
                        "z+1 axial CT slice",
                },

            "target":
                "pancreas mask of center slice z",

            "boundary_handling":
                "repeat nearest valid slice",

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
# 12. BUILD TRAINING DATA
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
# 13. TRAINING LOOP
# ============================================================

history = []

best_mean_validation_dice = -1.0
best_epoch = None
checks_without_improvement = 0

print("\n" + "=" * 78)
print("STARTING EXPERIMENT 08 TRAINING")
print("=" * 78)

for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_loss = 0.0

    for batch_inputs, batch_masks in train_loader:

        batch_inputs = batch_inputs.to(
            device
        )

        batch_masks = batch_masks.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            batch_inputs
        )

        loss = loss_function(
            logits,
            batch_masks,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss
        / len(train_loader)
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
        ) = validate_all(
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
                "  NEW BEST EXPERIMENT 08 MODEL SAVED: "
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
# 14. SAVE LAST MODEL / HISTORY
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
# 15. TRAINING / VALIDATION CURVE
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
    "Experiment 08 - 2.5D Three-Slice U-Net\n"
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
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("EXPERIMENT 08 COMPLETE")
print("=" * 78)

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
    "\nCompare directly against RAW Experiment 06A:"
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
    "\nExperiment 07 spatial-standardization reference:"
)

print(
    "Pat_11: Dice=0.0425, "
    "Precision=0.0354, Recall=0.0530"
)

print(
    "Pat_12: Dice=0.4693, "
    "Precision=0.5861, Recall=0.3913"
)

print(
    "Mean Dice=0.2559"
)