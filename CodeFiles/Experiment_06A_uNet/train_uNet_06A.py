from __future__ import annotations
import csv, json, random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset

# ============================================================
# EXPERIMENT 06A
# Question: does 1:1 negative:positive slice sampling reduce
# whole-volume false positives?
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")
RESULTS_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_06A_results/"
    r"Experiment_06A_1to1_Negative_Sampling"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATIENTS = [
    "Pat_01", "Pat_03", "Pat_04", "Pat_05",
    "Pat_06", "Pat_07", "Pat_09", "Pat_10"
]
VAL_PATIENTS = ["Pat_11", "Pat_12"]
FINAL_TEST_PATIENTS = ["Pat_13", "Pat_14", "Pat_15"]

SEED = 42
EPOCHS = 40
BATCH_SIZE = 8
LR = 1e-3
VAL_EVERY = 2
EARLY_STOP_PATIENCE = 8

HU_MIN, HU_MAX = -100.0, 250.0
MIN_POSITIVE_PIXELS = 50
PRED_THRESHOLD = 0.5

# Main Experiment 06A change:
NEGATIVE_TO_POSITIVE_RATIO = 1.0
HARD_NEGATIVE_FRACTION = 0.50
NEARBY_MARGIN = 30

BEST_MODEL = RESULTS_DIR / "Experiment_06A_best_model.pth"
LAST_MODEL = RESULTS_DIR / "Experiment_06A_last_model.pth"
HISTORY_CSV = RESULTS_DIR / "Experiment_06A_training_history.csv"
VAL_CSV = RESULTS_DIR / "Experiment_06A_validation_details.csv"
SLICE_CSV = RESULTS_DIR / "Experiment_06A_selected_training_slices.csv"
CONFIG_JSON = RESULTS_DIR / "Experiment_06A_config.json"
SPLIT_JSON = RESULTS_DIR / "Experiment_06A_split.json"
CURVE_PNG = RESULTS_DIR / "Experiment_06A_training_validation_curve.png"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

assert not (set(TRAIN_PATIENTS) & set(VAL_PATIENTS))
assert not (set(TRAIN_PATIENTS) & set(FINAL_TEST_PATIENTS))
assert not (set(VAL_PATIENTS) & set(FINAL_TEST_PATIENTS))


def is_nifti(p: Path):
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def find_file(patient_dir: Path, kind: str):
    files = [p for p in patient_dir.iterdir() if p.is_file() and is_nifti(p)]

    if kind == "ct":
        exact = {
            "origional_images.nii", "origional_images.nii.gz",
            "original_images.nii", "original_images.nii.gz"
        }
        for p in files:
            if p.name.lower() in exact:
                return p
        for p in files:
            n = p.name.lower()
            if ("origional" in n or "original" in n) and "mask" not in n and "prediction" not in n:
                return p

    if kind == "mask":
        exact = {
            "masked_images.nii", "masked_images.nii.gz",
            "mask_images.nii", "mask_images.nii.gz"
        }
        for p in files:
            if p.name.lower() in exact:
                return p
        for p in files:
            n = p.name.lower()
            if ("masked_images" in n or "mask_images" in n) and "prediction" not in n:
                return p

    raise FileNotFoundError(f"Could not find {kind} file in {patient_dir}")


def preprocess_ct(ct):
    ct = np.clip(ct, HU_MIN, HU_MAX)
    return ((ct - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def load_patient(name):
    folder = BASE_DIR / name
    ct_img = nib.load(str(find_file(folder, "ct")))
    mask_img = nib.load(str(find_file(folder, "mask")))

    ct = ct_img.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata()

    if ct.shape != mask.shape:
        raise ValueError(f"{name}: shape mismatch {ct.shape} vs {mask.shape}")
    if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError(f"{name}: affine mismatch")

    orientation = nib.aff2axcodes(ct_img.affine)
    if orientation[2] not in ("S", "I"):
        raise ValueError(f"{name}: axis 2 is not axial: {orientation}")

    return preprocess_ct(ct), (mask > 0).astype(np.uint8)


def select_slices(mask, rng):
    counts = mask.sum(axis=(0, 1))
    positive = np.where(counts > MIN_POSITIVE_PIXELS)[0]
    empty = np.where(counts == 0)[0]

    if len(positive) == 0:
        raise ValueError("No positive pancreas slices")

    wanted_neg = min(
        int(round(len(positive) * NEGATIVE_TO_POSITIVE_RATIO)),
        len(empty)
    )

    first, last = int(positive.min()), int(positive.max())
    low = max(0, first - NEARBY_MARGIN)
    high = min(mask.shape[2] - 1, last + NEARBY_MARGIN)

    nearby = empty[(empty >= low) & (empty <= high)]
    far = np.setdiff1d(empty, nearby)

    wanted_hard = int(round(wanted_neg * HARD_NEGATIVE_FRACTION))
    n_hard = min(wanted_hard, len(nearby))

    hard = (
        np.sort(rng.choice(nearby, n_hard, replace=False))
        if n_hard > 0 else np.array([], dtype=int)
    )

    remaining = wanted_neg - len(hard)
    n_far = min(remaining, len(far))
    random_neg = (
        np.sort(rng.choice(far, n_far, replace=False))
        if n_far > 0 else np.array([], dtype=int)
    )

    remaining = wanted_neg - len(hard) - len(random_neg)
    if remaining > 0:
        unused_nearby = np.setdiff1d(nearby, hard)
        n_extra = min(remaining, len(unused_nearby))
        if n_extra > 0:
            extra = np.sort(rng.choice(unused_nearby, n_extra, replace=False))
            random_neg = np.sort(np.concatenate([random_neg, extra]))

    return positive.astype(int), hard.astype(int), random_neg.astype(int)


class SliceDataset(Dataset):
    def __init__(self, images, masks):
        self.images = np.asarray(images, dtype=np.float16)
        self.masks = np.asarray(masks, dtype=np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i].astype(np.float32)).unsqueeze(0)
        y = torch.from_numpy(self.masks[i].astype(np.float32)).unsqueeze(0)
        return x, y


def build_training_data():
    rng = np.random.default_rng(SEED)
    images, masks, metadata = [], [], []

    pos_total = hard_total = random_total = 0
    print("\nBuilding Experiment 06A dataset...")

    for patient in TRAIN_PATIENTS:
        ct, mask = load_patient(patient)
        pos, hard, random_neg = select_slices(mask, rng)

        pos_total += len(pos)
        hard_total += len(hard)
        random_total += len(random_neg)

        print(
            f"{patient}: {len(pos)} positive + "
            f"{len(hard)} hard negative + {len(random_neg)} random negative"
        )

        for idx in pos:
            images.append(ct[:, :, idx])
            masks.append(mask[:, :, idx])
            metadata.append([patient, int(idx), "positive", int(mask[:, :, idx].sum())])

        for idx in hard:
            images.append(ct[:, :, idx])
            masks.append(mask[:, :, idx])
            metadata.append([patient, int(idx), "hard_negative", 0])

        for idx in random_neg:
            images.append(ct[:, :, idx])
            masks.append(mask[:, :, idx])
            metadata.append([patient, int(idx), "random_negative", 0])

    neg_total = hard_total + random_total

    print("\nTRAINING DATA SUMMARY")
    print(f"Positive slices:        {pos_total}")
    print(f"Hard negative slices:   {hard_total}")
    print(f"Random negative slices: {random_total}")
    print(f"Total negative slices:  {neg_total}")
    print(f"Total slices:           {len(images)}")
    print(f"Actual neg:pos ratio:   {neg_total / pos_total:.3f}:1")

    with SLICE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "slice_index", "slice_type", "foreground_pixels"])
        w.writerows(metadata)

    return SliceDataset(images, masks)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

loss_fn = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def range_text(indices):
    return "None" if len(indices) == 0 else f"{int(indices.min())}-{int(indices.max())}"


def validate_patient(patient):
    ct, ref = load_patient(patient)
    pred = np.zeros_like(ref, dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for start in range(0, ct.shape[2], BATCH_SIZE):
            end = min(start + BATCH_SIZE, ct.shape[2])

            x_np = np.moveaxis(ct[:, :, start:end], 2, 0)
            x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(1).to(device)

            probs = torch.sigmoid(model(x))
            batch_pred = (probs[:, 0].cpu().numpy() > PRED_THRESHOLD).astype(np.uint8)
            pred[:, :, start:end] = np.moveaxis(batch_pred, 0, 2)

    pb, rb = pred.astype(bool), ref.astype(bool)
    tp = int(np.logical_and(pb, rb).sum())
    fp = int(np.logical_and(pb, ~rb).sum())
    fn = int(np.logical_and(~pb, rb).sum())

    eps = 1e-8
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    ref_counts = ref.sum(axis=(0, 1))
    pred_counts = pred.sum(axis=(0, 1))

    ref_pos = np.where(ref_counts > 0)[0]
    pred_pos = np.where(pred_counts > 0)[0]

    fp_slices = np.setdiff1d(pred_pos, ref_pos)
    missed_slices = np.setdiff1d(ref_pos, pred_pos)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "reference_voxels": int(ref.sum()),
        "predicted_voxels": int(pred.sum()),
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "reference_slice_range": range_text(ref_pos),
        "predicted_slice_range": range_text(pred_pos),
        "reference_positive_slices": int(len(ref_pos)),
        "predicted_positive_slices": int(len(pred_pos)),
        "false_positive_slices": int(len(fp_slices)),
        "missed_reference_slices": int(len(missed_slices)),
    }


def validate_all(epoch):
    results = {p: validate_patient(p) for p in VAL_PATIENTS}
    mean_dice = float(np.mean([r["dice"] for r in results.values()]))

    exists = VAL_CSV.exists()
    with VAL_CSV.open("a", newline="", encoding="utf-8") as f:
        fields = [
            "epoch", "patient", "dice", "iou", "precision", "recall",
            "reference_voxels", "predicted_voxels",
            "false_positive_voxels", "false_negative_voxels",
            "reference_slice_range", "predicted_slice_range",
            "reference_positive_slices", "predicted_positive_slices",
            "false_positive_slices", "missed_reference_slices"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()

        for p, r in results.items():
            w.writerow({"epoch": epoch, "patient": p, **r})

    return mean_dice, results


with SPLIT_JSON.open("w", encoding="utf-8") as f:
    json.dump({
        "train": TRAIN_PATIENTS,
        "validation": VAL_PATIENTS,
        "final_test_UNTOUCHED": FINAL_TEST_PATIENTS,
        "excluded_bad_masks": ["Pat_02", "Pat_08"],
    }, f, indent=2)

with CONFIG_JSON.open("w", encoding="utf-8") as f:
    json.dump({
        "experiment": "06A",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "HU_window": [HU_MIN, HU_MAX],
        "loss": "DiceCELoss(sigmoid=True)",
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "hard_negative_fraction": HARD_NEGATIVE_FRACTION,
        "nearby_margin": NEARBY_MARGIN,
        "prediction_threshold": PRED_THRESHOLD,
    }, f, indent=2)


train_dataset = build_training_data()
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

history = []
best_mean_dice = -1.0
best_epoch = None
checks_without_improvement = 0

print("\nSTARTING EXPERIMENT 06A")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    row = {
        "epoch": epoch,
        "training_loss": avg_loss,
        "mean_validation_dice": "",
        "Pat_11_dice": "",
        "Pat_11_precision": "",
        "Pat_11_recall": "",
        "Pat_11_false_positive_slices": "",
        "Pat_11_predicted_slice_range": "",
        "Pat_12_dice": "",
        "Pat_12_precision": "",
        "Pat_12_recall": "",
        "Pat_12_false_positive_slices": "",
        "Pat_12_predicted_slice_range": "",
    }

    print(f"Epoch {epoch:02d}/{EPOCHS} - Train Loss: {avg_loss:.4f}", end="")

    if epoch % VAL_EVERY == 0:
        mean_dice, results = validate_all(epoch)
        row["mean_validation_dice"] = mean_dice

        print(f" - Mean Val Dice: {mean_dice:.4f}")

        for p, r in results.items():
            row[f"{p}_dice"] = r["dice"]
            row[f"{p}_precision"] = r["precision"]
            row[f"{p}_recall"] = r["recall"]
            row[f"{p}_false_positive_slices"] = r["false_positive_slices"]
            row[f"{p}_predicted_slice_range"] = r["predicted_slice_range"]

            print(
                f"  {p}: Dice={r['dice']:.4f}, "
                f"Precision={r['precision']:.4f}, Recall={r['recall']:.4f}"
            )
            print(
                f"     Ref range={r['reference_slice_range']} | "
                f"Pred range={r['predicted_slice_range']} | "
                f"FP slices={r['false_positive_slices']} | "
                f"Missed ref slices={r['missed_reference_slices']}"
            )

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_epoch = epoch
            checks_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"  NEW BEST MODEL SAVED: mean Dice={best_mean_dice:.4f}")
        else:
            checks_without_improvement += 1

        history.append(row)

        if checks_without_improvement >= EARLY_STOP_PATIENCE:
            print("\nEarly stopping triggered.")
            break
    else:
        print()
        history.append(row)

torch.save(model.state_dict(), LAST_MODEL)

fields = list(history[0].keys())
with HISTORY_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(history)

epochs_x = [r["epoch"] for r in history]
loss_y = [r["training_loss"] for r in history]
val_x = [r["epoch"] for r in history if r["mean_validation_dice"] != ""]
val_y = [float(r["mean_validation_dice"]) for r in history if r["mean_validation_dice"] != ""]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(epochs_x, loss_y, marker="o", markersize=3, label="Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Dice + BCE Training Loss")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(val_x, val_y, marker="s", markersize=5, label="Validation Dice")
ax2.set_ylabel("Whole-volume Validation Dice")

l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="center right")

plt.title(
    "Experiment 06A - 1:1 Negative Sampling\n"
    "Training Loss and Whole-volume Validation Dice"
)
plt.tight_layout()
plt.savefig(CURVE_PNG, dpi=300, bbox_inches="tight")
plt.show()

print("\nEXPERIMENT 06A COMPLETE")
print(f"Best epoch:         {best_epoch}")
print(f"Best mean Val Dice: {best_mean_dice:.4f}")
print(f"Best model:         {BEST_MODEL}")
print(f"History:            {HISTORY_CSV}")
print(f"Validation details: {VAL_CSV}")
print(f"Selected slices:    {SLICE_CSV}")
print(f"Curve:              {CURVE_PNG}")
print("\nDO NOT evaluate Pat_13, Pat_14, or Pat_15 yet.")