"""Experiment 14 - try to rescue the DiNTS failures by changing only the input.

Same pretrained DiNTS weights and preprocessing as the bundle (11A); the
network is called directly so the input can be changed. Strategies:

    S0_full      : whole scan, as the bundle does (must reproduce 11A)
    S1_pad       : whole scan padded with PAD_MM of air on every side
                   (PAN_14's pancreas lies at the inferior scan edge)
    S2_crop      : crop to the TotalSegmentator pancreas (12B) + CROP_MARGIN_MM,
                   run DiNTS inside the crop only, background elsewhere
    S3_crop_tta  : S2 averaged over flips along each axis (test-time augmentation)

Clipping the -2048 padding of PAN_14 is not tested: the bundle clips intensities
to [-87, 199] HU, so -2048 and -1024 already give the same network input.

Each output is compared with the TotalSegmentator pancreas and with the
original 11A mask. These are agreement measures, not accuracy.

Run on the GPU node (SLURM): see run_14.sbatch
"""
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import DiNTS, TopologyInstance
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "CodeFiles" / "Phase2" / "Experiment_13_MaskCleanup"))

from Task_13_Mask_Cleanup import TOTALSEG_PANCREAS, describe, dice  # noqa: E402


BUNDLE_ROOT = ROOT / "Models" / "pancreas_ct_dints_segmentation"

CT_ROOT = ROOT / "Results" / "_Phase2_Experiment10D_Baseline_NIfTI"

PRED_ROOT = ROOT / "Results" / "_Phase2_Experiment11A_MONAI_DiNTS"

TOTALSEG_ROOT = ROOT / "Results" / "_Phase2_Experiment12B_ContrastPhase"

OUTPUT_ROOT = ROOT / "Results" / "_Phase2_Experiment14_RescueFailures"

# failures (PAN_02, 08, 13, 14), borderline (PAN_07), controls that worked (PAN_03, 05, 34)
PATIENTS = ["PAN_02", "PAN_08", "PAN_13", "PAN_14", "PAN_07", "PAN_03", "PAN_05", "PAN_34"]

STRATEGIES = ["S0_full", "S1_pad", "S2_crop", "S3_crop_tta"]

PAD_MM = 48

CROP_MARGIN_MM = 40

ROI_SIZE = (96, 96, 96)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bundle preprocessing (configs/inference.yaml); the TotalSegmentator mask follows the same geometry
PREPROCESSING = Compose(
    [
        LoadImaged(keys=["image", "totalseg"]),
        EnsureChannelFirstd(keys=["image", "totalseg"]),
        Orientationd(keys=["image", "totalseg"], axcodes="RAS"),
        Spacingd(keys=["image", "totalseg"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=-87, a_max=199, b_min=0, b_max=1, clip=True),
    ]
)

INVERT = Invertd(
    keys="pred",
    transform=PREPROCESSING,
    orig_keys="image",
    nearest_interp=False,
    to_tensor=True,
)


def load_network():

    arch = torch.load(BUNDLE_ROOT / "models" / "search_code_18590.pt", map_location=DEVICE, weights_only=False)

    space = TopologyInstance(
        channel_mul=1,
        num_blocks=12,
        num_depths=4,
        use_downsample=True,
        arch_code=[arch["arch_code_a"], arch["arch_code_c"]],
        device=DEVICE,
    )

    network = DiNTS(
        dints_space=space,
        in_channels=1,
        num_classes=3,
        use_downsample=True,
        node_a=torch.from_numpy(arch["node_a"]),
    )

    state = torch.load(BUNDLE_ROOT / "models" / "model.pt", map_location=DEVICE, weights_only=False)

    network.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)

    return network.to(DEVICE).eval()


def predict(network, image, tta=False):
    """Softmax probabilities (3, X, Y, Z) on CPU for an image tensor (1, X, Y, Z)."""

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=4,
        overlap=0.625,
        sw_device=DEVICE,
        device=torch.device("cpu"),
    )

    flips = [None, (2,), (3,), (4,)] if tta else [None]

    total = None

    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):

        for axes in flips:

            x = image[None]

            x = torch.flip(x, axes) if axes else x

            prob = torch.softmax(inferer(x, network).float(), dim=1)

            prob = torch.flip(prob, axes) if axes else prob

            total = prob if total is None else total + prob

    return total[0] / len(flips)


def crop_box(totalseg):
    """Index box (1 mm grid) around the TotalSegmentator pancreas + CROP_MARGIN_MM."""

    coords = np.argwhere(totalseg[0].numpy() == TOTALSEG_PANCREAS)

    if len(coords) == 0:

        return None

    low = np.maximum(coords.min(axis=0) - CROP_MARGIN_MM, 0)

    high = np.minimum(coords.max(axis=0) + CROP_MARGIN_MM + 1, totalseg.shape[1:])

    return tuple(slice(a, b) for a, b in zip(low, high))


def run_strategy(network, data, strategy):

    image = data["image"]

    if strategy == "S0_full":

        return predict(network, image)

    if strategy == "S1_pad":

        p = PAD_MM

        padded = torch.nn.functional.pad(image, (p, p, p, p, p, p), value=0.0)

        return predict(network, padded)[:, p:-p, p:-p, p:-p]

    box = crop_box(data["totalseg"])

    if box is None:

        return None

    prob = torch.zeros((3,) + tuple(image.shape[1:]))

    prob[0] = 1.0

    prob[(slice(None),) + box] = predict(network, image[(slice(None),) + box], tta=strategy == "S3_crop_tta")

    return prob


def to_original_space(data, prob):

    pred = MetaTensor(prob, affine=data["image"].affine, meta=dict(data["image"].meta))

    inverted = INVERT({"image": data["image"], "pred": pred})["pred"]

    return torch.argmax(inverted, dim=0).numpy().astype(np.uint8)


def evaluate(patient, network):

    ct_path = CT_ROOT / patient / "baseline_ct.nii.gz"

    ts_path = TOTALSEG_ROOT / patient / "totalseg_roi.nii.gz"

    data = PREPROCESSING({"image": str(ct_path), "totalseg": str(ts_path)})

    ct_img = nib.load(str(ct_path))

    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=float)

    voxel_ml = float(np.prod(spacing) / 1000)

    ts_pancreas = np.asanyarray(nib.load(str(ts_path)).dataobj) == TOTALSEG_PANCREAS

    original = np.rint(np.asanyarray(nib.load(str(PRED_ROOT / patient / "baseline_ct" / "baseline_ct_trans.nii.gz")).dataobj)).astype(np.uint8)

    out_dir = OUTPUT_ROOT / patient

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    reference = {"patient": patient, "strategy": "11A_original", "runtime_sec": np.nan}

    reference.update(describe(original, ts_pancreas, spacing, voxel_ml, "mask"))

    rows.append(reference)

    for strategy in STRATEGIES:

        start = time.perf_counter()

        prob = run_strategy(network, data, strategy)

        if prob is None:

            rows.append({"patient": patient, "strategy": strategy, "error": "no TotalSegmentator pancreas"})

            continue

        label = to_original_space(data, prob)

        if label.shape != original.shape:

            raise ValueError(f"{strategy}: output shape {label.shape} != CT shape {original.shape}")

        nib.save(nib.Nifti1Image(label, ct_img.affine), str(out_dir / f"dints_{strategy}.nii.gz"))

        row = {"patient": patient, "strategy": strategy, "runtime_sec": round(time.perf_counter() - start, 1)}

        row.update(describe(label, ts_pancreas, spacing, voxel_ml, "mask"))

        row["dice_vs_11A_original"] = round(dice(label > 0, original > 0), 3)

        rows.append(row)

        print(
            f"    {strategy:12s} pancreas {row['mask_pancreas_ml']:7.1f} mL | "
            f"tumor {row['mask_tumor_candidate_ml']:6.2f} mL | "
            f"Dice vs TotalSeg {row['mask_dice_vs_totalseg']} | "
            f"vs 11A {row['dice_vs_11A_original']} | {row['runtime_sec']} s",
            flush=True,
        )

    return rows


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    patients = sys.argv[1:] or PATIENTS

    print("=" * 80)
    print("EXPERIMENT 14")
    print("RESCUE DiNTS FAILURES BY CHANGING THE INPUT")
    print("=" * 80)

    network = load_network()

    rows = []

    for number, patient in enumerate(patients, start=1):

        print(f"[{number:02d}/{len(patients)}] {patient}", flush=True)

        try:

            rows += evaluate(patient, network)

        except Exception as error:

            print(f"    ERROR: {error}", flush=True)

            rows.append({"patient": patient, "strategy": "ERROR", "error": str(error)})

        torch.cuda.empty_cache()

    table = pd.DataFrame(rows)

    table_csv = OUTPUT_ROOT / "Phase2_Experiment14_RescueFailures.csv"

    table.to_csv(table_csv, index=False, encoding="utf-8-sig")

    print("\nDice vs TotalSegmentator pancreas (agreement, not accuracy):")

    print(table.pivot_table(index="patient", columns="strategy", values="mask_dice_vs_totalseg", sort=False).to_string())

    print("\nTumor candidate volume (mL):")

    print(table.pivot_table(index="patient", columns="strategy", values="mask_tumor_candidate_ml", sort=False).to_string())

    print("\nResults:")
    print(table_csv)


if __name__ == "__main__":

    main()
