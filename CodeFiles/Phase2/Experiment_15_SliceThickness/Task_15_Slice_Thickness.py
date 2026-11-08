"""Experiment 15 - does slice thickness change the DiNTS segmentation?

Same patient, scanner and contrast phase; only the slice thickness changes.
Thin-slice scans (<= 1 mm) where DiNTS worked are made thicker synthetically:
each thick slice is the mean of the n thin slices it covers (n = thickness /
original spacing), which approximates a thick reconstruction. The TotalSegmentator
pancreas is taken from the centre thin slice of each slab, for the crop.

DiNTS is run on the thin original and on each thick version with
    S0_full      : the bundle pipeline (answers the professor's question)
    S3_crop_tta  : crop to the TotalSegmentator pancreas + flip TTA (Exp 14)
and every thick result is upsampled back to the thin grid (each thick slice
repeated n times) and compared with the thin result of the same strategy.

These are changes relative to the model's own thin-slice output, not accuracy.

Run on the GPU node (SLURM): see run_15.sbatch
"""
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "CodeFiles" / "Phase2" / "Experiment_14_RescueFailures"))

from Task_14_Rescue_Failures import (  # noqa: E402
    CT_ROOT,
    PREPROCESSING,
    TOTALSEG_PANCREAS,
    TOTALSEG_ROOT,
    dice,
    load_network,
    run_strategy,
    to_original_space,
)

OUTPUT_ROOT = ROOT / "Results" / "_Phase2_Experiment15_SliceThickness"

# thin-slice (<= 1 mm) cases where DiNTS worked, spread over tumor size, manufacturer and phase
PATIENTS = ["PAN_38", "PAN_12", "PAN_26", "PAN_32", "PAN_29", "PAN_31", "PAN_36", "PAN_24"]

THICKNESS_MM = [2.0, 3.0, 5.0]

STRATEGIES = ["S0_full", "S3_crop_tta"]

PANCREAS, TUMOR = 1, 2


def thicken(array, n, reduce):
    """Combine every n slices along z (axis 2); the remainder slices are dropped."""

    m = array.shape[2] // n

    slabs = array[:, :, : m * n].reshape(array.shape[0], array.shape[1], m, n)

    return slabs.mean(axis=-1) if reduce == "mean" else slabs[..., (n - 1) // 2]


def thick_affine(affine, n):
    """Slice spacing x n; origin moved to the centre of the first slab."""

    new = affine.copy()

    new[:3, 3] = affine[:3, 3] + affine[:3, 2] * (n - 1) / 2

    new[:3, 2] = affine[:3, 2] * n

    return new


def make_thick_inputs(patient, n, out_dir):

    ct_img = nib.load(str(CT_ROOT / patient / "baseline_ct.nii.gz"))

    ts_img = nib.load(str(TOTALSEG_ROOT / patient / "totalseg_roi.nii.gz"))

    affine = thick_affine(ct_img.affine, n)

    ct = thicken(np.asanyarray(ct_img.dataobj).astype(np.float32), n, "mean")

    ts = thicken(np.asanyarray(ts_img.dataobj), n, "centre")

    ct_path = out_dir / f"ct_n{n}.nii.gz"

    ts_path = out_dir / f"totalseg_n{n}.nii.gz"

    nib.save(nib.Nifti1Image(np.rint(ct).astype(np.int16), affine), str(ct_path))

    nib.save(nib.Nifti1Image(ts.astype(np.uint8), affine), str(ts_path))

    return ct_path, ts_path


def segment(network, ct_path, ts_path, strategy):

    data = PREPROCESSING({"image": str(ct_path), "totalseg": str(ts_path)})

    prob = run_strategy(network, data, strategy)

    return None if prob is None else to_original_space(data, prob)


def compare(label, thin, ts_pancreas, voxel_ml):

    return {
        "pancreas_ml": round(float((label == PANCREAS).sum() * voxel_ml), 3),
        "tumor_candidate_ml": round(float((label == TUMOR).sum() * voxel_ml), 3),
        "dice_complex_vs_thin": round(dice(label > 0, thin > 0), 3),
        "dice_pancreas_vs_thin": round(dice(label == PANCREAS, thin == PANCREAS), 3),
        "dice_tumor_vs_thin": round(dice(label == TUMOR, thin == TUMOR), 3),
        "dice_complex_vs_totalseg": round(dice(label > 0, ts_pancreas), 3),
    }


def evaluate(patient, network):

    ct_img = nib.load(str(CT_ROOT / patient / "baseline_ct.nii.gz"))

    z_spacing = float(ct_img.header.get_zooms()[2])

    voxel_ml = float(np.prod(ct_img.header.get_zooms()[:3]) / 1000)

    ts_full = np.asanyarray(nib.load(str(TOTALSEG_ROOT / patient / "totalseg_roi.nii.gz")).dataobj) == TOTALSEG_PANCREAS

    out_dir = OUTPUT_ROOT / patient

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for strategy in STRATEGIES:

        start = time.perf_counter()

        thin_full = segment(network, CT_ROOT / patient / "baseline_ct.nii.gz", TOTALSEG_ROOT / patient / "totalseg_roi.nii.gz", strategy)

        nib.save(nib.Nifti1Image(thin_full, ct_img.affine), str(out_dir / f"dints_{strategy}_thin.nii.gz"))

        for thickness in THICKNESS_MM:

            n = max(int(round(thickness / z_spacing)), 1)

            ct_path, ts_path = make_thick_inputs(patient, n, out_dir)

            label = segment(network, ct_path, ts_path, strategy)

            nib.save(nib.Nifti1Image(label, nib.load(str(ct_path)).affine), str(out_dir / f"dints_{strategy}_n{n}.nii.gz"))

            # back to the thin grid; the thin references are cut to the same slices
            upsampled = np.repeat(label, n, axis=2)

            depth = upsampled.shape[2]

            thin = thin_full[:, :, :depth]

            row = {
                "patient": patient,
                "strategy": strategy,
                "target_thickness_mm": thickness,
                "effective_thickness_mm": round(n * z_spacing, 2),
                "slices_merged": n,
            }

            row.update(compare(upsampled, thin, ts_full[:, :, :depth], voxel_ml))

            rows.append(row)

            print(
                f"    {strategy:12s} {row['effective_thickness_mm']:4.1f} mm | "
                f"complex Dice vs thin {row['dice_complex_vs_thin']} | "
                f"tumor Dice vs thin {row['dice_tumor_vs_thin']} | "
                f"tumor {row['tumor_candidate_ml']} mL | "
                f"vs TotalSeg {row['dice_complex_vs_totalseg']}",
                flush=True,
            )

        thin_row = {"patient": patient, "strategy": strategy, "target_thickness_mm": "thin", "effective_thickness_mm": z_spacing, "slices_merged": 1}

        thin_row.update(compare(thin_full, thin_full, ts_full, voxel_ml))

        thin_row["runtime_sec"] = round(time.perf_counter() - start, 1)

        rows.append(thin_row)

        print(f"    {strategy:12s} thin   | tumor {thin_row['tumor_candidate_ml']} mL | vs TotalSeg {thin_row['dice_complex_vs_totalseg']}", flush=True)

    return rows


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    patients = sys.argv[1:] or PATIENTS

    print("=" * 80)
    print("EXPERIMENT 15")
    print("SYNTHETIC SLICE THICKNESS vs DiNTS")
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

    table_csv = OUTPUT_ROOT / "Phase2_Experiment15_SliceThickness.csv"

    table.to_csv(table_csv, index=False, encoding="utf-8-sig")

    thick = table[table.target_thickness_mm != "thin"]

    for column in ["dice_complex_vs_thin", "dice_tumor_vs_thin", "dice_complex_vs_totalseg"]:

        print(f"\nMedian {column} (rows: strategy, columns: target thickness mm):")

        print(thick.pivot_table(index="strategy", columns="target_thickness_mm", values=column, aggfunc="median").round(3).to_string())

    print("\nTumor candidate volume (mL) per patient:")

    print(table.pivot_table(index=["patient", "strategy"], columns="target_thickness_mm", values="tumor_candidate_ml", sort=False).to_string())

    print("\nResults:")
    print(table_csv)


if __name__ == "__main__":

    main()
