"""Experiment 16 - second model: compare nnU-Net with DiNTS on all 38 patients.

nnU-Net (Isensee et al., v1 pretrained "Task007_Pancreas", 3d_fullres, 5-fold
ensemble) was trained on the same MSD Task07 data as DiNTS and predicts the same
labels (1 = pancreas, 2 = tumor). Predictions come from run_16.sbatch.

nnU-Net gets the same clean-up as DiNTS (Exp 13-B: keep components within 10 mm of the
TotalSegmentator pancreas, drop tumor specks, fill holes), so both models are compared on
equal terms. Without it nnU-Net's tumor label includes lesions in other organs (PAN_35:
a 108 mL lesion in the liver). The removed tumor volume and the TotalSegmentator organ
it lay in are reported.

Per patient, the two cleaned masks are compared:
    - Dice of the pancreas+tumor complex, of the pancreas label, of the tumor label
    - tumor candidate found by each model (>= MIN_TUMOR_ML) and whether the two
      candidates lie within TUMOR_MATCH_MM of each other (same lesion)
    - Dice of nnU-Net vs the TotalSegmentator pancreas (third opinion)
and a confidence level is assigned from the two-model agreement (see confidence).

All values are agreement between models, not accuracy.
"""
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "CodeFiles" / "Phase2" / "Experiment_13_MaskCleanup"))

from Task_13_Mask_Cleanup import TOTALSEG_PANCREAS, bounding_box, dice, distance_to, fill_holes, keep_anchored, remove_small_tumor  # noqa: E402


DINTS_ROOT = ROOT / "Results" / "_Phase2_Experiment13_MaskCleanup"

NNUNET_ROOT = ROOT / "Results" / "_Phase2_Experiment16_SecondModel" / "nnunet_predictions"

TOTALSEG_ROOT = ROOT / "Results" / "_Phase2_Experiment12B_ContrastPhase"

OUTPUT_ROOT = ROOT / "Results" / "_Phase2_Experiment16_SecondModel"

CASE_TABLE = (
    ROOT
    / "Results"
    / "_Phase2_Experiment12_CaseTable"
    / "Phase2_Experiment12A_CaseTable_All38.csv"
)

PATIENTS = [f"PAN_{i:02d}" for i in range(1, 39)]

PANCREAS, TUMOR = 1, 2

# a tumor candidate smaller than this counts as "no tumor found" (same as Exp 13)
MIN_TUMOR_ML = 0.05

# two tumor candidates closer than this are taken to be the same lesion
TUMOR_MATCH_MM = 10.0

# complex Dice thresholds for the confidence levels
HIGH_DICE, LOW_DICE = 0.70, 0.30

# labels of the TotalSegmentator ROI (12B), used to name where a removed lesion lay
TOTALSEG_ORGANS = {0: "none", 1: "spleen", 2: "kidney_right", 3: "kidney_left", 5: "liver", 7: "pancreas"}


def load(path):

    img = nib.load(str(path))

    return np.rint(np.asanyarray(img.dataobj)).astype(np.uint8), img


def tumor_match(a, b, spacing):
    """True if any voxel of tumor b lies within TUMOR_MATCH_MM of tumor a."""

    margin = np.ceil(TUMOR_MATCH_MM / spacing).astype(int) + 1

    box = bounding_box(a | b, margin)

    return bool((distance_to(a, spacing, box)[b[box]] <= TUMOR_MATCH_MM).any())


def confidence(row):
    """HIGH: both models agree on the organ and on the tumor; LOW: they disagree on either."""

    if row["dice_complex_dints_vs_nnunet"] < LOW_DICE:

        return "LOW_ORGAN_DISAGREE"

    if row["tumor_agreement"] in ("ONLY_DINTS", "ONLY_NNUNET", "DIFFERENT_LESIONS"):

        return "LOW_TUMOR_DISAGREE"

    if row["dice_complex_dints_vs_nnunet"] >= HIGH_DICE and (
        row["tumor_agreement"] == "NEITHER" or row["dice_tumor_dints_vs_nnunet"] >= 0.5
    ):

        return "HIGH"

    return "MEDIUM"


def compare(patient):

    dints, img = load(DINTS_ROOT / patient / "dints_clean.nii.gz")

    nnunet_raw, nn_img = load(NNUNET_ROOT / f"{patient}.nii.gz")

    if nnunet_raw.shape != dints.shape or not np.allclose(nn_img.affine, img.affine, atol=1e-3):

        raise ValueError(f"nnU-Net grid {nnunet_raw.shape} does not match DiNTS grid {dints.shape}")

    spacing = np.array(img.header.get_zooms()[:3], dtype=float)

    ts_roi = np.asanyarray(nib.load(str(TOTALSEG_ROOT / patient / "totalseg_roi.nii.gz")).dataobj).astype(np.int16)

    return {"patient": patient, **compare_masks(dints, nnunet_raw, ts_roi, spacing)[0]}


def compare_masks(dints, nnunet_raw, ts_roi, spacing):
    """Clean nnU-Net (13-B), compare it with the cleaned DiNTS mask; returns (row, cleaned nnU-Net).

    Also used by the frozen pipeline (Exp 17C).
    """

    voxel_ml = float(np.prod(spacing) / 1000)

    ts_pancreas = ts_roi == TOTALSEG_PANCREAS

    nnunet = keep_anchored(nnunet_raw, spacing, ts_pancreas)[0]

    nnunet = fill_holes(remove_small_tumor(nnunet, voxel_ml)[0])[0]

    removed_tumor = (nnunet_raw == TUMOR) & (nnunet == 0)

    removed_organ = TOTALSEG_ORGANS.get(int(np.bincount(ts_roi[removed_tumor].clip(0)).argmax()), "other") if removed_tumor.any() else ""

    d_tumor, n_tumor = dints == TUMOR, nnunet == TUMOR

    d_found = d_tumor.sum() * voxel_ml >= MIN_TUMOR_ML

    n_found = n_tumor.sum() * voxel_ml >= MIN_TUMOR_ML

    if d_found and n_found:

        agreement = "SAME_LESION" if tumor_match(d_tumor, n_tumor, spacing) else "DIFFERENT_LESIONS"

    elif d_found:

        agreement = "ONLY_DINTS"

    elif n_found:

        agreement = "ONLY_NNUNET"

    else:

        agreement = "NEITHER"

    row = {
        "dints_pancreas_ml": round(float((dints == PANCREAS).sum() * voxel_ml), 3),
        "dints_tumor_candidate_ml": round(float(d_tumor.sum() * voxel_ml), 3),
        "nnunet_pancreas_ml": round(float((nnunet == PANCREAS).sum() * voxel_ml), 3),
        "nnunet_tumor_candidate_ml": round(float(n_tumor.sum() * voxel_ml), 3),
        "nnunet_raw_tumor_candidate_ml": round(float((nnunet_raw == TUMOR).sum() * voxel_ml), 3),
        "nnunet_removed_tumor_ml": round(float(removed_tumor.sum() * voxel_ml), 3),
        "nnunet_removed_tumor_organ": removed_organ,
        "dice_complex_dints_vs_nnunet": round(dice(dints > 0, nnunet > 0), 3),
        "dice_pancreas_dints_vs_nnunet": round(dice(dints == PANCREAS, nnunet == PANCREAS), 3),
        "dice_tumor_dints_vs_nnunet": round(dice(d_tumor, n_tumor), 3),
        "tumor_agreement": agreement,
        "dice_complex_dints_vs_totalseg": round(dice(dints > 0, ts_pancreas), 3),
        "dice_complex_nnunet_vs_totalseg": round(dice(nnunet > 0, ts_pancreas), 3),
    }

    row["dice_complex_dints_vs_nnunet"] = 0.0 if np.isnan(row["dice_complex_dints_vs_nnunet"]) else row["dice_complex_dints_vs_nnunet"]

    row["confidence"] = confidence(row)

    return row, nnunet


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    patients = sys.argv[1:] or PATIENTS

    print("=" * 80)
    print("EXPERIMENT 16")
    print("SECOND MODEL: nnU-Net vs DiNTS")
    print("=" * 80)

    rows = []

    for number, patient in enumerate(patients, start=1):

        try:

            row = compare(patient)

        except Exception as error:

            row = {"patient": patient, "confidence": "ERROR", "error": str(error)}

        rows.append(row)

        print(
            f"[{number:02d}/{len(patients)}] {patient}: "
            f"complex Dice {row.get('dice_complex_dints_vs_nnunet')} | "
            f"tumor DiNTS {row.get('dints_tumor_candidate_ml')} mL vs nnU-Net {row.get('nnunet_tumor_candidate_ml')} mL "
            f"({row.get('tumor_agreement')}) | "
            f"nnU-Net vs TotalSeg {row.get('dice_complex_nnunet_vs_totalseg')} | {row.get('confidence')}",
            flush=True,
        )

    table = pd.DataFrame(rows)

    table.to_csv(OUTPUT_ROOT / "Phase2_Experiment16_SecondModel.csv", index=False, encoding="utf-8-sig")

    valid = table[table.confidence != "ERROR"]

    lines = [
        "EXPERIMENT 16 - nnU-Net (Task007, 3d_fullres) vs DiNTS, both after the Exp 13-B clean-up",
        "=" * 80,
        "Confidence level:",
        valid.confidence.value_counts().to_string(),
        "",
        "Tumor agreement:",
        valid.tumor_agreement.value_counts().to_string(),
        "",
        "Median Dice (all patients):",
        valid[["dice_complex_dints_vs_nnunet", "dice_pancreas_dints_vs_nnunet", "dice_complex_dints_vs_totalseg", "dice_complex_nnunet_vs_totalseg"]].median().round(3).to_string(),
        "",
        "Median tumor Dice where both models found a tumor:",
        str(valid.loc[valid.tumor_agreement.isin(["SAME_LESION", "DIFFERENT_LESIONS"]), "dice_tumor_dints_vs_nnunet"].median()),
        "",
        "DiNTS failures (PAN_02, 08, 13, 14) - does nnU-Net find a pancreas?",
        valid.loc[valid.patient.isin(["PAN_02", "PAN_08", "PAN_13", "PAN_14"]), ["patient", "nnunet_pancreas_ml", "nnunet_tumor_candidate_ml", "dice_complex_nnunet_vs_totalseg", "confidence"]].to_string(index=False),
        "",
        "nnU-Net tumor removed by the clean-up (outside the pancreas region, >= 1 mL):",
        valid.loc[valid.nnunet_removed_tumor_ml >= 1, ["patient", "nnunet_raw_tumor_candidate_ml", "nnunet_removed_tumor_ml", "nnunet_removed_tumor_organ"]].to_string(index=False),
        "",
        "Low-confidence patients:",
        valid.loc[valid.confidence.str.startswith("LOW"), ["patient", "dice_complex_dints_vs_nnunet", "dints_tumor_candidate_ml", "nnunet_tumor_candidate_ml", "tumor_agreement", "confidence"]].to_string(index=False),
        "",
        "IMPORTANT: agreement between two models trained on the same data is not accuracy;",
        "both can be wrong in the same way (e.g. both trained on portal-venous CT only).",
    ]

    summary = "\n".join(lines)

    (OUTPUT_ROOT / "Phase2_Experiment16_Summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)

    if len(patients) == len(PATIENTS) and CASE_TABLE.exists():

        merged = ["nnunet_pancreas_ml", "nnunet_tumor_candidate_ml", "dice_complex_dints_vs_nnunet", "dice_tumor_dints_vs_nnunet", "tumor_agreement", "confidence"]

        case_table = pd.read_csv(CASE_TABLE).drop(columns=merged, errors="ignore")

        case_table = case_table.merge(table[["patient"] + merged], on="patient", how="left")

        case_table.to_csv(CASE_TABLE, index=False, encoding="utf-8-sig")

        print("\nCase table updated:")
        print(CASE_TABLE)


if __name__ == "__main__":

    main()
