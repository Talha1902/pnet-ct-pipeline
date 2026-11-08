"""Experiment 17B - compare the automatic masks with the radiologist's masks.

Reads the masks returned by the radiologist (Package/Returned, written by
Slicer_Review.py; 1 = pancreas, 2 = tumor) and compares them with:
    start    - the start mask the radiologist corrected (17A)
    dints    - DiNTS after the Exp 13-B clean-up
    nnunet   - raw nnU-Net (Exp 16)
    totalseg - TotalSegmentator pancreas (organ only)

Per patient and model:
    - Dice of pancreas, tumor and the pancreas+tumor complex
    - HD95 (95th percentile of the boundary distance, mm) of pancreas and tumor
    - tumor detection: the model's tumor lies within TUMOR_MATCH_MM of the radiologist's
    - tumor volume error (%)
plus how much the radiologist changed the start mask (accepted unchanged or not).

For the BLIND_CASES the blind round (drawn from scratch) is compared with the review
round: if the review round is closer to the start mask than the blind round is, the
start mask biased the annotation.

This is the first measurement of accuracy in the project: the radiologist is the reference.

    python Task_17B_Compare_With_Radiologist.py             -> real returned masks
    python Task_17B_Compare_With_Radiologist.py --simulate  -> pipeline test with fake
                                                               "radiologist" masks
    python Task_17B_Compare_With_Radiologist.py --simulate PAN_02 PAN_09  -> only these patients
"""
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "CodeFiles" / "Phase2" / "Experiment_13_MaskCleanup"))

from Task_13_Mask_Cleanup import STRUCTURE, bounding_box, dice, distance_to  # noqa: E402

from Task_17A_Build_Package import BLIND_CASES, OUTPUT_ROOT, PACKAGE  # noqa: E402


REVIEW_ROOT = PACKAGE / "Cases"

PATIENTS = [f"PAN_{i:02d}" for i in range(1, 39)]

PANCREAS, TUMOR = 1, 2

# same tolerance as Exp 16 for "the same lesion"
TUMOR_MATCH_MM = 10.0

# below this the radiologist's tumor counts as absent (same as Exp 13 and 16)
MIN_TUMOR_ML = 0.05

MODELS = {
    "start": "start_mask.nii.gz",
    "dints": "ref_dints.nii.gz",
    "nnunet": "ref_nnunet.nii.gz",
    "totalseg": "ref_totalseg_pancreas.nii.gz",
}


def load(path):

    img = nib.load(str(path))

    return np.rint(np.asanyarray(img.dataobj)).astype(np.uint8), img


def check_grid(label, img, ref_img, what):

    if label.shape != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine, atol=1e-2):

        raise ValueError(f"{what} grid {label.shape} does not match the CT grid {ref_img.shape[:3]}")


def hd95(a, b, spacing):
    """95th percentile of the symmetric surface distance in mm (NaN if either mask is empty)."""

    if not a.any() or not b.any():

        return np.nan

    box = bounding_box(a | b, 2)

    a, b = a[box], b[box]

    surface_a = a & ~ndimage.binary_erosion(a, STRUCTURE)

    surface_b = b & ~ndimage.binary_erosion(b, STRUCTURE)

    distances = np.concatenate([
        ndimage.distance_transform_edt(~surface_b, sampling=spacing)[surface_a],
        ndimage.distance_transform_edt(~surface_a, sampling=spacing)[surface_b],
    ])

    return float(np.percentile(distances, 95))


def tumor_found(model_tumor, reference_tumor, spacing):
    """True if any voxel of the model's tumor lies within TUMOR_MATCH_MM of the reference tumor."""

    margin = np.ceil(TUMOR_MATCH_MM / spacing).astype(int) + 1

    box = bounding_box(model_tumor | reference_tumor, margin)

    return bool((distance_to(reference_tumor, spacing, box)[model_tumor[box]] <= TUMOR_MATCH_MM).any())


def detection(model_tumor, reference_tumor, spacing, voxel_ml):

    has_model = model_tumor.sum() * voxel_ml >= MIN_TUMOR_ML

    has_reference = reference_tumor.sum() * voxel_ml >= MIN_TUMOR_ML

    if has_reference and has_model:

        return "TP" if tumor_found(model_tumor, reference_tumor, spacing) else "FN_WRONG_PLACE"

    if has_reference:

        return "FN"

    return "FP" if has_model else "TN"


def compare_masks(model, reference, spacing, voxel_ml, prefix):

    row = {
        f"{prefix}_dice_pancreas": round(dice(model == PANCREAS, reference == PANCREAS), 3),
        f"{prefix}_dice_complex": round(dice(model > 0, reference > 0), 3),
        f"{prefix}_hd95_complex_mm": round(hd95(model > 0, reference > 0, spacing), 1),
    }

    if prefix == "totalseg":

        return row

    model_tumor, reference_tumor = model == TUMOR, reference == TUMOR

    reference_ml = reference_tumor.sum() * voxel_ml

    row.update({
        f"{prefix}_dice_tumor": round(dice(model_tumor, reference_tumor), 3),
        f"{prefix}_hd95_tumor_mm": round(hd95(model_tumor, reference_tumor, spacing), 1),
        f"{prefix}_tumor_detection": detection(model_tumor, reference_tumor, spacing, voxel_ml),
        f"{prefix}_tumor_volume_error_pct": round(float((model_tumor.sum() * voxel_ml - reference_ml) / reference_ml * 100), 1) if reference_ml else np.nan,
    })

    return row


def compare_patient(patient, returned):

    final, final_img = load(returned / f"{patient}_final.nii.gz")

    ct_img = nib.load(str(REVIEW_ROOT / patient / "ct.nii.gz"))

    check_grid(final, final_img, ct_img, "Returned mask")

    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=float)

    voxel_ml = float(np.prod(spacing) / 1000)

    row = {
        "patient": patient,
        "radiologist_pancreas_ml": round(float((final == PANCREAS).sum() * voxel_ml), 3),
        "radiologist_tumor_ml": round(float((final == TUMOR).sum() * voxel_ml), 3),
    }

    masks = {}

    for prefix, filename in MODELS.items():

        masks[prefix] = load(REVIEW_ROOT / patient / filename)[0]

        row.update(compare_masks(masks[prefix], final, spacing, voxel_ml, prefix))

    changed = masks["start"] != final

    row["changed_ml"] = round(float(changed.sum() * voxel_ml), 3)

    row["accepted_unchanged"] = not changed.any()

    blind_path = returned / f"{patient}_blind.nii.gz"

    if patient in BLIND_CASES and blind_path.exists():

        blind, blind_img = load(blind_path)

        check_grid(blind, blind_img, ct_img, "Blind mask")

        row["blind_vs_final_dice_tumor"] = round(dice(blind == TUMOR, final == TUMOR), 3)

        row["blind_vs_start_dice_tumor"] = round(dice(blind == TUMOR, masks["start"] == TUMOR), 3)

        row["final_vs_start_dice_tumor"] = round(dice(final == TUMOR, masks["start"] == TUMOR), 3)

        row["blind_vs_final_dice_complex"] = round(dice(blind > 0, final > 0), 3)

    return row


def simulate_returned(folder, patients):
    """Fake 'radiologist' masks to test the pipeline: TotalSegmentator pancreas + start-mask
    tumor grown by one voxel; blind round = DiNTS. NOT a result."""

    folder.mkdir(parents=True, exist_ok=True)

    for patient in patients:

        start, img = load(REVIEW_ROOT / patient / "start_mask.nii.gz")

        totalseg = load(REVIEW_ROOT / patient / "ref_totalseg_pancreas.nii.gz")[0] > 0

        fake = np.where(totalseg, PANCREAS, 0).astype(np.uint8)

        fake[ndimage.binary_dilation(start == TUMOR, STRUCTURE)] = TUMOR

        nib.save(nib.Nifti1Image(fake, img.affine), str(folder / f"{patient}_final.nii.gz"))

        if patient in BLIND_CASES:

            dints = load(REVIEW_ROOT / patient / "ref_dints.nii.gz")[0]

            nib.save(nib.Nifti1Image(dints, img.affine), str(folder / f"{patient}_blind.nii.gz"))


def summarise(table, prefix):

    lines = ["EXPERIMENT 17B - AUTOMATIC MASKS vs RADIOLOGIST", "=" * 80]

    if prefix:

        lines += ["*** SIMULATED RADIOLOGIST - PIPELINE TEST ONLY, NOT A RESULT ***", ""]

    valid = table[table.get("error", pd.Series(index=table.index, dtype=object)).isna()]

    lines += [f"Patients compared: {len(valid)} (errors: {len(table) - len(valid)})", "", "Median (IQR) per model:"]

    for model in MODELS:

        for metric in ("dice_pancreas", "dice_complex", "hd95_complex_mm", "dice_tumor", "hd95_tumor_mm"):

            column = f"{model}_{metric}"

            if column in valid:

                values = valid[column].dropna()

                if len(values):

                    lines.append(f"  {column:30s} {values.median():7.3f}  ({values.quantile(0.25):.3f} - {values.quantile(0.75):.3f})  n={len(values)}")

    lines += ["", "Tumor detection (TP found, FN missed, FN_WRONG_PLACE, FP extra, TN none):"]

    for model in ("start", "dints", "nnunet"):

        lines.append(f"  {model:8s} " + valid[f"{model}_tumor_detection"].value_counts().to_string().replace("\n", " | "))

    lines += [
        "",
        f"Start mask accepted unchanged: {int(valid.accepted_unchanged.sum())}/{len(valid)}",
        f"Median change to the start mask: {valid.changed_ml.median():.1f} mL",
    ]

    if "confidence" in valid:

        lines += ["", "Start mask by confidence level (median):", valid.groupby("confidence")[["start_dice_complex", "start_dice_tumor", "changed_ml"]].median().round(3).to_string()]

    if "blind_vs_final_dice_tumor" in valid:

        blind = valid.dropna(subset=["blind_vs_final_dice_tumor"])

        lines += [
            "",
            "Blind round vs review round (anchoring bias):",
            blind[["patient", "blind_vs_final_dice_tumor", "blind_vs_start_dice_tumor", "final_vs_start_dice_tumor", "blind_vs_final_dice_complex"]].to_string(index=False),
            "If final_vs_start is clearly higher than blind_vs_start, the start mask pulled the annotation towards the model.",
        ]

    return "\n".join(lines)


def main():

    simulate = "--simulate" in sys.argv

    prefix = "SIMULATED_" if simulate else ""

    returned = OUTPUT_ROOT / "Simulated_Returned" if simulate else PACKAGE / "Returned"

    chosen = [a for a in sys.argv[1:] if not a.startswith("--")]

    if simulate:

        simulate_returned(returned, chosen or PATIENTS)

    patients = chosen or [p for p in PATIENTS if (returned / f"{p}_final.nii.gz").exists()]

    print("=" * 80)
    print("EXPERIMENT 17B" + (" (SIMULATED)" if simulate else ""))
    print("AUTOMATIC MASKS vs RADIOLOGIST")
    print("=" * 80)

    rows = []

    for number, patient in enumerate(patients, start=1):

        try:

            row = compare_patient(patient, returned)

        except Exception as error:

            row = {"patient": patient, "error": str(error)}

        rows.append(row)

        print(
            f"[{number:02d}/{len(patients)}] {patient}: start Dice complex {row.get('start_dice_complex')} tumor {row.get('start_dice_tumor')} | "
            f"DiNTS tumor {row.get('dints_dice_tumor')} | nnU-Net tumor {row.get('nnunet_dice_tumor')} | changed {row.get('changed_ml')} mL"
            + (f" | ERROR {row['error']}" if "error" in row else ""),
            flush=True,
        )

    table = pd.DataFrame(rows)

    package = pd.read_csv(OUTPUT_ROOT / "Phase2_Experiment17A_Package.csv")[["patient", "confidence", "start_mask_source"]]

    table = package.merge(table, on="patient", how="right")

    table.to_csv(OUTPUT_ROOT / f"{prefix}Phase2_Experiment17B_Radiologist.csv", index=False, encoding="utf-8-sig")

    summary = summarise(table, prefix)

    (OUTPUT_ROOT / f"{prefix}Phase2_Experiment17B_Summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)


if __name__ == "__main__":

    main()
