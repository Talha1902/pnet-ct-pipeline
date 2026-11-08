"""Experiment 17A - review package for the radiologist.

For every patient the package holds the CT and an editable start mask
(1 = pancreas, 2 = tumor) that the radiologist confirms or corrects in 3D Slicer:
    - normally the nnU-Net mask (Exp 16) after the same clean-up rules as Exp 13-B;
    - for the organ failures (confidence LOW_ORGAN_DISAGREE) the TotalSegmentator
      pancreas only, with no tumor, because both models fail there.
The DiNTS mask (Exp 13-B), the raw nnU-Net mask and the TotalSegmentator pancreas are
added as hidden reference layers.

Package layout: Cases/PAN_xx/ (CT, start mask, references), Returned/ (saved
corrections), Review_Sheet.xlsx, Slicer_Review.py, README_Radiologist.txt.

BLIND_CASES (optional, empty for now) would first be drawn from scratch without any
model output, to measure how much the start mask biases the radiologist (17B).

The review sheet is sorted by priority (LOW confidence first) and contains no grade,
Ki-67 or other clinical data, so the radiologist stays blinded to the endpoint.

Runs on CPU (login node), a few minutes in total (mostly copying the CTs).
"""
from pathlib import Path
import shutil
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "CodeFiles" / "Phase2" / "Experiment_13_MaskCleanup"))

from Task_13_Mask_Cleanup import TOTALSEG_PANCREAS, fill_holes, keep_anchored, remove_small_tumor  # noqa: E402


CT_ROOT = ROOT / "Results" / "_Phase2_Experiment10D_Baseline_NIfTI"

DINTS_ROOT = ROOT / "Results" / "_Phase2_Experiment13_MaskCleanup"

NNUNET_ROOT = ROOT / "Results" / "_Phase2_Experiment16_SecondModel" / "nnunet_predictions"

TOTALSEG_ROOT = ROOT / "Results" / "_Phase2_Experiment12B_ContrastPhase"

EXP16_TABLE = ROOT / "Results" / "_Phase2_Experiment16_SecondModel" / "Phase2_Experiment16_SecondModel.csv"

OUTPUT_ROOT = ROOT / "Results" / "_Phase2_Experiment17_RadiologistPackage"

PACKAGE = OUTPUT_ROOT / "Package"

HERE = Path(__file__).resolve().parent

PATIENTS = [f"PAN_{i:02d}" for i in range(1, 39)]

PANCREAS, TUMOR = 1, 2

# cases to draw from scratch before the review; dropped for now to keep the review simple
# (the earlier choice was PAN_09, 23, 26, 35, 38)
BLIND_CASES = []

# review order and what the radiologist should look at, per confidence level (Exp 16)
PRIORITY = {
    "LOW_ORGAN_DISAGREE": (1, "Hard case: the program could not draw this pancreas reliably. Only a rough pancreas is drawn, no tumor. Please correct the pancreas and draw the tumor."),
    "LOW_TUMOR_DISAGREE": (2, "Unsure tumor: the program is not sure whether there is a tumor. Please look carefully for a lesion."),
    "MEDIUM": (3, "Tumor found, size unsure: please check the tumor border."),
    "HIGH": (4, "Program is confident: please confirm or correct."),
}

PANCREAS_CHOICES = ["accepted", "minor edit", "major edit", "redrawn", "not visible"]

TUMOR_CHOICES = ["accepted", "minor edit", "major edit", "redrawn", "added (model missed it)", "removed (not a tumor)", "no tumor present"]


def load(path):

    img = nib.load(str(path))

    return np.rint(np.asanyarray(img.dataobj)).astype(np.uint8), img


def save(label, img, path):

    out = nib.Nifti1Image(label.astype(np.uint8), img.affine, img.header)

    out.set_data_dtype(np.uint8)

    nib.save(out, str(path))


def start_mask(patient, confidence):
    """nnU-Net after the 13-B rules, or the TotalSegmentator pancreas for organ failures."""

    nnunet, img = load(NNUNET_ROOT / f"{patient}.nii.gz")

    spacing = np.array(img.header.get_zooms()[:3], dtype=float)

    voxel_ml = float(np.prod(spacing) / 1000)

    ts_pancreas = np.asanyarray(nib.load(str(TOTALSEG_ROOT / patient / "totalseg_roi.nii.gz")).dataobj) == TOTALSEG_PANCREAS

    if confidence == "LOW_ORGAN_DISAGREE":

        return np.where(ts_pancreas, PANCREAS, 0).astype(np.uint8), nnunet, ts_pancreas, img, "TotalSegmentator pancreas (no tumor)", voxel_ml

    label, _ = keep_anchored(nnunet, spacing, ts_pancreas)

    label = fill_holes(remove_small_tumor(label, voxel_ml)[0])[0]

    return label, nnunet, ts_pancreas, img, "nnU-Net (13-B clean-up)", voxel_ml


def build_patient(patient, confidence):

    folder = PACKAGE / "Cases" / patient

    folder.mkdir(parents=True, exist_ok=True)

    shutil.copy2(CT_ROOT / patient / "baseline_ct.nii.gz", folder / "ct.nii.gz")

    label, nnunet, ts_pancreas, img, source, voxel_ml = start_mask(patient, confidence)

    dints, dints_img = load(DINTS_ROOT / patient / "dints_clean.nii.gz")

    if dints.shape != label.shape or not np.allclose(dints_img.affine, img.affine, atol=1e-3):

        raise ValueError(f"DiNTS grid {dints.shape} does not match nnU-Net grid {label.shape}")

    save(label, img, folder / "start_mask.nii.gz")

    save(dints, img, folder / "ref_dints.nii.gz")

    save(nnunet, img, folder / "ref_nnunet.nii.gz")

    save(ts_pancreas.astype(np.uint8), img, folder / "ref_totalseg_pancreas.nii.gz")

    if patient in BLIND_CASES:

        blind = PACKAGE / "Blind" / patient

        blind.mkdir(parents=True, exist_ok=True)

        shutil.copy2(folder / "ct.nii.gz", blind / "ct.nii.gz")

    return {
        "patient": patient,
        "confidence": confidence,
        "start_mask_source": source,
        "start_pancreas_ml": round(float((label == PANCREAS).sum() * voxel_ml), 3),
        "start_tumor_ml": round(float((label == TUMOR).sum() * voxel_ml), 3),
        "blind_case": patient in BLIND_CASES,
    }


def write_review_sheet(table, package=PACKAGE, blind_cases=BLIND_CASES):
    """Excel sheet for the radiologist: blind cases (if any) first, then all cases by priority."""

    workbook = Workbook()

    sheet = workbook.active

    sheet.title = "Review"

    header = ["Order", "Patient", "What to check", "Pancreas", "Tumor", "Number of lesions", "Your confidence (1-5)", "Time (min)", "Notes"]

    sheet.append(header)

    order = 1

    for patient in blind_cases:

        sheet.append([order, f"{patient} (blind)", "Draw pancreas and tumor from scratch (no model output). Do these first.", "", "", "", "", "", ""])

        order += 1

    for row in table.itertuples():

        sheet.append([order, row.patient, PRIORITY[row.confidence][1], "", "", "", "", "", ""])

        order += 1

    last = sheet.max_row

    for column, choices in (("D", PANCREAS_CHOICES), ("E", TUMOR_CHOICES), ("F", ["0", "1", "2", "3 or more"]), ("G", ["1", "2", "3", "4", "5"])):

        validation = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True)

        sheet.add_data_validation(validation)

        validation.add(f"{column}2:{column}{last}")

    for cell in sheet[1]:

        cell.font = Font(bold=True)

        cell.fill = PatternFill("solid", fgColor="DDDDDD")

    for column, width in zip("ABCDEFGHI", (7, 10, 70, 14, 24, 12, 12, 10, 40)):

        sheet.column_dimensions[column].width = width

    for cells in sheet.iter_rows(min_row=2):

        cells[2].alignment = Alignment(wrap_text=True, vertical="top")

    sheet.freeze_panes = "A2"

    workbook.save(package / "Review_Sheet.xlsx")

    # Slicer_Review.py lists the patients in this order
    (package / "review_order.txt").write_text("\n".join(table.patient) + "\n", encoding="utf-8")


def main():

    patients = sys.argv[1:] or PATIENTS

    PACKAGE.mkdir(parents=True, exist_ok=True)

    (PACKAGE / "Returned").mkdir(exist_ok=True)

    confidence = pd.read_csv(EXP16_TABLE).set_index("patient")["confidence"]

    print("=" * 80)
    print("EXPERIMENT 17A")
    print("RADIOLOGIST REVIEW PACKAGE")
    print("=" * 80)

    rows = []

    for number, patient in enumerate(patients, start=1):

        row = build_patient(patient, confidence[patient])

        rows.append(row)

        print(
            f"[{number:02d}/{len(patients)}] {patient}: {row['confidence']} | start = {row['start_mask_source']} | "
            f"pancreas {row['start_pancreas_ml']} mL, tumor {row['start_tumor_ml']} mL"
            + (" | BLIND" if row["blind_case"] else ""),
            flush=True,
        )

    table = pd.DataFrame(rows)

    table["priority"] = table.confidence.map(lambda level: PRIORITY[level][0])

    table = table.sort_values(["priority", "patient"]).reset_index(drop=True)

    table.to_csv(OUTPUT_ROOT / "Phase2_Experiment17A_Package.csv", index=False, encoding="utf-8-sig")

    write_review_sheet(table)

    for name in ("Slicer_Review.py", "README_Radiologist.txt"):

        shutil.copy2(HERE / name, PACKAGE / name)

    lines = [
        "EXPERIMENT 17A - RADIOLOGIST REVIEW PACKAGE",
        "=" * 80,
        f"Package: {PACKAGE}",
        "",
        "Start mask source:",
        table.start_mask_source.value_counts().to_string(),
        "",
        "Review order (priority 1 = first):",
        *[f"  {priority} {level:20s} {', '.join(group.patient)}" for (priority, level), group in table.groupby(["priority", "confidence"])],
        "",
        f"Blind cases (drawn from scratch first): {', '.join(BLIND_CASES) or 'none'}",
        "",
        "The review sheet contains no grade, Ki-67 or clinical data.",
    ]

    summary = "\n".join(lines)

    (OUTPUT_ROOT / "Phase2_Experiment17A_Summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)


if __name__ == "__main__":

    main()
