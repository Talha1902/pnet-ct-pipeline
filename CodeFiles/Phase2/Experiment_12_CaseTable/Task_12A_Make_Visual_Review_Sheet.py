"""Experiment 12A (part 2) - Excel sheet for the manual 3D Slicer review.

The reviewer answers two questions per patient with dropdowns:
    pancreas_check : Good / Partial or fragmented / Failed
    tumor_check    : Plausible lesion / Partial lesion / Doubtful / None
and the A-E grade is derived by formula, so every case is graded with the same rule:
    Failed                      -> D  clear segmentation failure
    Partial or fragmented       -> C  partial or fragmented segmentation
    Good + None                 -> E  no tumor candidate generated
    Good + Doubtful             -> B  plausible pancreas, uncertain tumor candidate
    Good + Partial lesion       -> B  tumor found, but yellow covers only part of it
    Good + Plausible lesion     -> A  plausible pancreas and plausible tumor candidate
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parents[3]

OUTPUT_ROOT = ROOT / "Results" / "_Phase2_Experiment12_CaseTable"

CASE_TABLE = OUTPUT_ROOT / "Phase2_Experiment12A_CaseTable_All38.csv"

QC_CSV = (
    ROOT
    / "Results"
    / "_Phase2_Experiment11C_CohortAudit"
    / "Phase2_Experiment11C_DiNTS_CohortAudit_All38.csv"
)

REVIEW_XLSX = OUTPUT_ROOT / "Phase2_Experiment12A_VisualReview.xlsx"

PRIORITY_ORDER = {"HIGH_PRIORITY": 0, "MEDIUM_PRIORITY": 1, "ROUTINE_REVIEW": 2}

INFO_COLUMNS = [
    ("review_order", 8),
    ("patient", 9),
    ("auto_qc_priority", 17),
    ("auto_qc_reason", 40),
    ("suggested_phase", 22),
    ("slice_thickness_mm", 9),
    ("pancreas_volume_ml", 10),
    ("tumor_candidate_volume_ml", 10),
    ("tumor_slices_K", 12),
    ("ct_file", 50),
    ("mask_file", 62),
]

REVIEW_COLUMNS = [
    ("pancreas_check", 22),
    ("tumor_check", 18),
    ("visual_grade_A_to_E", 10),
    ("confidence", 11),
    ("visual_notes", 50),
]

INSTRUCTIONS = """HOW TO GRADE (one patient at a time, in review_order)

1. In 3D Slicer drag in ct_file (load as Volume) and mask_file (load as Segmentation).
   Segment_1 = pancreas, Segment_2 = tumor candidate. Use the CT-Abdomen window preset.
2. Scroll through all axial slices that contain colour, then check coronal and sagittal.
   tumor_slices_K = slice range (K index in Data Probe) where the tumor candidate is.

QUESTION 1 - pancreas_check (the pancreas colour)
   Good                  : covers most of head, body and tail; sits where the pancreas is
                           (behind the stomach, in front of the spine/aorta, head in the
                           duodenal C-loop, tail towards the spleen); little spill onto
                           stomach, bowel, liver, spleen or vessels.
   Partial or fragmented : right organ, but a large part is missing (e.g. no tail), it is
                           broken into pieces, or it clearly spills onto neighbouring organs.
   Failed                : empty, tiny, or mostly on the wrong organ / wrong place.

QUESTION 2 - tumor_check (the tumor-candidate colour)
   Plausible lesion : a focal blob in or touching the pancreas that looks different from
                      the normal pancreas (often brighter in arterial phase, or darker),
                      and the yellow covers most of that lesion.
   Partial lesion   : the yellow is on a real-looking lesion but covers only part of it
                      (e.g. most of a large mass is painted green as "pancreas").
   Doubtful        : present but on a vessel, bowel or outside the pancreas, a few voxels
                      only, or it just paints normal-looking pancreas.
   None             : no tumor-candidate colour anywhere.

The grade (A-E) is filled in automatically from the two answers:
   Failed -> D | Partial or fragmented -> C | Good+None -> E | Good+Doubtful -> B |
   Good+Partial lesion -> B | Good+Plausible lesion -> A

confidence : Sure / Unsure. Mark Unsure whenever you hesitate; those go to the radiologist first.
visual_notes : anything unusual (artefacts, stent, tumor visible but not segmented, wrong
               phase, scan cut off, etc.). If you can SEE a tumor the model missed, write where.

This is a research QC grade by a non-radiologist. It is not clinical validation.
"""


def main():

    table = pd.read_csv(CASE_TABLE)

    qc = pd.read_csv(QC_CSV)[[
        "patient",
        "tumor_candidate_first_slice",
        "tumor_candidate_last_slice",
    ]]

    table = table.merge(qc, on="patient", how="left")

    table = table.sort_values(
        by=["auto_qc_priority", "patient"],
        key=lambda s: s.map(PRIORITY_ORDER) if s.name == "auto_qc_priority" else s,
    ).reset_index(drop=True)

    table["review_order"] = range(1, len(table) + 1)

    table["tumor_slices_K"] = [
        f"{int(a)}-{int(b)}" if pd.notna(a) else ""
        for a, b in zip(table.tumor_candidate_first_slice, table.tumor_candidate_last_slice)
    ]

    table["ct_file"] = [
        str(ROOT / "Results" / "_Phase2_Experiment10D_Baseline_NIfTI" / p / "baseline_ct.nii.gz")
        for p in table.patient
    ]

    table["mask_file"] = [
        str(ROOT / "Results" / "_Phase2_Experiment11A_MONAI_DiNTS" / p / "baseline_ct" / "baseline_ct_trans.nii.gz")
        for p in table.patient
    ]

    wb = Workbook()

    ws = wb.active

    ws.title = "Review"

    columns = INFO_COLUMNS + REVIEW_COLUMNS

    header_fill = PatternFill("solid", fgColor="DDDDDD")

    review_fill = PatternFill("solid", fgColor="FFF2CC")

    for col, (name, width) in enumerate(columns, start=1):

        cell = ws.cell(row=1, column=col, value=name)

        cell.font = Font(bold=True)

        cell.fill = review_fill if (name, width) in REVIEW_COLUMNS else header_fill

        cell.alignment = Alignment(wrap_text=True, vertical="top")

        ws.column_dimensions[cell.column_letter].width = width

    letter = {name: ws.cell(row=1, column=i).column_letter for i, (name, _) in enumerate(columns, 1)}

    for r, row in enumerate(table.itertuples(index=False), start=2):

        for col, (name, _) in enumerate(INFO_COLUMNS, start=1):

            value = getattr(row, name)

            ws.cell(row=r, column=col, value=None if pd.isna(value) else value)

        p, t = f"{letter['pancreas_check']}{r}", f"{letter['tumor_check']}{r}"

        ws[f"{letter['visual_grade_A_to_E']}{r}"] = (
            f'=IF({p}="","",IF({p}="Failed","D",IF({p}="Partial or fragmented","C",'
            f'IF({t}="","",IF({t}="None","E",IF(OR({t}="Doubtful",{t}="Partial lesion"),"B","A"))))))'
        )

        for name, _ in REVIEW_COLUMNS:

            ws[f"{letter[name]}{r}"].fill = review_fill

    last = len(table) + 1

    for name, options in [
        ("pancreas_check", "Good,Partial or fragmented,Failed"),
        ("tumor_check", "Plausible lesion,Partial lesion,Doubtful,None"),
        ("confidence", "Sure,Unsure"),
    ]:

        dv = DataValidation(type="list", formula1=f'"{options}"', allow_blank=True)

        ws.add_data_validation(dv)

        dv.add(f"{letter[name]}2:{letter[name]}{last}")

    ws.freeze_panes = "C2"

    guide = wb.create_sheet("How to grade")

    for i, line in enumerate(INSTRUCTIONS.splitlines(), start=1):

        guide.cell(row=i, column=1, value=line).font = Font(name="Consolas")

    guide.column_dimensions["A"].width = 100

    wb.save(REVIEW_XLSX)

    print("Review sheet:")
    print(REVIEW_XLSX)


if __name__ == "__main__":

    main()
