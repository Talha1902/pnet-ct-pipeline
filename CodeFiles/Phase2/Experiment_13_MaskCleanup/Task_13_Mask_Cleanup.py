"""Experiment 13 - rule-based clean-up of the DiNTS masks (no retraining).

DiNTS output (11A) often contains fragments away from the pancreas and tiny
"tumor candidate" specks. The model is left unchanged; only transparent
post-processing rules are applied to its output. Two variants are compared:

    13-A "largest": keep the largest connected component of (pancreas + tumor)
         plus any component within GAP_MM of it.
    13-B "anchored" (final): keep every component that lies within ANCHOR_MM
         of the TotalSegmentator pancreas (12B), i.e. inside the anatomical
         pancreas region. If no component does, fall back to 13-A and flag it,
         so the second model never silently empties a DiNTS mask.

13-A was tried first and fails when DiNTS splits the pancreas into pieces more
than GAP_MM apart (PAN_04 lost a 12.9 mL piece containing the tumor candidate).

Both variants then:
    - remove tumor candidate components smaller than MIN_TUMOR_ML
      (relabelled pancreas if they touch the pancreas, otherwise background);
    - fill holes inside the complex (labelled pancreas).

Agreement with the TotalSegmentator pancreas (Dice, tumor within TUMOR_NEAR_MM)
is reported before and after. This is agreement between two models, not accuracy.

Runs on CPU (local or login node), about 5-10 s per patient.
"""
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]

PRED_ROOT = ROOT / "Results" / "_Phase2_Experiment11A_MONAI_DiNTS"

TOTALSEG_ROOT = ROOT / "Results" / "_Phase2_Experiment12B_ContrastPhase"

OUTPUT_ROOT = ROOT / "Results" / "_Phase2_Experiment13_MaskCleanup"

CASE_TABLE = (
    ROOT
    / "Results"
    / "_Phase2_Experiment12_CaseTable"
    / "Phase2_Experiment12A_CaseTable_All38.csv"
)

PATIENTS = [f"PAN_{i:02d}" for i in range(1, 39)]

PANCREAS, TUMOR = 1, 2

# TotalSegmentator "total" task label for the pancreas
TOTALSEG_PANCREAS = 7

# fragments closer than this to the main complex are kept
GAP_MM = 5.0

# 13-B: components within this distance of the TotalSegmentator pancreas are kept
ANCHOR_MM = 10.0

# tumor candidate components smaller than this are removed (0.05 mL ~ 4.6 mm sphere)
MIN_TUMOR_ML = 0.05

# a tumor candidate within this distance of the TotalSegmentator pancreas counts as "near"
TUMOR_NEAR_MM = 10.0

# clean-up that removes more than this share of the foreground is flagged for review
MAJOR_CHANGE_PCT = 5.0

# 26-connectivity in 3D
STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def get_prediction(patient):

    return PRED_ROOT / patient / "baseline_ct" / "baseline_ct_trans.nii.gz"


def get_totalseg(patient):

    return TOTALSEG_ROOT / patient / "totalseg_roi.nii.gz"


def bounding_box(mask, margin):
    """Slices around the foreground, padded by margin voxels per axis."""

    coords = np.argwhere(mask)

    low = np.maximum(coords.min(axis=0) - margin, 0)

    high = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)

    return tuple(slice(a, b) for a, b in zip(low, high))


def distance_to(mask, spacing, box):
    """Distance in mm from every voxel of the box to the nearest voxel of mask."""

    return ndimage.distance_transform_edt(~mask[box], sampling=spacing)


def dice(a, b):

    total = a.sum() + b.sum()

    return float(2 * (a & b).sum() / total) if total else np.nan


def keep_main_complex(label, spacing):
    """Keep the largest pancreas+tumor component and fragments within GAP_MM of it."""

    complex_mask = label > 0

    components, number = ndimage.label(complex_mask, structure=STRUCTURE)

    if number <= 1:

        return label.copy(), number, number

    sizes = np.bincount(components.ravel())[1:]

    main = components == (np.argmax(sizes) + 1)

    margin = np.ceil(GAP_MM / spacing).astype(int) + 1

    box = bounding_box(complex_mask, margin)

    distance = distance_to(main, spacing, box)

    near = np.unique(components[box][(distance <= GAP_MM) & complex_mask[box]])

    keep = np.isin(components, near[near > 0])

    cleaned = np.where(keep, label, 0).astype(np.uint8)

    return cleaned, number, len(near[near > 0])


def keep_anchored(label, spacing, ts_pancreas):
    """Keep components within ANCHOR_MM of the TotalSegmentator pancreas.

    Returns the cleaned label and whether the anchor was used (False -> fell back to 13-A).
    """

    complex_mask = label > 0

    if not complex_mask.any() or not ts_pancreas.any():

        return keep_main_complex(label, spacing)[0], False

    components, _ = ndimage.label(complex_mask, structure=STRUCTURE)

    margin = np.ceil(ANCHOR_MM / spacing).astype(int) + 1

    box = bounding_box(ts_pancreas, margin)

    region = distance_to(ts_pancreas, spacing, box) <= ANCHOR_MM

    anchored = np.unique(components[box][region])

    anchored = anchored[anchored > 0]

    if len(anchored) == 0:

        return keep_main_complex(label, spacing)[0], False

    return np.where(np.isin(components, anchored), label, 0).astype(np.uint8), True


def remove_small_tumor(label, voxel_ml):
    """Drop tumor components below MIN_TUMOR_ML; specks touching pancreas become pancreas."""

    tumor = label == TUMOR

    components, number = ndimage.label(tumor, structure=STRUCTURE)

    if number == 0:

        return label, 0

    sizes = np.bincount(components.ravel())

    small_ids = np.flatnonzero(sizes * voxel_ml < MIN_TUMOR_ML)

    small_ids = small_ids[small_ids > 0]

    if len(small_ids) == 0:

        return label, 0

    cleaned = label.copy()

    pancreas_border = ndimage.binary_dilation(label == PANCREAS, structure=STRUCTURE)

    for component_id in small_ids:

        speck = components == component_id

        cleaned[speck] = PANCREAS if (speck & pancreas_border).any() else 0

    return cleaned, len(small_ids)


def fill_holes(label):

    complex_mask = label > 0

    if not complex_mask.any():

        return label, 0

    box = bounding_box(complex_mask, 1)

    holes = ndimage.binary_fill_holes(complex_mask[box]) & ~complex_mask[box]

    cleaned = label.copy()

    cleaned[box][holes] = PANCREAS

    return cleaned, int(holes.sum())


def component_count(mask):

    return ndimage.label(mask, structure=STRUCTURE)[1]


def tumor_near_totalseg(tumor, ts_pancreas, spacing):
    """Share of tumor candidate voxels within TUMOR_NEAR_MM of the TotalSegmentator pancreas."""

    if not tumor.any() or not ts_pancreas.any():

        return np.nan

    margin = np.ceil(TUMOR_NEAR_MM / spacing).astype(int) + 1

    box = bounding_box(tumor | ts_pancreas, margin)

    distance = distance_to(ts_pancreas, spacing, box)

    return float((distance[tumor[box]] <= TUMOR_NEAR_MM).mean() * 100)


def describe(label, ts_pancreas, spacing, voxel_ml, prefix):

    pancreas = label == PANCREAS

    tumor = label == TUMOR

    return {
        f"{prefix}_pancreas_ml": round(float(pancreas.sum() * voxel_ml), 3),
        f"{prefix}_tumor_candidate_ml": round(float(tumor.sum() * voxel_ml), 3),
        f"{prefix}_pancreas_components": component_count(pancreas),
        f"{prefix}_tumor_candidate_components": component_count(tumor),
        f"{prefix}_dice_vs_totalseg": round(dice(label > 0, ts_pancreas), 3),
        f"{prefix}_tumor_near_totalseg_pct": round(tumor_near_totalseg(tumor, ts_pancreas, spacing), 1),
    }


def cleanup_flag(row):

    if row["clean_pancreas_ml"] + row["clean_tumor_candidate_ml"] < 1:

        return "EMPTY_OR_NEAR_EMPTY"

    if not row["anchor_used"]:

        return "NOT_NEAR_TOTALSEG_PANCREAS"

    if row["raw_tumor_candidate_ml"] > 0 and row["clean_tumor_candidate_ml"] == 0:

        return "TUMOR_CANDIDATE_REMOVED"

    if row["removed_pct"] >= MAJOR_CHANGE_PCT:

        return "MAJOR_CHANGE"

    if row["removed_ml"] > 0 or row["holes_filled_ml"] > 0:

        return "MINOR_CHANGE"

    return "NO_CHANGE"


def clean_patient(patient):

    pred_img = nib.load(str(get_prediction(patient)))

    raw = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.uint8)

    spacing = np.array(pred_img.header.get_zooms()[:3], dtype=float)

    voxel_ml = float(np.prod(spacing) / 1000)

    ts_pancreas = np.asanyarray(nib.load(str(get_totalseg(patient))).dataobj) == TOTALSEG_PANCREAS

    row = {"patient": patient}

    row.update(describe(raw, ts_pancreas, spacing, voxel_ml, "raw"))

    # 13-A: largest component (kept for comparison only)
    largest, n_complex, _ = keep_main_complex(raw, spacing)

    largest = fill_holes(remove_small_tumor(largest, voxel_ml)[0])[0]

    row["largest_removed_pct"] = round(float(((raw > 0) & (largest == 0)).sum() / max((raw > 0).sum(), 1) * 100), 2)

    row["largest_tumor_candidate_ml"] = round(float((largest == TUMOR).sum() * voxel_ml), 3)

    row["largest_dice_vs_totalseg"] = round(dice(largest > 0, ts_pancreas), 3)

    # 13-B: anchored on the TotalSegmentator pancreas (final)
    label, anchor_used = keep_anchored(raw, spacing, ts_pancreas)

    label, n_specks = remove_small_tumor(label, voxel_ml)

    label, n_holes = fill_holes(label)

    row["complex_components_raw"] = n_complex

    row["complex_components_kept"] = component_count(label > 0)

    row["anchor_used"] = anchor_used

    row["tumor_specks_removed"] = n_specks

    removed = (raw > 0) & (label == 0)

    row["removed_ml"] = round(float(removed.sum() * voxel_ml), 3)

    row["removed_pct"] = round(float(removed.sum() / max((raw > 0).sum(), 1) * 100), 2)

    row["holes_filled_ml"] = round(n_holes * voxel_ml, 3)

    row.update(describe(label, ts_pancreas, spacing, voxel_ml, "clean"))

    row["totalseg_pancreas_ml"] = round(float(ts_pancreas.sum() * voxel_ml), 3)

    row["cleanup_flag"] = cleanup_flag(row)

    out_dir = OUTPUT_ROOT / patient

    out_dir.mkdir(parents=True, exist_ok=True)

    out_img = nib.Nifti1Image(label, pred_img.affine, pred_img.header)

    out_img.set_data_dtype(np.uint8)

    nib.save(out_img, str(out_dir / "dints_clean.nii.gz"))

    return row


def write_summary(table):

    lines = [
        "EXPERIMENT 13 - DiNTS MASK CLEAN-UP",
        "=" * 80,
        f"13-A: keep largest pancreas+tumor complex + fragments within {GAP_MM} mm.",
        f"13-B (final): keep components within {ANCHOR_MM} mm of the TotalSegmentator pancreas.",
        f"Both: remove tumor candidate components < {MIN_TUMOR_ML} mL; fill holes.",
        "",
        "Clean-up flag (13-B):",
        table.cleanup_flag.value_counts().to_string(),
        "",
        "Median (min-max) over patients with a non-empty raw mask:",
    ]

    valid = table[table.raw_pancreas_ml + table.raw_tumor_candidate_ml > 0]

    for column in [
        "largest_removed_pct",
        "removed_pct",
        "raw_dice_vs_totalseg",
        "largest_dice_vs_totalseg",
        "clean_dice_vs_totalseg",
        "raw_tumor_candidate_components",
        "clean_tumor_candidate_components",
        "clean_tumor_near_totalseg_pct",
    ]:

        values = valid[column].dropna()

        lines.append(
            f"  {column:34s} {values.median():8.3f}  ({values.min():.3f} - {values.max():.3f})"
        )

    lines += [
        "",
        "13-A vs 13-B, patients where the variants differ:",
        table.loc[table.largest_removed_pct != table.removed_pct, ["patient", "largest_removed_pct", "largest_dice_vs_totalseg", "largest_tumor_candidate_ml", "removed_pct", "clean_dice_vs_totalseg", "clean_tumor_candidate_ml"]].to_string(index=False),
        "",
        "Patients whose tumor candidate is mostly (< 50 %) away from the TotalSegmentator pancreas:",
        table.loc[table.clean_tumor_near_totalseg_pct < 50, ["patient", "clean_tumor_candidate_ml", "clean_tumor_near_totalseg_pct"]].to_string(index=False),
        "",
        "Flagged patients:",
        table.loc[table.cleanup_flag != "NO_CHANGE", ["patient", "cleanup_flag", "removed_ml", "removed_pct", "tumor_specks_removed", "raw_dice_vs_totalseg", "clean_dice_vs_totalseg"]].to_string(index=False),
        "",
        "IMPORTANT: Dice vs TotalSegmentator is agreement between two models, not accuracy.",
        "TotalSegmentator's pancreas may exclude a large tumor, which lowers the Dice for those cases.",
    ]

    return "\n".join(lines)


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    patients = sys.argv[1:] or PATIENTS

    print("=" * 80)
    print("EXPERIMENT 13")
    print("DiNTS MASK CLEAN-UP")
    print("=" * 80)

    rows = []

    for number, patient in enumerate(patients, start=1):

        try:

            row = clean_patient(patient)

        except Exception as error:

            row = {"patient": patient, "cleanup_flag": "ERROR", "error": str(error)}

        rows.append(row)

        print(
            f"[{number:02d}/{len(patients)}] {patient}: "
            f"removed {row.get('removed_ml')} mL ({row.get('removed_pct')} %) | "
            f"tumor comps {row.get('raw_tumor_candidate_components')} -> {row.get('clean_tumor_candidate_components')} | "
            f"Dice vs TotalSeg {row.get('raw_dice_vs_totalseg')} -> {row.get('clean_dice_vs_totalseg')} | "
            f"{row.get('cleanup_flag')}",
            flush=True,
        )

    table = pd.DataFrame(rows)

    table_csv = OUTPUT_ROOT / "Phase2_Experiment13_MaskCleanup.csv"

    table.to_csv(table_csv, index=False, encoding="utf-8-sig")

    summary = write_summary(table)

    (OUTPUT_ROOT / "Phase2_Experiment13_Summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)

    if len(patients) == len(PATIENTS) and CASE_TABLE.exists():

        merged_columns = [
            "clean_pancreas_ml",
            "clean_tumor_candidate_ml",
            "clean_dice_vs_totalseg",
            "clean_tumor_near_totalseg_pct",
            "totalseg_pancreas_ml",
            "cleanup_flag",
        ]

        case_table = pd.read_csv(CASE_TABLE).drop(columns=merged_columns, errors="ignore")

        case_table = case_table.merge(table[["patient"] + merged_columns], on="patient", how="left")

        case_table.to_csv(CASE_TABLE, index=False, encoding="utf-8-sig")

        print("\nCase table updated:")
        print(CASE_TABLE)


if __name__ == "__main__":

    main()
