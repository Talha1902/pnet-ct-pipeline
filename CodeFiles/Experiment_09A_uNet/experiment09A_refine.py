"""
Experiment 09A-R — Diversity-Constrained Hard-Negative Refinement
=================================================================

Purpose
-------
Refine the hard negatives mined in Experiment 09A BEFORE Experiment 09B.

Why
---
The original 09A ranking correctly finds difficult false-positive slices,
but many selected slices are consecutive or nearly consecutive. Because the
2.5D U-Net input for center slice z is [z-1, z, z+1], consecutive center
slices produce strongly overlapping 2.5D inputs.

This refinement keeps the same per-patient quota but greedily favors the
hardest slices while enforcing a minimum center-slice distance of 3 whenever
possible. This prevents overlapping 3-slice windows:

    selected z      -> [z-1, z, z+1]
    next allowed    -> z+3 or farther

If a patient cannot meet the quota with distance >=3, the script relaxes to
distance >=2, then >=1 only as needed.

No model inference and no retraining are performed.

Input
-----
Experiment_09A_all_negative_candidates.csv

Outputs
-------
Experiment_09A_diversified_hard_negatives.csv
Experiment_09A_diversification_summary.csv
Experiment_09A_diversification_summary.txt

Run:
    python experiment_09A_refine_diversity.py
"""

from pathlib import Path
import csv
import pandas as pd

INPUT_CSV = Path(
    r"D:/Thesis Folder/Results/Experiment_09A_results/"
    r"Experiment_09A_Hard_Negative_Mining/"
    r"Experiment_09A_all_negative_candidates.csv"
)

OUTPUT_DIR = INPUT_CSV.parent

OUTPUT_CSV = OUTPUT_DIR / "Experiment_09A_diversified_hard_negatives.csv"
SUMMARY_CSV = OUTPUT_DIR / "Experiment_09A_diversification_summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "Experiment_09A_diversification_summary.txt"

PRIMARY_MIN_CENTER_DISTANCE = 3

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found:\n{INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

required = {
    "patient",
    "slice_index",
    "predicted_pixels",
    "largest_2d_component_pixels",
    "sum_probability",
    "max_probability",
    "desired_hard_negatives_for_09B",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

def rank_candidates(group):
    return group.sort_values(
        by=[
            "predicted_pixels",
            "largest_2d_component_pixels",
            "sum_probability",
            "max_probability",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

def far_enough(candidate_slice, selected_slices, min_distance):
    if not selected_slices:
        return True
    return all(
        abs(int(candidate_slice) - int(existing)) >= min_distance
        for existing in selected_slices
    )

selected_rows = []
summary_rows = []

for patient, group in df.groupby("patient", sort=True):
    ranked = rank_candidates(group)

    quota = int(ranked["desired_hard_negatives_for_09B"].iloc[0])

    chosen_indices = []
    chosen_slices = set()
    spacing_stage_counts = {3: 0, 2: 0, 1: 0}

    # First pass: non-overlapping 2.5D windows.
    for min_distance in [3, 2, 1]:
        if len(chosen_indices) >= quota:
            break

        for idx, row in ranked.iterrows():
            if len(chosen_indices) >= quota:
                break

            slice_index = int(row["slice_index"])

            if slice_index in chosen_slices:
                continue

            if far_enough(
                slice_index,
                chosen_slices,
                min_distance,
            ):
                chosen_indices.append(idx)
                chosen_slices.add(slice_index)
                spacing_stage_counts[min_distance] += 1

    chosen = ranked.loc[chosen_indices].copy()

    # Re-rank chosen slices by hardness for readability.
    chosen = rank_candidates(chosen)
    chosen["diversified_hard_rank"] = range(1, len(chosen) + 1)

    # Record the minimum distance to another chosen slice.
    selected_slice_list = sorted(chosen["slice_index"].astype(int).tolist())

    min_neighbor_distances = {}
    for s in selected_slice_list:
        others = [x for x in selected_slice_list if x != s]
        min_neighbor_distances[s] = (
            min(abs(s - x) for x in others)
            if others
            else 999
        )

    chosen["nearest_selected_center_distance"] = (
        chosen["slice_index"]
        .astype(int)
        .map(min_neighbor_distances)
    )

    selected_rows.append(chosen)

    original_selected = group[group.get("selected_for_09B", 0) == 1].copy()

    original_overlap_count = 0
    if len(original_selected) > 1:
        original_slices = original_selected["slice_index"].astype(int).tolist()
        for s in original_slices:
            if any(abs(s - x) < 3 for x in original_slices if x != s):
                original_overlap_count += 1

    refined_overlap_count = int(
        (chosen["nearest_selected_center_distance"] < 3).sum()
    )

    summary_rows.append({
        "patient": patient,
        "quota": quota,
        "candidate_count": len(group),
        "original_09A_selected_count": len(original_selected),
        "original_selected_with_overlapping_2p5D_neighbor": original_overlap_count,
        "original_overlap_fraction": (
            original_overlap_count / len(original_selected)
            if len(original_selected) else 0.0
        ),
        "refined_selected_count": len(chosen),
        "refined_selected_with_overlapping_2p5D_neighbor": refined_overlap_count,
        "refined_overlap_fraction": (
            refined_overlap_count / len(chosen)
            if len(chosen) else 0.0
        ),
        "selected_at_distance_3_stage": spacing_stage_counts[3],
        "selected_at_distance_2_stage": spacing_stage_counts[2],
        "selected_at_distance_1_stage": spacing_stage_counts[1],
        "mean_fp_pixels_refined": float(chosen["predicted_pixels"].mean()),
        "median_fp_pixels_refined": float(chosen["predicted_pixels"].median()),
        "min_fp_pixels_refined": int(chosen["predicted_pixels"].min()),
        "max_fp_pixels_refined": int(chosen["predicted_pixels"].max()),
    })

final_selected = pd.concat(selected_rows, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

final_selected.to_csv(OUTPUT_CSV, index=False)
summary_df.to_csv(SUMMARY_CSV, index=False)

lines = [
    "=" * 88,
    "EXPERIMENT 09A-R — DIVERSITY-CONSTRAINED HARD-NEGATIVE REFINEMENT",
    "=" * 88,
    "",
    f"Input:  {INPUT_CSV}",
    f"Output: {OUTPUT_CSV}",
    "",
    f"Primary minimum center-slice distance: {PRIMARY_MIN_CENTER_DISTANCE}",
    "Reason: 2.5D input uses [z-1, z, z+1], so centers >=3 apart do not overlap.",
    "",
]

for row in summary_rows:
    lines.append(
        f"{row['patient']}: quota={row['quota']}, "
        f"original overlap={row['original_overlap_fraction']:.1%}, "
        f"refined overlap={row['refined_overlap_fraction']:.1%}, "
        f"distance3={row['selected_at_distance_3_stage']}, "
        f"distance2={row['selected_at_distance_2_stage']}, "
        f"distance1={row['selected_at_distance_1_stage']}"
    )

lines += [
    "",
    f"Total refined hard negatives: {len(final_selected)}",
    "",
    "Use this refined CSV — not the original 09A selected CSV — as the",
    "hard-negative source for Experiment 09B.",
]

SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

print("=" * 88)
print("EXPERIMENT 09A-R COMPLETE")
print("=" * 88)
print(summary_df.to_string(index=False))
print(f"\nTotal refined hard negatives: {len(final_selected)}")
print(f"\nRefined CSV:\n{OUTPUT_CSV}")
print(f"\nSummary CSV:\n{SUMMARY_CSV}")
print(f"\nSummary TXT:\n{SUMMARY_TXT}")