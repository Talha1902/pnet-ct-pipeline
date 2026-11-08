"""Recreate the figures of the Exp 12-15 progress report from the result CSVs.

    Figure 1 (Exp 13): Dice of the DiNTS mask vs the TotalSegmentator pancreas,
                       one bar per patient, sorted - shows which cases fail.
    Figure 2 (Exp 14): tumor candidate volume per input strategy, relative to the
                       original DiNTS output - shows how unstable the tumor is.
    Figure 3 (Exp 15): Dice vs the thin-slice result as slices get thicker,
                       pancreas and tumor side by side - shows the thickness effect.

Runs locally (CPU, a few seconds). Output: Results/_Phase2_Report_Figures/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

RESULTS = ROOT / "Results"

OUTPUT_ROOT = RESULTS / "_Phase2_Report_Figures"

GREY = "#b4b2a9"

ACCENT = "#2a78d6"

STRATEGY_COLOURS = {
    "S1_pad": ("Pad 48 mm", "#1baf7a"),
    "S2_crop": ("Crop", ACCENT),
    "S3_crop_tta": ("Crop + TTA", "#e8643c"),
}


def figure_exp13():
    """One bar per patient: how much the DiNTS mask overlaps the TotalSegmentator pancreas."""

    table = pd.read_csv(RESULTS / "_Phase2_Experiment13_MaskCleanup" / "Phase2_Experiment13_MaskCleanup.csv")

    table = table.sort_values("clean_dice_vs_totalseg")

    dice = table.clean_dice_vs_totalseg

    # failures (Dice < 0.1) in the accent colour, everything else grey
    colours = [ACCENT if d < 0.1 else GREY for d in dice]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.bar(table.patient.str.replace("PAN_", ""), dice, color=colours, width=0.6)

    ax.axhline(dice.median(), color="grey", linestyle="--", linewidth=1)

    ax.text(0, dice.median() + 0.02, f"median {dice.median():.2f}", color="grey")

    ax.set_ylim(0, 1)

    ax.set_xlabel("Patient (PAN_xx), sorted by agreement")

    ax.set_ylabel("Dice vs TotalSegmentator")

    ax.set_title("DiNTS agrees with TotalSegmentator on most scans; four are near zero", loc="left")

    return fig


def figure_exp14():
    """One row per control patient: tumor candidate volume per strategy / original volume."""

    table = pd.read_csv(RESULTS / "_Phase2_Experiment14_RescueFailures" / "Phase2_Experiment14_RescueFailures.csv")

    patients = ["PAN_07", "PAN_03", "PAN_34", "PAN_05"]

    fig, ax = plt.subplots(figsize=(10, 4))

    for row, patient in enumerate(patients):

        rows = table[table.patient == patient].set_index("strategy")

        original = rows.loc["11A_original", "mask_tumor_candidate_ml"]

        ratios = {s: rows.loc[s, "mask_tumor_candidate_ml"] / original for s in STRATEGY_COLOURS}

        # grey line = spread between the strategies (and the original at 1x)
        values = list(ratios.values()) + [1.0]

        ax.plot([min(values), max(values)], [row, row], color=GREY, linewidth=3, zorder=1)

        for strategy, ratio in ratios.items():

            label, colour = STRATEGY_COLOURS[strategy]

            ax.scatter(ratio, row, s=70, color=colour, zorder=2, label=label if row == 0 else None)

    ax.axvline(1, color="grey", linestyle="--", linewidth=1)

    ax.set_yticks(range(len(patients)), patients)

    ax.invert_yaxis()

    ax.set_xlabel("Tumor candidate volume relative to the original DiNTS output (1x = unchanged)")

    ax.set_title("Input changes alone change the tumor candidate volume up to 2.6x", loc="left")

    ax.legend(loc="lower right")

    return fig


def figure_exp15():
    """Two panels: Dice vs the thin-slice result for the pancreas+tumor complex and for the tumor."""

    table = pd.read_csv(RESULTS / "_Phase2_Experiment15_SliceThickness" / "Phase2_Experiment15_SliceThickness.csv")

    table = table[(table.strategy == "S0_full") & (table.target_thickness_mm != "thin")].copy()

    table["target_thickness_mm"] = table.target_thickness_mm.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    panels = [
        ("dice_complex_vs_thin", "Pancreas + tumor complex", "black"),
        ("dice_tumor_vs_thin", "Tumor candidate", ACCENT),
    ]

    for ax, (column, title, colour) in zip(axes, panels):

        # one thin grey line per patient, starting at Dice 1.0 on the thin scan
        for _, rows in table.groupby("patient"):

            ax.plot([0.8] + list(rows.effective_thickness_mm), [1.0] + list(rows[column]), color=GREY, linewidth=1)

        median = table.groupby("target_thickness_mm")[[column, "effective_thickness_mm"]].median()

        ax.plot([0.8] + list(median.effective_thickness_mm), [1.0] + list(median[column]), color=colour, linewidth=2.5, marker="o")

        for x, y in zip(median.effective_thickness_mm, median[column]):

            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(6, 6), fontweight="bold")

        ax.set_title(title, loc="left")

        ax.set_xlabel("Slice thickness (mm), 0.8 = original thin scan")

        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Dice vs thin-slice result")

    fig.suptitle("Thicker slices barely move the pancreas but break up the tumor candidate", x=0.01, ha="left")

    return fig


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for name, make in [
        ("Fig1_Exp13_Agreement_vs_TotalSeg.png", figure_exp13),
        ("Fig2_Exp14_Tumor_Volume_Instability.png", figure_exp14),
        ("Fig3_Exp15_Slice_Thickness.png", figure_exp15),
    ]:

        fig = make()

        for ax in fig.axes:

            ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()

        fig.savefig(OUTPUT_ROOT / name, dpi=200)

        plt.close(fig)

        print("saved", OUTPUT_ROOT / name)


if __name__ == "__main__":

    main()
