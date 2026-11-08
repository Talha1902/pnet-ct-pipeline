from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import matplotlib.pyplot as plt
import pydicom


# =============================================================================
# PHASE 2 — TASK 10C
# BASELINE / FOLLOW-UP IDENTIFICATION + VISUAL VERIFICATION
# =============================================================================


# -------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------

ROOT = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder\ctpred_sunitinib_pannet"
)

AUDIT_CSV = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder\Results"
    r"\_Phase2_Experiment10B_Audit_DICOMDATASET"
    r"\Phase2_DICOM_Audit_Series.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder\Results"
    r"\_Phase2_Experiment10C_Baseline_VisualReview"
)

MONTAGE_DIR = OUTPUT_DIR / "Montages"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MONTAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


REVIEW_CSV = (
    OUTPUT_DIR
    / "Phase2_Task10C_Timepoint_Review.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "Phase2_Task10C_Summary.txt"
)

HTML_INDEX = (
    OUTPUT_DIR
    / "Phase2_Task10C_Montage_Index.html"
)


# -------------------------------------------------------------------------
# DISPLAY SETTINGS
# -------------------------------------------------------------------------

# Standard abdominal soft-tissue window
WINDOW_CENTER = 40
WINDOW_WIDTH = 400

# Representative locations through each CT volume
SLICE_FRACTIONS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# =============================================================================
# HELPERS
# =============================================================================


def clean(value):

    if value is None:
        return ""

    return str(value).strip()


def parse_date(value):

    value = clean(value)

    if (
        len(value) == 8
        and value.isdigit()
    ):
        return int(value)

    return 99999999


def pretty_date(value):

    value = clean(value)

    if (
        len(value) == 8
        and value.isdigit()
    ):

        return (
            value[0:4]
            + "-"
            + value[4:6]
            + "-"
            + value[6:8]
        )

    return value


def get_series_folder(example_file):

    full_path = (
        ROOT
        / Path(example_file)
    )

    return full_path.parent


# =============================================================================
# DICOM SORTING
# =============================================================================


def read_header(path):

    try:

        return pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True
        )

    except Exception:

        return None


def slice_position(ds, fallback):

    # ---------------------------------------------------------
    # Best method:
    # ImagePositionPatient projected along slice normal
    # ---------------------------------------------------------

    try:

        orientation = np.array(
            ds.ImageOrientationPatient,
            dtype=float
        )

        position = np.array(
            ds.ImagePositionPatient,
            dtype=float
        )

        row = orientation[:3]
        column = orientation[3:]

        normal = np.cross(
            row,
            column
        )

        return float(
            np.dot(
                position,
                normal
            )
        )

    except Exception:
        pass


    # ---------------------------------------------------------
    # Second option
    # ---------------------------------------------------------

    try:

        return float(
            ds.SliceLocation
        )

    except Exception:
        pass


    # ---------------------------------------------------------
    # Third option
    # ---------------------------------------------------------

    try:

        return float(
            ds.InstanceNumber
        )

    except Exception:
        pass


    return float(fallback)


def load_series_file_list(
    series_folder
):

    files = []


    if not series_folder.exists():

        print(
            "WARNING: Missing series folder:",
            series_folder
        )

        return files


    for index, path in enumerate(
        series_folder.iterdir()
    ):

        if not path.is_file():
            continue


        ds = read_header(path)

        if ds is None:
            continue


        # Avoid random non-DICOM files
        if not (
            getattr(
                ds,
                "SeriesInstanceUID",
                None
            )
            or
            getattr(
                ds,
                "SOPInstanceUID",
                None
            )
        ):
            continue


        position = slice_position(
            ds,
            index
        )


        files.append(
            (
                position,
                path
            )
        )


    files.sort(
        key=lambda x: x[0]
    )


    return files


# =============================================================================
# CT DISPLAY
# =============================================================================


def dicom_to_display(ds):

    image = (
        ds.pixel_array
        .astype(np.float32)
    )


    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1.0
        )
    )


    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0.0
        )
    )


    image = (
        image * slope
        + intercept
    )


    low = (
        WINDOW_CENTER
        - WINDOW_WIDTH / 2
    )


    high = (
        WINDOW_CENTER
        + WINDOW_WIDTH / 2
    )


    image = np.clip(
        image,
        low,
        high
    )


    image = (
        image - low
    ) / (
        high - low
    )


    if (
        clean(
            getattr(
                ds,
                "PhotometricInterpretation",
                ""
            )
        )
        == "MONOCHROME1"
    ):

        image = 1.0 - image


    return image


# =============================================================================
# REPRESENTATIVE SLICES
# =============================================================================


def representative_indices(
    number_of_slices
):

    indices = []


    for fraction in SLICE_FRACTIONS:

        index = round(
            (
                number_of_slices - 1
            )
            * fraction
        )

        index = max(
            0,
            min(
                number_of_slices - 1,
                index
            )
        )


        indices.append(
            index
        )


    return indices


# =============================================================================
# COVERAGE FLAG
# =============================================================================


def automatic_coverage_flag(
    row
):

    text = " ".join(
        [
            clean(
                row.get(
                    "study_description"
                )
            ),

            clean(
                row.get(
                    "series_description"
                )
            ),

            clean(
                row.get(
                    "protocol_name"
                )
            ),
        ]
    ).lower()


    if (
        "head" in text
        or "brain" in text
    ):

        return (
            "REVIEW_HEAD_OR_BRAIN"
        )


    if (
        "localizer" in text
        or "scout" in text
    ):

        return (
            "REJECT_LOCALIZER"
        )


    if (
        "mpr" in text
        or "reformat" in text
    ):

        return (
            "REVIEW_DERIVED"
        )


    if (
        "pancreas" in text
        or "abdomen" in text
        or "abd " in text
    ):

        return (
            "LIKELY_ABDOMEN_COVERAGE"
        )


    if (
        "chest" in text
        or "thorax" in text
    ):

        return (
            "REVIEW_CHEST_COVERAGE"
        )


    return (
        "REVIEW_UNKNOWN_COVERAGE"
    )


# =============================================================================
# CREATE PATIENT MONTAGE
# =============================================================================


def make_patient_montage(
    patient,
    rows
):


    number_of_series = len(
        rows
    )


    number_of_columns = len(
        SLICE_FRACTIONS
    )


    fig, axes = plt.subplots(
        number_of_series,
        number_of_columns,
        figsize=(
            27,
            3.4 * number_of_series
        ),
        squeeze=False
    )


    for row_index, row in enumerate(
        rows
    ):


        series_folder = (
            get_series_folder(
                row[
                    "example_file"
                ]
            )
        )


        dicom_files = (
            load_series_file_list(
                series_folder
            )
        )


        print(
            patient,
            "timepoint",
            row_index + 1,
            ":",
            len(dicom_files),
            "images"
        )


        # Clear row
        for column in range(
            number_of_columns
        ):

            axes[
                row_index,
                column
            ].axis(
                "off"
            )


        if not dicom_files:
            continue


        indices = (
            representative_indices(
                len(
                    dicom_files
                )
            )
        )


        for column, index in enumerate(
            indices
        ):


            _, dicom_path = (
                dicom_files[
                    index
                ]
            )


            try:


                ds = (
                    pydicom.dcmread(
                        str(
                            dicom_path
                        ),
                        force=True
                    )
                )


                image = (
                    dicom_to_display(
                        ds
                    )
                )


                axes[
                    row_index,
                    column
                ].imshow(
                    image,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )


                instance = clean(
                    getattr(
                        ds,
                        "InstanceNumber",
                        ""
                    )
                )


                percent = int(
                    SLICE_FRACTIONS[
                        column
                    ]
                    * 100
                )


                axes[
                    row_index,
                    column
                ].set_title(
                    f"{percent}%\n"
                    f"Instance {instance}",
                    fontsize=8
                )


                axes[
                    row_index,
                    column
                ].axis(
                    "off"
                )


            except Exception as error:


                axes[
                    row_index,
                    column
                ].text(
                    0.5,
                    0.5,
                    "READ FAILED",
                    ha="center",
                    va="center"
                )


        # -----------------------------------------------------
        # Row information
        # -----------------------------------------------------


        date = pretty_date(
            row[
                "study_date"
            ]
        )


        description = clean(
            row[
                "series_description"
            ]
        )


        phase = clean(
            row[
                "suggested_phase"
            ]
        )


        label = (
            f"TIMEPOINT "
            f"{row_index + 1}\n"
            f"{date}\n"
            f"{phase}\n"
            f"{description[:30]}"
        )


        axes[
            row_index,
            0
        ].set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            labelpad=80,
            fontsize=9
        )


    fig.suptitle(
        f"{patient}\n"
        "Phase 2 Task 10C — "
        "Timepoint / Anatomy Review",
        fontsize=16
    )


    fig.tight_layout()


    output_path = (
        MONTAGE_DIR
        / (
            patient
            + "_Timepoint_Review.png"
        )
    )


    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    return output_path


# =============================================================================
# MAIN
# =============================================================================


print(
    "=" * 90
)

print(
    "PHASE 2 — TASK 10C"
)

print(
    "BASELINE / FOLLOW-UP "
    "VISUAL VERIFICATION"
)

print(
    "=" * 90
)


print(
    "\nDataset:"
)

print(
    ROOT
)


print(
    "\nTask 10B audit:"
)

print(
    AUDIT_CSV
)


if not ROOT.exists():

    raise FileNotFoundError(
        ROOT
    )


if not AUDIT_CSV.exists():

    raise FileNotFoundError(
        AUDIT_CSV
    )


# =============================================================================
# LOAD TASK 10B RESULTS
# =============================================================================


with AUDIT_CSV.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as file:


    audit_rows = list(
        csv.DictReader(
            file
        )
    )


print(
    "\nSeries loaded:",
    len(
        audit_rows
    )
)


patients = defaultdict(
    list
)


for row in audit_rows:

    patients[
        row[
            "patient_folder"
        ]
    ].append(
        row
    )


print(
    "Patients loaded:",
    len(
        patients
    )
)


# =============================================================================
# REVIEW TABLE
# =============================================================================


review_rows = []

html_entries = []


for number, patient in enumerate(
    sorted(
        patients
    ),
    start=1
):


    patient_rows = (
        patients[
            patient
        ]
    )


    # ---------------------------------------------------------
    # Sort by study date.
    #
    # IMPORTANT:
    # Earlier scan = candidate baseline only.
    # ---------------------------------------------------------


    patient_rows = sorted(
        patient_rows,
        key=lambda x:
        parse_date(
            x[
                "study_date"
            ]
        )
    )


    print(
        "\n"
        + "-" * 70
    )

    print(
        f"[{number:02d}/"
        f"{len(patients):02d}] "
        f"{patient}"
    )


    montage = (
        make_patient_montage(
            patient,
            patient_rows
        )
    )


    html_entries.append(
        (
            patient,
            montage
        )
    )


    for index, row in enumerate(
        patient_rows
    ):


        if index == 0:

            baseline_candidate = (
                "EARLIER_SCAN_CANDIDATE"
            )

        else:

            baseline_candidate = (
                "LATER_SCAN_CANDIDATE"
            )


        review_rows.append(
            {

                "patient":
                    patient,

                "timepoint_order":
                    index + 1,

                "timepoint_label":
                    (
                        "TIMEPOINT_"
                        + str(
                            index + 1
                        )
                    ),

                "study_date":
                    row[
                        "study_date"
                    ],

                "study_date_formatted":
                    pretty_date(
                        row[
                            "study_date"
                        ]
                    ),

                "baseline_candidate":
                    baseline_candidate,

                "series_number":
                    row[
                        "series_number"
                    ],

                "series_instance_uid":
                    row[
                        "series_instance_uid"
                    ],

                "study_description":
                    row[
                        "study_description"
                    ],

                "series_description":
                    row[
                        "series_description"
                    ],

                "protocol_name":
                    row[
                        "protocol_name"
                    ],

                "suggested_phase":
                    row[
                        "suggested_phase"
                    ],

                "orientation":
                    row[
                        "orientation"
                    ],

                "number_of_images":
                    row[
                        "number_of_images"
                    ],

                "pixel_spacing_mm":
                    row[
                        "pixel_spacing_mm"
                    ],

                "slice_thickness_mm":
                    row[
                        "slice_thickness_mm"
                    ],

                "manufacturer":
                    row[
                        "manufacturer"
                    ],

                "manufacturer_model":
                    row[
                        "manufacturer_model"
                    ],

                "reconstruction_kernel":
                    row[
                        "reconstruction_kernel"
                    ],

                "automatic_coverage_flag":
                    automatic_coverage_flag(
                        row
                    ),

                "montage":
                    str(
                        montage
                    ),

                # =============================================
                # MANUAL REVIEW
                # =============================================

                "manual_timepoint_role":
                    "",

                # PRETREATMENT
                # FOLLOWUP
                # UNCERTAIN


                "manual_pancreas_visible":
                    "",

                # YES / NO / UNCERTAIN


                "manual_phase":
                    "",

                # ARTERIAL
                # PORTAL_VENOUS
                # OTHER
                # UNCERTAIN


                "manual_use_for_phase2":
                    "",

                # YES / NO / REVIEW


                "manual_notes":
                    "",
            }
        )


# =============================================================================
# SAVE REVIEW CSV
# =============================================================================


with REVIEW_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:


    writer = csv.DictWriter(
        file,
        fieldnames=
        review_rows[0].keys()
    )


    writer.writeheader()


    writer.writerows(
        review_rows
    )


# =============================================================================
# CREATE HTML INDEX
# =============================================================================


html = []


html.append(
    "<html>"
)

html.append(
    "<head>"
)

html.append(
    "<title>"
    "Phase 2 Task 10C"
    "</title>"
)

html.append(
    """
    <style>

    body {
        font-family: Arial;
        margin: 30px;
        background: #f3f3f3;
    }

    .card {
        background: white;
        padding: 20px;
        margin-bottom: 30px;
        border-radius: 10px;
    }

    img {
        width: 100%;
        max-width: 1800px;
    }

    </style>
    """
)

html.append(
    "</head>"
)

html.append(
    "<body>"
)


html.append(
    "<h1>"
    "Phase 2 — Task 10C"
    "</h1>"
)


html.append(
    "<h2>"
    "Pretreatment / Follow-up "
    "Visual Review"
    "</h2>"
)


html.append(
    """
    <p>
    <b>Important:</b>
    TIMEPOINT 1 is only the earlier
    scan candidate. It must not be
    considered confirmed pretreatment
    imaging until manually reviewed.
    </p>
    """
)


for patient, montage in html_entries:


    relative = montage.relative_to(
        OUTPUT_DIR
    )


    relative = str(
        relative
    ).replace(
        "\\",
        "/"
    )


    html.append(
        "<div class='card'>"
    )


    html.append(
        f"<h2>{patient}</h2>"
    )


    html.append(
        f"<img src='{relative}'>"
    )


    html.append(
        "</div>"
    )


html.append(
    "</body>"
)

html.append(
    "</html>"
)


HTML_INDEX.write_text(
    "\n".join(
        html
    ),
    encoding="utf-8"
)


# =============================================================================
# SUMMARY
# =============================================================================


summary = [

    "=" * 90,

    "PHASE 2 — TASK 10C SUMMARY",

    "=" * 90,

    "",

    f"Patients processed: "
    f"{len(patients)}",

    f"Series reviewed: "
    f"{len(review_rows)}",

    f"Montages created: "
    f"{len(html_entries)}",

    "",

    "IMPORTANT:",

    "TIMEPOINT 1 is the earlier "
    "scan candidate only.",

    "It is NOT automatically "
    "confirmed as pretreatment.",

    "",

    "Review the montages before "
    "DICOM -> NIfTI conversion.",

    "",

    f"Review CSV:\n{REVIEW_CSV}",

    "",

    f"HTML index:\n{HTML_INDEX}",

]


SUMMARY_TXT.write_text(
    "\n".join(
        summary
    ),
    encoding="utf-8"
)


# =============================================================================
# FINISH
# =============================================================================


print(
    "\n"
    + "=" * 90
)


print(
    "TASK 10C COMPLETE"
)


print(
    "=" * 90
)


print(
    "\nReview CSV:"
)

print(
    REVIEW_CSV
)


print(
    "\nMontages:"
)

print(
    MONTAGE_DIR
)


print(
    "\nHTML index:"
)

print(
    HTML_INDEX
)


print(
    "\nSummary:"
)
