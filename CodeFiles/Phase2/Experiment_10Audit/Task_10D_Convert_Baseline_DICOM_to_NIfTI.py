from pathlib import Path
from collections import defaultdict
import csv
import gc

import SimpleITK as sitk


# ============================================================
# PHASE 2 — TASK 10D
# PRETREATMENT SERIES SELECTION + DICOM -> NIFTI
# ============================================================


DICOM_ROOT = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder"
    r"\ctpred_sunitinib_pannet"
)


TASK10B_CSV = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder"
    r"\Results"
    r"\_Phase2_Experiment10B_Audit_DICOMDATASET"
    r"\Phase2_DICOM_Audit_Series.csv"
)


OUTPUT_ROOT = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder"
    r"\Results"
    r"\_Phase2_Experiment10D_Baseline_NIfTI"
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


SELECTED_CSV = (
    OUTPUT_ROOT
    / "Phase2_Task10D_Selected_Baseline_Series.csv"
)


QC_CSV = (
    OUTPUT_ROOT
    / "Phase2_Task10D_Conversion_QC.csv"
)


SUMMARY_TXT = (
    OUTPUT_ROOT
    / "Phase2_Task10D_Summary.txt"
)


# ============================================================
# HELPERS
# ============================================================


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


def series_folder_from_example(
    example_file
):

    path = (
        DICOM_ROOT
        / Path(example_file)
    )

    return path.parent


# ============================================================
# START
# ============================================================


print("=" * 90)

print(
    "PHASE 2 — TASK 10D"
)

print(
    "PRETREATMENT DICOM -> NIFTI CONVERSION"
)

print("=" * 90)


print(
    "\nDICOM dataset:"
)

print(
    DICOM_ROOT
)


if not DICOM_ROOT.exists():

    raise FileNotFoundError(
        DICOM_ROOT
    )


if not TASK10B_CSV.exists():

    raise FileNotFoundError(
        TASK10B_CSV
    )


# ============================================================
# LOAD TASK 10B AUDIT
# ============================================================


with TASK10B_CSV.open(
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


# ============================================================
# GROUP BY PATIENT
# ============================================================


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


if len(patients) != 38:

    raise RuntimeError(
        "Expected 38 patients but found "
        + str(
            len(
                patients
            )
        )
    )


# ============================================================
# SELECT EARLIEST SERIES
#
# Dataset rule:
# earlier scan = pretreatment candidate
# later scan   = first follow-up candidate
#
# Derived from the structure of the published
# CTpred-Sunitinib-panNET collection.
# ============================================================


selected_rows = []


for patient in sorted(
    patients
):

    rows = sorted(
        patients[
            patient
        ],
        key=lambda x:
        parse_date(
            x[
                "study_date"
            ]
        )
    )


    if len(rows) != 2:

        raise RuntimeError(
            f"{patient}: expected 2 series, "
            f"found {len(rows)}"
        )


    baseline = rows[0]

    followup = rows[1]


    if (
        parse_date(
            baseline[
                "study_date"
            ]
        )
        >=
        parse_date(
            followup[
                "study_date"
            ]
        )
    ):

        raise RuntimeError(
            f"{patient}: dates are not "
            f"chronologically distinct."
        )


    selected_rows.append(
        {
            "patient":
                patient,

            "selection_status":
                (
                    "PRETREATMENT_INFERRED_"
                    "FROM_COLLECTION_DESIGN"
                ),

            "selection_rule":
                (
                    "CHRONOLOGICALLY_"
                    "EARLIEST_STUDY"
                ),

            "baseline_date":
                baseline[
                    "study_date"
                ],

            "baseline_date_formatted":
                pretty_date(
                    baseline[
                        "study_date"
                    ]
                ),

            "followup_date":
                followup[
                    "study_date"
                ],

            "followup_date_formatted":
                pretty_date(
                    followup[
                        "study_date"
                    ]
                ),

            "baseline_series_uid":
                baseline[
                    "series_instance_uid"
                ],

            "baseline_series_number":
                baseline[
                    "series_number"
                ],

            "baseline_series_description":
                baseline[
                    "series_description"
                ],

            "baseline_protocol_name":
                baseline[
                    "protocol_name"
                ],

            "baseline_suggested_phase":
                baseline[
                    "suggested_phase"
                ],

            "baseline_orientation":
                baseline[
                    "orientation"
                ],

            "baseline_slice_thickness_mm":
                baseline[
                    "slice_thickness_mm"
                ],

            "baseline_pixel_spacing_mm":
                baseline[
                    "pixel_spacing_mm"
                ],

            "baseline_number_of_images":
                baseline[
                    "number_of_images"
                ],

            "manufacturer":
                baseline[
                    "manufacturer"
                ],

            "manufacturer_model":
                baseline[
                    "manufacturer_model"
                ],

            "reconstruction_kernel":
                baseline[
                    "reconstruction_kernel"
                ],

            "baseline_example_file":
                baseline[
                    "example_file"
                ],

            "radiomics_warning":
                (
                    "DERIVED_REFORMAT"
                    if
                    baseline[
                        "suggested_phase"
                    ]
                    == "DERIVED_REFORMAT"
                    else ""
                ),
        }
    )


# ============================================================
# SAVE BASELINE MANIFEST
# ============================================================


with SELECTED_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=
        selected_rows[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        selected_rows
    )


print(
    "\nBaseline manifest created:"
)

print(
    SELECTED_CSV
)


# ============================================================
# CONVERSION
# ============================================================


qc_rows = []


success_count = 0

failure_count = 0


for number, row in enumerate(
    selected_rows,
    start=1
):


    patient = row[
        "patient"
    ]


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"[{number:02d}/38] "
        f"{patient}"
    )


    series_uid = (
        row[
            "baseline_series_uid"
        ]
    )


    series_folder = (
        series_folder_from_example(
            row[
                "baseline_example_file"
            ]
        )
    )


    patient_output = (
        OUTPUT_ROOT
        / patient
    )


    patient_output.mkdir(
        parents=True,
        exist_ok=True
    )


    nifti_path = (
        patient_output
        / "baseline_ct.nii.gz"
    )


    qc = {

        "patient":
            patient,

        "baseline_date":
            row[
                "baseline_date"
            ],

        "series_uid":
            series_uid,

        "source_folder":
            str(
                series_folder
            ),

        "output_nifti":
            str(
                nifti_path
            ),

        "conversion_status":
            "",

        "size_x":
            "",

        "size_y":
            "",

        "size_z":
            "",

        "spacing_x_mm":
            "",

        "spacing_y_mm":
            "",

        "spacing_z_mm":
            "",

        "origin":
            "",

        "direction":
            "",

        "pixel_type":
            "",

        "intensity_min":
            "",

        "intensity_max":
            "",

        "qc_notes":
            "",
    }


    try:

        if not series_folder.exists():

            raise FileNotFoundError(
                series_folder
            )


        # ----------------------------------------------------
        # Find the DICOM files belonging specifically
        # to the selected SeriesInstanceUID
        # ----------------------------------------------------


        filenames = (
            sitk.ImageSeriesReader
            .GetGDCMSeriesFileNames(
                str(
                    series_folder
                ),
                series_uid
            )
        )


        if not filenames:

            # Some folders contain only one series and GDCM
            # may resolve it more reliably this way.

            available_uids = (
                sitk.ImageSeriesReader
                .GetGDCMSeriesIDs(
                    str(
                        series_folder
                    )
                )
            )


            if not available_uids:

                raise RuntimeError(
                    "No DICOM series detected "
                    "in source folder."
                )


            if (
                series_uid
                not in available_uids
            ):

                raise RuntimeError(
                    "Selected SeriesInstanceUID "
                    "was not found in folder."
                )


            filenames = (
                sitk.ImageSeriesReader
                .GetGDCMSeriesFileNames(
                    str(
                        series_folder
                    ),
                    series_uid
                )
            )


        print(
            "DICOM files:",
            len(
                filenames
            )
        )


        reader = (
            sitk.ImageSeriesReader()
        )


        reader.SetFileNames(
            filenames
        )


        image = (
            reader.Execute()
        )


        # ----------------------------------------------------
        # Native image properties
        # ----------------------------------------------------


        size = (
            image.GetSize()
        )


        spacing = (
            image.GetSpacing()
        )


        origin = (
            image.GetOrigin()
        )


        direction = (
            image.GetDirection()
        )


        # ----------------------------------------------------
        # Intensity statistics
        # ----------------------------------------------------


        statistics = (
            sitk.StatisticsImageFilter()
        )


        statistics.Execute(
            image
        )


        intensity_min = (
            statistics.GetMinimum()
        )


        intensity_max = (
            statistics.GetMaximum()
        )


        # ----------------------------------------------------
        # Write NIfTI WITHOUT resampling.
        # ----------------------------------------------------


        sitk.WriteImage(
            image,
            str(
                nifti_path
            ),
            True
        )


        # ----------------------------------------------------
        # Read it back for basic integrity check.
        # ----------------------------------------------------


        check = (
            sitk.ReadImage(
                str(
                    nifti_path
                )
            )
        )


        if (
            check.GetSize()
            !=
            image.GetSize()
        ):

            raise RuntimeError(
                "NIfTI read-back size "
                "does not match source."
            )


        qc[
            "conversion_status"
        ] = "SUCCESS"


        qc[
            "size_x"
        ] = size[0]

        qc[
            "size_y"
        ] = size[1]

        qc[
            "size_z"
        ] = size[2]


        qc[
            "spacing_x_mm"
        ] = spacing[0]

        qc[
            "spacing_y_mm"
        ] = spacing[1]

        qc[
            "spacing_z_mm"
        ] = spacing[2]


        qc[
            "origin"
        ] = str(
            origin
        )


        qc[
            "direction"
        ] = str(
            direction
        )


        qc[
            "pixel_type"
        ] = (
            image
            .GetPixelIDTypeAsString()
        )


        qc[
            "intensity_min"
        ] = intensity_min


        qc[
            "intensity_max"
        ] = intensity_max


        notes = []


        if (
            row[
                "radiomics_warning"
            ]
        ):

            notes.append(
                row[
                    "radiomics_warning"
                ]
            )


        if intensity_min > -500:

            notes.append(
                "CHECK_HU_MIN"
            )


        if intensity_max < 200:

            notes.append(
                "CHECK_HU_MAX"
            )


        qc[
            "qc_notes"
        ] = " | ".join(
            notes
        )


        success_count += 1


        print(
            "Size:",
            size
        )


        print(
            "Spacing:",
            spacing
        )


        print(
            "Intensity range:",
            intensity_min,
            "to",
            intensity_max
        )


        print(
            "SUCCESS"
        )


        del image
        del check
        del reader

        gc.collect()


    except Exception as error:


        failure_count += 1


        qc[
            "conversion_status"
        ] = "FAILED"


        qc[
            "qc_notes"
        ] = str(
            error
        )


        print(
            "FAILED:"
        )

        print(
            error
        )


    qc_rows.append(
        qc
    )


# ============================================================
# SAVE QC
# ============================================================


with QC_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:


    writer = csv.DictWriter(
        file,
        fieldnames=
        qc_rows[0].keys()
    )


    writer.writeheader()

    writer.writerows(
        qc_rows
    )


# ============================================================
# SUMMARY
# ============================================================


summary = [

    "=" * 90,

    "PHASE 2 — TASK 10D SUMMARY",

    "=" * 90,

    "",

    "Selection rule:",

    (
        "Chronologically earliest study "
        "selected as pretreatment/baseline."
    ),

    (
        "This assignment is inferred from "
        "the documented collection design."
    ),

    "",

    f"Patients expected: 38",

    f"Successful conversions: "
    f"{success_count}",

    f"Failed conversions: "
    f"{failure_count}",

    "",

    "No resampling was performed.",

    "No intensity normalization was performed.",

    "No CT windowing was performed.",

    "No segmentation was performed.",

    "",

    f"Baseline manifest:\n"
    f"{SELECTED_CSV}",

    "",

    f"QC file:\n"
    f"{QC_CSV}",

]


SUMMARY_TXT.write_text(
    "\n".join(
        summary
    ),
    encoding="utf-8"
)


# ============================================================
# FINISH
# ============================================================


print(
    "\n"
    + "=" * 90
)


print(
    "TASK 10D COMPLETE"
)


print(
    "=" * 90
)


print(
    "\nSuccessful:",
    success_count
)


print(
    "Failed:",
    failure_count
)


print(
    "\nManifest:"
)

print(
    SELECTED_CSV
)


print(
    "\nQC:"
)

print(
    QC_CSV
)
