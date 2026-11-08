"""
Audit all CT + pancreas mask pairs before training.

Expected structure:
Training_dataset/
    Pat_01/
        Origional_images.nii or Origional_images.nii.gz
        masked_images.nii or masked_images.nii.gz
    Pat_02/
        ...

Outputs:
    dataset_audit.csv

Checks:
- file existence
- CT/mask shape match
- affine match
- voxel spacing
- mask values (binary or not)
- foreground voxel count / percentage
- approximate mask volume in mL
- positive axial slice range
- empty mask detection
- NaN/Inf detection
"""

from pathlib import Path
import csv
import numpy as np
import nibabel as nib

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")
OUTPUT_CSV = BASE_DIR / "dataset_audit.csv"


def find_existing(patient_dir: Path, stem: str):
    candidates = [
        patient_dir / f"{stem}.nii",
        patient_dir / f"{stem}.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def audit_patient(patient_dir: Path):
    ct_path = find_existing(patient_dir, "Origional_images")
    mask_path = find_existing(patient_dir, "masked_images")

    result = {
        "patient": patient_dir.name,
        "ct_exists": ct_path is not None,
        "mask_exists": mask_path is not None,
        "ct_path": str(ct_path) if ct_path else "",
        "mask_path": str(mask_path) if mask_path else "",
        "status": "NOT_CHECKED",
    }

    if ct_path is None or mask_path is None:
        result["status"] = "FAIL_MISSING_FILE"
        return result

    try:
        ct_img = nib.load(str(ct_path))
        mask_img = nib.load(str(mask_path))

        ct = ct_img.get_fdata(dtype=np.float32)
        mask = mask_img.get_fdata()

        result["ct_shape"] = str(ct.shape)
        result["mask_shape"] = str(mask.shape)
        result["shape_match"] = ct.shape == mask.shape

        result["affine_match"] = bool(
            np.allclose(ct_img.affine, mask_img.affine, atol=1e-4)
        )

        ct_spacing = tuple(round(float(x), 4) for x in ct_img.header.get_zooms()[:3])
        mask_spacing = tuple(round(float(x), 4) for x in mask_img.header.get_zooms()[:3])

        result["ct_spacing_mm"] = str(ct_spacing)
        result["mask_spacing_mm"] = str(mask_spacing)
        result["spacing_match"] = np.allclose(ct_spacing, mask_spacing, atol=1e-4)

        result["ct_min"] = float(np.nanmin(ct))
        result["ct_max"] = float(np.nanmax(ct))
        result["ct_has_nan"] = bool(np.isnan(ct).any())
        result["ct_has_inf"] = bool(np.isinf(ct).any())

        result["mask_has_nan"] = bool(np.isnan(mask).any())
        result["mask_has_inf"] = bool(np.isinf(mask).any())

        unique_values = np.unique(mask)
        result["mask_unique_count"] = int(len(unique_values))
        result["mask_unique_values"] = str(unique_values[:20].tolist())

        binary_mask = (mask > 0).astype(np.uint8)

        result["mask_is_binary_0_1"] = bool(
            np.array_equal(unique_values, np.array([0, 1]))
            or np.array_equal(unique_values, np.array([0]))
            or np.array_equal(unique_values, np.array([1]))
        )

        foreground_voxels = int(binary_mask.sum())
        result["foreground_voxels"] = foreground_voxels
        result["foreground_percent"] = round(
            100.0 * foreground_voxels / binary_mask.size, 6
        )

        voxel_volume_mm3 = float(np.prod(ct_img.header.get_zooms()[:3]))
        result["mask_volume_ml"] = round(
            foreground_voxels * voxel_volume_mm3 / 1000.0, 3
        )

        slice_counts = np.sum(binary_mask, axis=(0, 1))
        positive_slices = np.where(slice_counts > 0)[0]

        result["positive_slice_count"] = int(len(positive_slices))

        if len(positive_slices) > 0:
            result["first_positive_slice"] = int(positive_slices.min())
            result["last_positive_slice"] = int(positive_slices.max())
            result["max_area_slice"] = int(np.argmax(slice_counts))
            result["max_area_pixels"] = int(slice_counts.max())
        else:
            result["first_positive_slice"] = ""
            result["last_positive_slice"] = ""
            result["max_area_slice"] = ""
            result["max_area_pixels"] = ""

        # Decide structural audit status.
        problems = []

        if not result["shape_match"]:
            problems.append("SHAPE_MISMATCH")

        if not result["affine_match"]:
            problems.append("AFFINE_MISMATCH")

        if not result["spacing_match"]:
            problems.append("SPACING_MISMATCH")

        if result["ct_has_nan"] or result["ct_has_inf"]:
            problems.append("INVALID_CT_VALUES")

        if result["mask_has_nan"] or result["mask_has_inf"]:
            problems.append("INVALID_MASK_VALUES")

        if foreground_voxels == 0:
            problems.append("EMPTY_MASK")

        if not result["mask_is_binary_0_1"]:
            problems.append("MASK_NOT_BINARY")

        if problems:
            result["status"] = "FAIL_" + "|".join(problems)
        else:
            result["status"] = "PASS_STRUCTURAL_AUDIT"

        return result

    except Exception as exc:
        result["status"] = f"ERROR: {exc}"
        return result


patient_dirs = sorted(
    p for p in BASE_DIR.iterdir()
    if p.is_dir() and p.name.lower().startswith(("pat_", "patient"))
)

if not patient_dirs:
    raise RuntimeError(
        f"No patient folders found inside:\n{BASE_DIR}"
    )

results = []

print("=" * 80)
print("PANCREAS DATASET AUDIT")
print("=" * 80)

for patient_dir in patient_dirs:
    result = audit_patient(patient_dir)
    results.append(result)

    print(f"\n{result['patient']}")
    print(f"  Status:          {result['status']}")

    if result.get("ct_shape"):
        print(f"  CT shape:        {result['ct_shape']}")
        print(f"  Mask shape:      {result['mask_shape']}")
        print(f"  Shape match:     {result['shape_match']}")
        print(f"  Affine match:    {result['affine_match']}")
        print(f"  CT spacing:      {result['ct_spacing_mm']}")
        print(f"  Mask spacing:    {result['mask_spacing_mm']}")
        print(f"  Mask values:     {result['mask_unique_values']}")
        print(f"  Foreground:      {result['foreground_voxels']} voxels")
        print(f"  Mask volume:     {result['mask_volume_ml']} mL")
        print(
            "  Positive slices: "
            f"{result['positive_slice_count']} "
            f"({result['first_positive_slice']}–{result['last_positive_slice']})"
        )

# Save one combined audit CSV.
all_keys = []
for row in results:
    for key in row.keys():
        if key not in all_keys:
            all_keys.append(key)

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

passed = sum(r["status"] == "PASS_STRUCTURAL_AUDIT" for r in results)
print(f"Patients checked: {len(results)}")
print(f"Passed structural audit: {passed}")
print(f"Needs attention: {len(results) - passed}")
print(f"\nAudit CSV saved to:\n{OUTPUT_CSV}")

print(
    "\nImportant: PASS_STRUCTURAL_AUDIT means the files are technically suitable "
    "for model training. It does NOT mean that the anatomical segmentation has "
    "been clinically validated by a radiologist."
)