from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import UNet
from scipy import ndimage

# ============================================================
# EXPERIMENT 08 — BEST-MODEL FAILURE ANALYSIS
# NO RETRAINING. Pat_13/14/15 remain untouched.
# ============================================================

BASE_DIR = Path(r"D:/Thesis Folder/Training_dataset")
MODEL_PATH = Path(
    r"D:/Thesis Folder/Results/Experiment_08_results/"
    r"Experiment_08_2p5D_Three_Slice_UNet/"
    r"Experiment_08_best_model.pth"
)
OUTPUT_DIR = Path(
    r"D:/Thesis Folder/Results/Experiment_08_results/"
    r"Experiment_08_2p5D_Three_Slice_UNet/"
    r"Best_Model_Failure_Analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ["Pat_11", "Pat_12"]
HU_MIN, HU_MAX = -100.0, 250.0
PRED_THRESHOLD = 0.5
BATCH_SIZE = 8
MIN_MEANINGFUL_REFERENCE_PIXELS = 50
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)

WHOLE_SUMMARY = OUTPUT_DIR / "Experiment_08_best_model_whole_volume_summary.csv"
COMPONENT_SUMMARY = OUTPUT_DIR / "Experiment_08_component_summary.csv"
TEXT_SUMMARY = OUTPUT_DIR / "Experiment_08_failure_analysis_summary.txt"


def is_nifti(p: Path):
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def find_file(patient_dir: Path, kind: str):
    files = [p for p in patient_dir.iterdir() if p.is_file() and is_nifti(p)]
    if kind == "ct":
        preferred = [
            "origional_images.nii", "origional_images.nii.gz",
            "original_images.nii", "original_images.nii.gz",
        ]
        for name in preferred:
            for p in files:
                if p.name.lower() == name:
                    return p
        for p in files:
            n = p.name.lower()
            if ("origional" in n or "original" in n) and "mask" not in n and "prediction" not in n:
                return p
    if kind == "mask":
        preferred = [
            "masked_images.nii", "masked_images.nii.gz",
            "mask_images.nii", "mask_images.nii.gz",
        ]
        for name in preferred:
            for p in files:
                if p.name.lower() == name:
                    return p
        for p in files:
            n = p.name.lower()
            if ("masked_images" in n or "mask_images" in n) and "prediction" not in n:
                return p
    raise FileNotFoundError(f"Could not find {kind} NIfTI in {patient_dir}")


def preprocess_ct(ct):
    ct = np.clip(ct, HU_MIN, HU_MAX)
    return ((ct - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def load_patient(name):
    folder = BASE_DIR / name
    ct_img = nib.load(str(find_file(folder, "ct")))
    mask_img = nib.load(str(find_file(folder, "mask")))
    raw_ct = ct_img.get_fdata(dtype=np.float32)
    raw_mask = mask_img.get_fdata()
    if raw_ct.shape != raw_mask.shape:
        raise ValueError(f"{name}: shape mismatch {raw_ct.shape} vs {raw_mask.shape}")
    if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError(f"{name}: affine mismatch")
    orientation = nib.aff2axcodes(ct_img.affine)
    if orientation[2] not in ("S", "I"):
        raise ValueError(f"{name}: axis 2 is not axial: {orientation}")
    spacing = ct_img.header.get_zooms()[:3]
    return preprocess_ct(raw_ct), (raw_mask > 0).astype(np.uint8), ct_img, orientation, spacing


def three_slice_input(ct, z):
    last = ct.shape[2] - 1
    return np.stack([
        ct[:, :, max(0, z - 1)],
        ct[:, :, z],
        ct[:, :, min(last, z + 1)],
    ], axis=0).astype(np.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

print("=" * 84)
print("EXPERIMENT 08 — BEST-MODEL FAILURE ANALYSIS")
print("=" * 84)
print(f"Device: {device}")
print(f"Model:  {MODEL_PATH}")

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def predict_volume(ct):
    pred = np.zeros(ct.shape, dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, ct.shape[2], BATCH_SIZE):
            end = min(start + BATCH_SIZE, ct.shape[2])
            x_np = np.stack([three_slice_input(ct, z) for z in range(start, end)], axis=0)
            x = torch.from_numpy(x_np).to(device)
            probs = torch.sigmoid(model(x))[:, 0].cpu().numpy()
            batch_pred = (probs > PRED_THRESHOLD).astype(np.uint8)
            for i, z in enumerate(range(start, end)):
                pred[:, :, z] = batch_pred[i]
    return pred


def counts(pred, ref):
    pb, rb = pred.astype(bool), ref.astype(bool)
    tp = int(np.logical_and(pb, rb).sum())
    fp = int(np.logical_and(pb, ~rb).sum())
    fn = int(np.logical_and(~pb, rb).sum())
    return tp, fp, fn


def metric_values(tp, fp, fn):
    eps = 1e-8
    return (
        float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        float((tp + eps) / (tp + fp + fn + eps)),
        float((tp + eps) / (tp + fp + eps)),
        float((tp + eps) / (tp + fn + eps)),
    )


def range_text(indices):
    return "None" if len(indices) == 0 else f"{int(indices.min())}-{int(indices.max())}"


def volume_metrics(pred, ref):
    tp, fp, fn = counts(pred, ref)
    dice, iou, precision, recall = metric_values(tp, fp, fn)
    ref_counts = ref.sum(axis=(0, 1))
    pred_counts = pred.sum(axis=(0, 1))
    ref_pos = np.where(ref_counts > 0)[0]
    pred_pos = np.where(pred_counts > 0)[0]
    fp_slices = np.setdiff1d(pred_pos, ref_pos)
    missed = np.setdiff1d(ref_pos, pred_pos)
    return {
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "reference_voxels": int(ref.sum()), "predicted_voxels": int(pred.sum()),
        "true_positive_voxels": tp, "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "reference_positive_slices": int(len(ref_pos)),
        "predicted_positive_slices": int(len(pred_pos)),
        "reference_slice_range": range_text(ref_pos),
        "predicted_slice_range": range_text(pred_pos),
        "false_positive_slice_count": int(len(fp_slices)),
        "missed_reference_slice_count": int(len(missed)),
    }


def per_slice_metrics(pred, ref):
    rows = []
    for z in range(ref.shape[2]):
        p, r = pred[:, :, z], ref[:, :, z]
        tp, fp, fn = counts(p, r)
        rp, pp = int(r.sum()), int(p.sum())
        if rp == 0 and pp == 0:
            dice = iou = precision = recall = np.nan
        else:
            dice, iou, precision, recall = metric_values(tp, fp, fn)
        rows.append({
            "slice_index": z,
            "reference_pixels": rp,
            "predicted_pixels": pp,
            "true_positive_pixels": tp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "dice": dice, "iou": iou, "precision": precision, "recall": recall,
            "reference_positive": int(rp > 0),
            "prediction_positive": int(pp > 0),
        })
    return rows


def choose_slices(rows):
    ref_rows = [r for r in rows if r["reference_pixels"] > 0]
    meaningful = [r for r in rows if r["reference_pixels"] > MIN_MEANINGFUL_REFERENCE_PIXELS]
    if not ref_rows:
        raise ValueError("No reference-positive slices")
    largest_ref = max(ref_rows, key=lambda r: r["reference_pixels"])
    best = max(ref_rows, key=lambda r: -1 if np.isnan(r["dice"]) else r["dice"])
    pool = meaningful if meaningful else ref_rows
    worst = min(pool, key=lambda r: 2 if np.isnan(r["dice"]) else r["dice"])
    strongest_fp = max(rows, key=lambda r: r["false_positive_pixels"])
    return {
        "best_dice": best,
        "largest_reference": largest_ref,
        "worst_meaningful": worst,
        "strongest_false_positive": strongest_fp,
    }


def component_analysis(pred, ref, spacing):
    labeled, n = ndimage.label(pred.astype(bool), structure=CONNECTIVITY_26)
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    rows = []
    ref_total = int(ref.sum())
    for lab in range(1, n + 1):
        m = labeled == lab
        size = int(m.sum())
        overlap = int(np.logical_and(m, ref.astype(bool)).sum())
        coords = np.argwhere(m)
        mins, maxs, centroid = coords.min(0), coords.max(0), coords.mean(0)
        rows.append({
            "component_label": lab,
            "size_voxels": size,
            "volume_mm3": float(size * voxel_volume),
            "volume_ml": float(size * voxel_volume / 1000.0),
            "overlap_reference_voxels": overlap,
            "false_positive_voxels": size - overlap,
            "component_precision": float(overlap / size if size else 0),
            "fraction_of_total_reference_covered": float(overlap / ref_total if ref_total else 0),
            "touches_reference": int(overlap > 0),
            "axis0_min": int(mins[0]), "axis0_max": int(maxs[0]),
            "axis1_min": int(mins[1]), "axis1_max": int(maxs[1]),
            "axial_slice_min": int(mins[2]), "axial_slice_max": int(maxs[2]),
            "centroid_axis0": float(centroid[0]),
            "centroid_axis1": float(centroid[1]),
            "centroid_axial": float(centroid[2]),
        })
    rows.sort(key=lambda r: r["size_voxels"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["size_rank"] = rank

    pure_fp = [r for r in rows if not r["touches_reference"]]
    touching = [r for r in rows if r["touches_reference"]]
    pure_fp_voxels = sum(r["size_voxels"] for r in pure_fp)

    def frac_top(k):
        if pure_fp_voxels == 0:
            return 0.0
        return float(sum(r["size_voxels"] for r in pure_fp[:k]) / pure_fp_voxels)

    summary = {
        "total_components": len(rows),
        "components_touching_reference": len(touching),
        "pure_false_positive_components": len(pure_fp),
        "pure_false_positive_voxels": int(pure_fp_voxels),
        "largest_component_voxels": int(rows[0]["size_voxels"] if rows else 0),
        "largest_component_touches_reference": int(rows[0]["touches_reference"] if rows else 0),
        "largest_touching_component_voxels": int(touching[0]["size_voxels"] if touching else 0),
        "largest_pure_fp_component_voxels": int(pure_fp[0]["size_voxels"] if pure_fp else 0),
        "fraction_pure_fp_voxels_in_largest_pure_fp_component": frac_top(1),
        "fraction_pure_fp_voxels_in_top5_pure_fp_components": frac_top(5),
        "fraction_pure_fp_voxels_in_top10_pure_fp_components": frac_top(10),
    }
    return rows, summary


def error_map(pred_slice, ref_slice):
    e = np.zeros_like(ref_slice, dtype=np.uint8)
    e[(pred_slice == 1) & (ref_slice == 1)] = 1
    e[(pred_slice == 1) & (ref_slice == 0)] = 2
    e[(pred_slice == 0) & (ref_slice == 1)] = 3
    return e


def save_slice_figure(patient, title, z, ct, ref, pred, row, whole, path):
    c, r, p = ct[:, :, z], ref[:, :, z], pred[:, :, z]
    ro = np.ma.masked_where(r == 0, r)
    po = np.ma.masked_where(p == 0, p)
    eo = np.ma.masked_where(error_map(p, r) == 0, error_map(p, r))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(c, cmap="gray"); axes[0].set_title(f"Original CT\nSlice {z}")
    axes[1].imshow(c, cmap="gray"); axes[1].imshow(ro, alpha=.55); axes[1].set_title("Reference Mask")
    axes[2].imshow(c, cmap="gray"); axes[2].imshow(po, alpha=.55); axes[2].set_title("Experiment 08 Prediction")
    axes[3].imshow(c, cmap="gray"); axes[3].imshow(eo, alpha=.65); axes[3].set_title("Error Map\n1=overlap, 2=FP, 3=FN")
    for ax in axes: ax.axis("off")
    sd = "N/A" if np.isnan(row["dice"]) else f'{row["dice"]:.4f}'
    fig.suptitle(
        f"{patient} - {title}\nSlice Dice={sd} | Whole-volume Dice={whole['dice']:.4f} | "
        f"Precision={whole['precision']:.4f} | Recall={whole['recall']:.4f}", fontsize=12
    )
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def save_profiles(patient, rows, out_dir):
    x = np.array([r["slice_index"] for r in rows])
    refp = np.array([r["reference_pixels"] for r in rows])
    predp = np.array([r["predicted_pixels"] for r in rows])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, refp, label="Reference foreground pixels")
    ax.plot(x, predp, label="Predicted foreground pixels")
    ax.set_xlabel("Axial slice index"); ax.set_ylabel("Foreground pixels")
    ax.set_title(f"{patient} - Reference vs Prediction Across Axial Slices")
    ax.grid(True); ax.legend(); plt.tight_layout()
    plt.savefig(out_dir / "foreground_pixels_across_slices.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    ref_rows = [r for r in rows if r["reference_pixels"] > 0]
    xd = np.array([r["slice_index"] for r in ref_rows])
    yd = np.array([r["dice"] for r in ref_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xd, yd, marker="o", markersize=3)
    ax.set_xlabel("Axial slice index"); ax.set_ylabel("Slice Dice"); ax.set_ylim(-.02, 1.02)
    ax.set_title(f"{patient} - Dice on Reference-Positive Slices")
    ax.grid(True); plt.tight_layout()
    plt.savefig(out_dir / "dice_across_reference_positive_slices.png", dpi=300, bbox_inches="tight"); plt.close(fig)


whole_rows = []
component_summary_rows = []
text = []

for patient in PATIENTS:
    print("\n" + "=" * 84)
    print(f"ANALYZING {patient}")
    print("=" * 84)

    out_dir = OUTPUT_DIR / patient
    out_dir.mkdir(parents=True, exist_ok=True)

    ct, ref, ct_img, orientation, spacing = load_patient(patient)
    print(f"Shape:       {ct.shape}")
    print(f"Orientation: {orientation}")
    print(f"Spacing:     {spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f} mm")
    print("Running whole-volume 2.5D inference...")

    pred = predict_volume(ct)
    m = volume_metrics(pred, ref)

    print("\nWHOLE-VOLUME RESULTS")
    for k in ["dice", "iou", "precision", "recall"]:
        print(f"{k.capitalize():<10}{m[k]:.4f}")
    print(f"Reference voxels:      {m['reference_voxels']}")
    print(f"Predicted voxels:      {m['predicted_voxels']}")
    print(f"True positives:        {m['true_positive_voxels']}")
    print(f"False positives:       {m['false_positive_voxels']}")
    print(f"False negatives:       {m['false_negative_voxels']}")
    print(f"Reference slice range: {m['reference_slice_range']}")
    print(f"Predicted slice range: {m['predicted_slice_range']}")
    print(f"False-positive slices: {m['false_positive_slice_count']}")
    print(f"Missed reference:      {m['missed_reference_slice_count']}")

    whole_rows.append({
        "patient": patient,
        "shape": str(ct.shape),
        "orientation": "".join(orientation),
        "spacing_axis0_mm": float(spacing[0]),
        "spacing_axis1_mm": float(spacing[1]),
        "spacing_axis2_mm": float(spacing[2]),
        **m,
    })

    tp_mask = ((pred == 1) & (ref == 1)).astype(np.uint8)
    fp_mask = ((pred == 1) & (ref == 0)).astype(np.uint8)
    fn_mask = ((pred == 0) & (ref == 1)).astype(np.uint8)
    for arr, name in [
        (pred, "best_prediction"), (tp_mask, "true_positive"),
        (fp_mask, "false_positive"), (fn_mask, "false_negative")
    ]:
        nib.save(nib.Nifti1Image(arr.astype(np.uint8), ct_img.affine, ct_img.header),
                 str(out_dir / f"{patient}_Experiment08_{name}.nii.gz"))

    slice_rows = per_slice_metrics(pred, ref)
    with (out_dir / f"{patient}_Experiment08_per_slice_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(slice_rows[0].keys())); w.writeheader(); w.writerows(slice_rows)

    chosen = choose_slices(slice_rows)
    print("\nDIAGNOSTIC SLICES")
    for key, row in chosen.items():
        d = "N/A" if np.isnan(row["dice"]) else f'{row["dice"]:.4f}'
        print(f"{key}: slice {row['slice_index']} | Dice={d} | Ref={row['reference_pixels']} | "
              f"Pred={row['predicted_pixels']} | FP={row['false_positive_pixels']}")

    specs = [
        ("Best Dice Slice", "best_dice", "best_dice_slice.png"),
        ("Largest Reference Pancreas Slice", "largest_reference", "largest_reference_slice.png"),
        ("Worst Meaningful Pancreas Slice", "worst_meaningful", "worst_meaningful_slice.png"),
        ("Strongest False-Positive Slice", "strongest_false_positive", "strongest_false_positive_slice.png"),
    ]
    for title, key, fn in specs:
        row = chosen[key]
        save_slice_figure(patient, title, int(row["slice_index"]), ct, ref, pred, row, m, out_dir / fn)
    save_profiles(patient, slice_rows, out_dir)

    comp_rows, comp_summary = component_analysis(pred, ref, spacing)
    if comp_rows:
        fields = [
            "size_rank", "component_label", "size_voxels", "volume_mm3", "volume_ml",
            "overlap_reference_voxels", "false_positive_voxels", "component_precision",
            "fraction_of_total_reference_covered", "touches_reference", "axis0_min", "axis0_max",
            "axis1_min", "axis1_max", "axial_slice_min", "axial_slice_max",
            "centroid_axis0", "centroid_axis1", "centroid_axial"
        ]
        with (out_dir / f"{patient}_Experiment08_connected_components.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(comp_rows)

    component_summary_rows.append({"patient": patient, **comp_summary})
    print("\n3D CONNECTED-COMPONENT SUMMARY")
    for k, v in comp_summary.items():
        if isinstance(v, float): print(f"{k}: {v:.4f}")
        else: print(f"{k}: {v}")

    text.extend([
        "=" * 84, patient, "=" * 84,
        f"Dice={m['dice']:.4f}, IoU={m['iou']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}",
        f"TP={m['true_positive_voxels']}, FP={m['false_positive_voxels']}, FN={m['false_negative_voxels']}",
        f"Reference range={m['reference_slice_range']}, Predicted range={m['predicted_slice_range']}",
        f"False-positive slices={m['false_positive_slice_count']}, Missed reference slices={m['missed_reference_slice_count']}",
        f"Best Dice slice={chosen['best_dice']['slice_index']} Dice={chosen['best_dice']['dice']:.4f}",
        f"Largest-reference slice={chosen['largest_reference']['slice_index']} Dice={chosen['largest_reference']['dice']:.4f}",
        f"Worst meaningful slice={chosen['worst_meaningful']['slice_index']} Dice={chosen['worst_meaningful']['dice']:.4f}",
        f"Strongest FP slice={chosen['strongest_false_positive']['slice_index']} FP pixels={chosen['strongest_false_positive']['false_positive_pixels']}",
        f"Total components={comp_summary['total_components']}",
        f"Reference-touching components={comp_summary['components_touching_reference']}",
        f"Pure FP components={comp_summary['pure_false_positive_components']}",
        f"Largest component={comp_summary['largest_component_voxels']} voxels, touches reference={comp_summary['largest_component_touches_reference']}",
        f"Largest reference-touching component={comp_summary['largest_touching_component_voxels']} voxels",
        f"Largest pure FP component={comp_summary['largest_pure_fp_component_voxels']} voxels",
        f"Pure-FP fraction in top 5 components={comp_summary['fraction_pure_fp_voxels_in_top5_pure_fp_components']:.4f}",
        ""
    ])

with WHOLE_SUMMARY.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(whole_rows[0].keys())); w.writeheader(); w.writerows(whole_rows)

with COMPONENT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(component_summary_rows[0].keys())); w.writeheader(); w.writerows(component_summary_rows)

mean_dice = float(np.mean([r["dice"] for r in whole_rows]))
mean_precision = float(np.mean([r["precision"] for r in whole_rows]))
mean_recall = float(np.mean([r["recall"] for r in whole_rows]))
text.extend([
    "=" * 84,
    "EXPERIMENT 08 BEST-MODEL VALIDATION SUMMARY",
    "=" * 84,
    f"Mean Dice={mean_dice:.4f}",
    f"Mean Precision={mean_precision:.4f}",
    f"Mean Recall={mean_recall:.4f}",
    "Experiment 06A reference mean Dice=0.2698",
    "Pat_13, Pat_14, Pat_15 remain untouched.",
])
TEXT_SUMMARY.write_text("\n".join(text), encoding="utf-8")

print("\n" + "=" * 84)
print("EXPERIMENT 08 FAILURE ANALYSIS COMPLETE")
print("=" * 84)
for r in whole_rows:
    print(f"{r['patient']}: Dice={r['dice']:.4f}, Precision={r['precision']:.4f}, "
          f"Recall={r['recall']:.4f}, FP voxels={r['false_positive_voxels']}, "
          f"FP slices={r['false_positive_slice_count']}")
print(f"Mean validation Dice: {mean_dice:.4f}")
print(f"Mean precision:       {mean_precision:.4f}")
print(f"Mean recall:          {mean_recall:.4f}")
print(f"\nWhole-volume summary:\n{WHOLE_SUMMARY}")
print(f"\nComponent summary:\n{COMPONENT_SUMMARY}")
print(f"\nReadable summary:\n{TEXT_SUMMARY}")