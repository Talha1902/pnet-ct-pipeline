import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet

# ---------------------------------------------------------
# 1. SET YOUR PATHS
# ---------------------------------------------------------
TEST_IMAGE_PATH = "D:/Thesis Folder/Training_dataset/Pat_06/Origional_images.nii.gz"
MODEL_WEIGHTS_PATH = "pancreas_unet_weights.pth"
OUTPUT_MASK_PATH = "D:/Thesis Folder/Training_dataset/Pat_06/UNET_AI_Pancreas_Prediction.nii.gz"

# ---------------------------------------------------------
# 2. LOAD THE SAVED MODEL
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading AI model on: {device}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval() 
print("Model loaded successfully!")

# ---------------------------------------------------------
# 3. LOAD & PREPARE THE RAW 3D CT SCAN
# ---------------------------------------------------------
print("Loading patient CT scan...")
original_nifti = nib.load(TEST_IMAGE_PATH)
img_nifti = original_nifti.get_fdata()

# We need an empty 3D block to store our AI's predictions
predicted_3d_volume = np.zeros_like(img_nifti)

# Preprocess the CT scan
img_nifti = np.clip(img_nifti, -100, 250)
img_nifti = (img_nifti - img_nifti.min()) / (img_nifti.max() - img_nifti.min() + 1e-8)

# ---------------------------------------------------------
# 4. RECONSTRUCT THE 3D PREDICTION
# ---------------------------------------------------------
print("AI is scanning the patient and building the 3D mask...")

for slice_idx in range(img_nifti.shape[2]):
    test_slice = img_nifti[:, :, slice_idx]
    
    input_tensor = torch.tensor(test_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad(): 
        prediction_tensor = torch.sigmoid(model(input_tensor))

    # Binarize prediction (1 = Pancreas, 0 = Background)
    ai_pred = (prediction_tensor[0, 0].cpu().numpy() > 0.5).astype(float)
    
    # Insert this 2D prediction slice back into our empty 3D volume
    predicted_3d_volume[:, :, slice_idx] = ai_pred

# ---------------------------------------------------------
# 5. SAVE THE FINAL 3D NIFTI FILE FOR 3D SLICER
# ---------------------------------------------------------
# We use the 'affine' from the original CT so it aligns perfectly in 3D space
new_nifti_file = nib.Nifti1Image(predicted_3d_volume, original_nifti.affine)
nib.save(new_nifti_file, OUTPUT_MASK_PATH)

print(f"\nSUCCESS! Fully 3D AI Prediction saved to:\n{OUTPUT_MASK_PATH}")

# ---------------------------------------------------------
# 6. QUICK 2D PREVIEW (Middle of the Pancreas)
# ---------------------------------------------------------
# Find the slice with the most pancreas pixels just to show a quick preview
max_pancreas_slice = np.argmax(np.sum(predicted_3d_volume, axis=(0, 1)))

preview_ct = img_nifti[:, :, max_pancreas_slice]
preview_mask = predicted_3d_volume[:, :, max_pancreas_slice]
preview_mask_transparent = np.ma.masked_where(preview_mask == 0, preview_mask)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(preview_ct, cmap="gray")
axes[0].set_title(f"Raw CT (Slice {max_pancreas_slice})")
axes[0].axis("off")

axes[1].imshow(preview_ct, cmap="gray")
axes[1].imshow(preview_mask_transparent, cmap="Greens", alpha=0.6)
axes[1].set_title("AI Pancreas Prediction")
axes[1].axis("off")

plt.tight_layout()
plt.show()