import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------
# 1. SET YOUR MASTER DIRECTORY & PATIENT LIST
# ---------------------------------------------------------
BASE_DIR = "D:/Thesis Folder/Training_dataset"
PATIENT_LIST = ["Pat_01", "Pat_02", "Pat_03", "Pat_04", "Pat_05"]

# ---------------------------------------------------------
# 2. LOOP THROUGH PATIENTS & EXTRACT 2D SLICES
# ---------------------------------------------------------
X_slices, Y_slices = [], []

print("Loading NIfTI files from multiple patients...")

for patient in PATIENT_LIST:
    img_path = f"{BASE_DIR}/{patient}/Origional_images.nii"
    mask_path = f"{BASE_DIR}/{patient}/masked_images.nii"
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Skipping {patient} - Files not found.")
        continue
        
    print(f"Processing {patient}...")
    
    img_nifti = nib.load(img_path).get_fdata()
    mask_nifti = nib.load(mask_path).get_fdata()

    img_nifti = np.clip(img_nifti, -100, 250)
    img_nifti = (img_nifti - img_nifti.min()) / (img_nifti.max() - img_nifti.min() + 1e-8)

    mask_nifti = (mask_nifti > 0).astype(np.float32)

    for slice_idx in range(img_nifti.shape[2]):
        mask_slice = mask_nifti[:, :, slice_idx]
        if np.sum(mask_slice) > 50:
            img_slice = img_nifti[:, :, slice_idx]
            X_slices.append(img_slice)
            Y_slices.append(mask_slice)

X = torch.tensor(np.array(X_slices), dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(np.array(Y_slices), dtype=torch.float32).unsqueeze(1)

print(f"\nSUCCESS: Extracted a total of {len(X)} valid slices across all patients.")

# ---------------------------------------------------------
# 3. DEFINE MONAI U-NET MODEL, LOSS, AND OPTIMIZER
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------
# 4. PREPARE THE DATALOADER & TRAIN THE U-NET
# ---------------------------------------------------------
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

epochs = 40  
model.train()
print("\nStarting Training Loop...")

# NEW: Create a list to store the loss history for our graph
loss_history = []

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        inputs, targets = batch_x.to(device), batch_y.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    
    # NEW: Save the average loss for this epoch to our list
    loss_history.append(avg_loss)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs} --- Avg Dice Loss: {avg_loss:.4f}")

print("\nTraining Complete!")

# 5. VISUALIZE RESULTS (LOSS GRAPH + CT vs PREDICTION)
model.eval()
with torch.no_grad():
    sample_idx = len(X) // 2
    sample_img = X[sample_idx:sample_idx+1].to(device)
    prediction = torch.sigmoid(model(sample_img)).cpu().numpy()

ct_image = X[sample_idx, 0].numpy()
ground_truth = Y[sample_idx, 0].numpy()
ai_pred = (prediction[0, 0] > 0.5).astype(float)

# Hide the background zeros so we don't get giant colored squares
ground_truth_masked = np.ma.masked_where(ground_truth == 0, ground_truth)
ai_pred_masked = np.ma.masked_where(ai_pred == 0, ai_pred)

# Create a 1x4 grid for our plots
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot 1: The Loss Graph
axes[0].plot(range(1, epochs + 1), loss_history, color='blue', linewidth=2, marker='o', markersize=4)
axes[0].set_title("Training Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Average Dice Loss")
axes[0].grid(True)

# Plot 2: Original CT
axes[1].imshow(ct_image, cmap="gray")
axes[1].set_title("Original CT Slice")
axes[1].axis("off")

# Plot 3: Ground Truth
axes[2].imshow(ct_image, cmap="gray")
axes[2].imshow(ground_truth_masked, cmap="Reds", alpha=0.6)
axes[2].set_title("Ground Truth Mask")
axes[2].axis("off")

# Plot 4: AI Prediction
axes[3].imshow(ct_image, cmap="gray")
axes[3].imshow(ai_pred_masked, cmap="spring", alpha=0.6)
axes[3].set_title("AI Predicted Mask")
axes[3].axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. SAVE THE MODEL WEIGHTS
# ---------------------------------------------------------
torch.save(model.state_dict(), "pancreas_unet_weights.pth")
print("Model saved successfully as 'pancreas_unet_weights.pth'!")