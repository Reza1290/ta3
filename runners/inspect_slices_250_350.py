import SimpleITK as sitk
import numpy as np

mask_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
mask_arr = sitk.GetArrayFromImage(mask_img)

# In slice Z=250:
slice_250 = mask_arr[250, :, :]
unique_250 = np.unique(slice_250)
print("Unique labels at Z=250:", unique_250.tolist())

# For each label at Z=250, print centroid in slice
for u in unique_250:
    if u == 0:
        continue
    pts = np.argwhere(slice_250 == u)
    cy, cx = pts.mean(axis=0)
    print(f"  FDI {u:2d}: count={len(pts):4d}, Y={cy:5.1f}, X={cx:5.1f}")

# In slice Z=350:
slice_350 = mask_arr[350, :, :]
unique_350 = np.unique(slice_350)
print("\nUnique labels at Z=350:", unique_350.tolist())
for u in unique_350:
    if u == 0:
        continue
    pts = np.argwhere(slice_350 == u)
    cy, cx = pts.mean(axis=0)
    print(f"  FDI {u:2d}: count={len(pts):4d}, Y={cy:5.1f}, X={cx:5.1f}")
