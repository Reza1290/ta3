import SimpleITK as sitk
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import ndimage

def extract_surface_voxels(binary_mask: np.ndarray) -> np.ndarray:
    if not np.any(binary_mask):
        return np.empty((0, 3), dtype=np.int64)
    struct = ndimage.generate_binary_structure(3, 3)
    eroded = ndimage.binary_erosion(binary_mask, structure=struct)
    surface = binary_mask ^ eroded
    return np.argwhere(surface)

mask_path = Path("data/DICOMM/output_aligned/pred_fdi_mask.mha")
vol_path = Path("data/DICOMM/output_aligned/volume.mha")

mask_img = sitk.ReadImage(str(mask_path))
vol_img = sitk.ReadImage(str(vol_path))

mask_arr = sitk.GetArrayFromImage(mask_img) # Z, Y, X
spacing = np.array(mask_img.GetSpacing()) # X, Y, Z
spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])

print(f"Mask shape (Z, Y, X): {mask_arr.shape}")
print(f"Spacing (Z, Y, X): {spacing_zyx} mm")

# Find all unique labels in mask
unique_labels, counts = np.unique(mask_arr, return_counts=True)
print("\nUnique labels in mask:")
for lbl, cnt in zip(unique_labels, counts):
    if lbl == 0:
        continue
    pts = np.argwhere(mask_arr == lbl)
    z_min, z_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    x_min, x_max = pts[:, 2].min(), pts[:, 2].max()
    cz, cy, cx = np.round(pts.mean(axis=0)).astype(int)
    print(f"  FDI {lbl:2d}: voxels={cnt:6d} | Centroid=(Z={cz:3d}, Y={cy:3d}, X={cx:3d}) | Bounding Box Z:[{z_min:3d},{z_max:3d}] Y:[{y_min:3d},{y_max:3d}] X:[{x_min:3d},{x_max:3d}]")

# Let's inspect Q2 specifically: 21, 22, 23, 24, 25, 26, 27, 28
print("\n--- Quadrant 2 In-depth Analysis ---")
for fdi in [21, 22, 23, 24, 25, 26, 27, 28]:
    present = np.any(mask_arr == fdi)
    print(f"FDI {fdi}: {'PRESENT' if present else 'NOT FOUND'}")

# Now let's inspect the crop for 27-28 used in test_dicom_aligned.py
surf_27 = extract_surface_voxels(mask_arr == 27)
surf_28 = extract_surface_voxels(mask_arr == 28)
surf_26 = extract_surface_voxels(mask_arr == 26)

sp = spacing_zyx.astype(np.float64)

# Distance 27 to 28
tree_27 = cKDTree(surf_27.astype(np.float64) * sp)
d28_27, _ = tree_27.query(surf_28.astype(np.float64) * sp)
min_d_28_27 = d28_27.min()
print(f"\nPhysical min distance 28 to 27: {min_d_28_27:.3f} mm")

# Distance 28 to 26
tree_26 = cKDTree(surf_26.astype(np.float64) * sp)
d28_26, _ = tree_26.query(surf_28.astype(np.float64) * sp)
min_d_28_26 = d28_26.min()
print(f"Physical min distance 28 to 26: {min_d_28_26:.3f} mm")

# Distance 27 to 26
d27_26, _ = tree_26.query(surf_27.astype(np.float64) * sp)
min_d_27_26 = d27_26.min()
print(f"Physical min distance 27 to 26: {min_d_27_26:.3f} mm")

# Check where best_m3 and best_m2 voxels were:
best_idx = np.argmin(d28_27)
best_28 = surf_28[best_idx]
best_27 = surf_27[np.argmin(np.linalg.norm((surf_27 - best_28) * sp, axis=1))]
print(f"Best 28 voxel: {best_28}, Best 27 voxel: {best_27}")
center_zyx = np.round((best_28 + best_27) / 2.0).astype(int)
print(f"Center between 27 and 28: {center_zyx}")

# Let's check what is visible in the sagittal slice at X = center_zyx[2]
sag_slice_x = center_zyx[2]
mask_sag = mask_arr[:, :, sag_slice_x]
# In the 224x224 crop centered at (center_zyx[0], center_zyx[1]):
cz, cy = center_zyx[0], center_zyx[1]
crop_mask = mask_sag[cz-112:cz+112, cy-112:cy+112]
u_crop, c_crop = np.unique(crop_mask, return_counts=True)
print(f"\nUnique tooth labels visible in the 224x224 crop at X={sag_slice_x}:")
for u, c in zip(u_crop, c_crop):
    print(f"  Label {u}: {c} voxels")
