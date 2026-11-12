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
mask_img = sitk.ReadImage(str(mask_path))
mask_arr = sitk.GetArrayFromImage(mask_img)

spacing = np.array(mask_img.GetSpacing())
spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
sp = spacing_zyx.astype(np.float64)

# In test_dicom_aligned.py:
mask_m2 = (mask_arr == 27)
mask_m3 = (mask_arr == 28)

surf_m2 = extract_surface_voxels(mask_m2)
surf_m3 = extract_surface_voxels(mask_m3)

phys_m2 = surf_m2.astype(np.float64) * sp
phys_m3 = surf_m3.astype(np.float64) * sp

tree = cKDTree(phys_m2)
dists, nearest_m2_idxs = tree.query(phys_m3)
best_m3_idx = int(np.argmin(dists))
dmin_mm = float(dists[best_m3_idx])
print(f"Computed D_min(27, 28): {dmin_mm} mm")
print(f"best_m3_idx: {best_m3_idx}, best_m3_voxel: {surf_m3[best_m3_idx]}")
print(f"nearest_m2_voxel: {surf_m2[nearest_m2_idxs[best_m3_idx]]}")

# Wait, why did test_dicom_aligned.py print 11.379 mm earlier?!
# Could test_dicom_aligned.py have read a different mask_path or was output_aligned updated?!
print("Checking file modification times and paths...")
