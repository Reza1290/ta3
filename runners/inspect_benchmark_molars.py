import SimpleITK as sitk
import numpy as np
from pathlib import Path

vol_path = Path("data/DICOM_BENCHMARK/output/volume.mha")
mask_path = Path("data/DICOM_BENCHMARK/output/pred_fdi_mask.mha")

vol_img = sitk.ReadImage(str(vol_path))
mask_img = sitk.ReadImage(str(mask_path))

vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

spacing = np.array(mask_img.GetSpacing()) # X, Y, Z (0.3, 0.3, 0.3)
spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
sp = spacing_zyx.astype(np.float64)

print(f"Volume shape: {vol_arr.shape}")
print(f"Spacing: {sp}")

teeth = [16, 17, 18, 26, 27, 28, 36, 37, 38, 46, 47, 48]
for t in teeth:
    pts = np.argwhere(mask_arr == t)
    if len(pts):
        cz, cy, cx = np.round(pts.mean(axis=0)).astype(int)
        zmin, zmax = pts[:,0].min(), pts[:,0].max()
        ymin, ymax = pts[:,1].min(), pts[:,1].max()
        xmin, xmax = pts[:,2].min(), pts[:,2].max()
        print(f"FDI {t}: cnt={len(pts):5d} | Centroid=[{cz:3d}, {cy:3d}, {cx:3d}] | Z:[{zmin:3d},{zmax:3d}] Y:[{ymin:3d},{ymax:3d}] X:[{xmin:3d},{xmax:3d}]")
    else:
        print(f"FDI {t}: NOT FOUND (0 voxels)")
