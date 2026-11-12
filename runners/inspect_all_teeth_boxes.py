import SimpleITK as sitk
import numpy as np

vol_img = sitk.ReadImage("data/DICOMM/output_aligned/volume.mha")
mask_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

print("Vol shape:", vol_arr.shape)
print("Mask shape:", mask_arr.shape)

# Let's inspect where teeth are along Y-axis:
# Why are there contours at Y > 400?
for fdi in sorted(np.unique(mask_arr)):
    if fdi == 0: continue
    pts = np.argwhere(mask_arr == fdi)
    z_min, z_max = pts[:,0].min(), pts[:,0].max()
    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    x_min, x_max = pts[:,2].min(), pts[:,2].max()
    print(f"FDI {fdi:2d}: voxels={len(pts):5d} | Z:[{z_min:3d}, {z_max:3d}] | Y:[{y_min:3d}, {y_max:3d}] | X:[{x_min:3d}, {x_max:3d}]")
