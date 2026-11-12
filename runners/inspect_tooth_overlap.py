import SimpleITK as sitk
import numpy as np

vol_img = sitk.ReadImage("data/DICOMM/output_aligned/volume.mha")
mask_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

# Let's inspect Tooth 27:
pts_27 = np.argwhere(mask_arr == 27)
pts_28 = np.argwhere(mask_arr == 28)

print(f"Tooth 27: {len(pts_27)} voxels")
print(f"  Z: min={pts_27[:,0].min()}, max={pts_27[:,0].max()}, span={pts_27[:,0].max()-pts_27[:,0].min()+1} ({ (pts_27[:,0].max()-pts_27[:,0].min()+1)*0.16:.2f} mm)")
print(f"  Y: min={pts_27[:,1].min()}, max={pts_27[:,1].max()}, span={pts_27[:,1].max()-pts_27[:,1].min()+1} ({ (pts_27[:,1].max()-pts_27[:,1].min()+1)*0.16:.2f} mm)")
print(f"  X: min={pts_27[:,2].min()}, max={pts_27[:,2].max()}, span={pts_27[:,2].max()-pts_27[:,2].min()+1} ({ (pts_27[:,2].max()-pts_27[:,2].min()+1)*0.16:.2f} mm)")

print(f"\nTooth 28: {len(pts_28)} voxels")
print(f"  Z: min={pts_28[:,0].min()}, max={pts_28[:,0].max()}, span={pts_28[:,0].max()-pts_28[:,0].min()+1} ({ (pts_28[:,0].max()-pts_28[:,0].min()+1)*0.16:.2f} mm)")
print(f"  Y: min={pts_28[:,1].min()}, max={pts_28[:,1].max()}, span={pts_28[:,1].max()-pts_28[:,1].min()+1} ({ (pts_28[:,1].max()-pts_28[:,1].min()+1)*0.16:.2f} mm)")
print(f"  X: min={pts_28[:,2].min()}, max={pts_28[:,2].max()}, span={pts_28[:,2].max()-pts_28[:,2].min()+1} ({ (pts_28[:,2].max()-pts_28[:,2].min()+1)*0.16:.2f} mm)")

# Check overlap in Z, Y, X:
overlap_z = max(0, min(pts_27[:,0].max(), pts_28[:,0].max()) - max(pts_27[:,0].min(), pts_28[:,0].min()))
overlap_y = max(0, min(pts_27[:,1].max(), pts_28[:,1].max()) - max(pts_27[:,1].min(), pts_28[:,1].min()))
overlap_x = max(0, min(pts_27[:,2].max(), pts_28[:,2].max()) - max(pts_27[:,2].min(), pts_28[:,2].min()))

print(f"\nBounding box overlap:")
print(f"  Z overlap: {overlap_z} voxels ({overlap_z*0.16:.2f} mm) -> Z range 28:[140, 190], 27:[180, 284]")
print(f"  Y overlap: {overlap_y} voxels ({overlap_y*0.16:.2f} mm)")
print(f"  X overlap: {overlap_x} voxels ({overlap_x*0.16:.2f} mm)")
