import SimpleITK as sitk
import numpy as np

img_itk = sitk.ReadImage("data/DICOMM/output_aligned/volume.mha")
mask_itk = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
arr_mask = sitk.GetArrayFromImage(mask_itk)
arr_mask_rad = arr_mask[:, :, ::-1]

# In z_maxilla = 250:
slice_m = arr_mask_rad[250, :, :]
teeth = np.unique(slice_m)
print("Teeth in slice Z=250:", teeth.tolist())
for fdi in teeth:
    if fdi == 0: continue
    bin_m = (slice_m == fdi)
    pts = np.argwhere(bin_m)
    cy, cx = pts.mean(axis=0)
    print(f"  FDI {fdi}: count={len(pts)}, Y={cy:.1f}, X={cx:.1f}")

# In z_mandible = 350:
slice_mand = arr_mask_rad[350, :, :]
teeth_mand = np.unique(slice_mand)
print("\nTeeth in slice Z=350:", teeth_mand.tolist())
for fdi in teeth_mand:
    if fdi == 0: continue
    bin_m = (slice_mand == fdi)
    pts = np.argwhere(bin_m)
    cy, cx = pts.mean(axis=0)
    print(f"  FDI {fdi}: count={len(pts)}, Y={cy:.1f}, X={cx:.1f}")
