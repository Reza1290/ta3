import SimpleITK as sitk
import numpy as np

# Load original (raw) DICOM volume
orig_img = sitk.ReadImage("data/DICOMM/PRATIWI_SISWI_DIANA.mha")
orig_arr = sitk.GetArrayFromImage(orig_img) # Z, Y, X

# Load aligned mask
mask_aligned_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
mask_aligned_arr = sitk.GetArrayFromImage(mask_aligned_img)

# In mask_aligned_arr:
# Tooth 11 (Upper Right Incisor) has Centroid Z=265
# Tooth 41 (Lower Right Incisor) has Centroid Z=378
# If we look at orig_arr at Z=265 vs Z=378:
# In orig_arr, what is at Z = 640 - 1 - 265 = 374?
# What is at Z = 640 - 1 - 378 = 261?

print(f"Volume shape: {orig_arr.shape}")

# Let's inspect teeth positions in original vs aligned:
# If mask_raw = mask_aligned_arr[::-1, :, :]:
mask_raw = mask_aligned_arr[::-1, :, :]

# Now in mask_raw:
# Tooth 11 will be at Z = 640 - 1 - 265 = 374!
# Tooth 41 will be at Z = 640 - 1 - 378 = 261!
# Let's check bone intensity in orig_arr at both locations:
for fdi in [11, 21, 31, 41, 17, 27, 37, 47]:
    pts_aligned = np.argwhere(mask_aligned_arr == fdi)
    pts_raw = np.argwhere(mask_raw == fdi)
    
    # Sample mean intensity in orig_arr using aligned coords vs raw coords:
    cz_al, cy_al, cx_al = np.round(pts_aligned.mean(axis=0)).astype(int)
    cz_rw, cy_rw, cx_rw = np.round(pts_raw.mean(axis=0)).astype(int)
    
    val_at_aligned = orig_arr[cz_al, cy_al, cx_al]
    val_at_raw = orig_arr[cz_rw, cy_rw, cx_rw]
    
    print(f"FDI {fdi:2d}: In orig_arr at aligned_coord Z={cz_al:3d}: HU={val_at_aligned:5.0f} | at raw_coord Z={cz_rw:3d}: HU={val_at_raw:5.0f}")
