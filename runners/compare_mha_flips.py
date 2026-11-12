import SimpleITK as sitk
import numpy as np

# Compare PRATIWI_SISWI_DIANA.mha vs output_aligned/volume.mha
orig_path = "data/DICOMM/PRATIWI_SISWI_DIANA.mha"
aligned_path = "data/DICOMM/output_aligned/volume.mha"

orig_img = sitk.ReadImage(orig_path)
aligned_img = sitk.ReadImage(aligned_path)

orig_arr = sitk.GetArrayFromImage(orig_img) # Z, Y, X
aligned_arr = sitk.GetArrayFromImage(aligned_img)

print("Original MHA shape:", orig_arr.shape)
print("Aligned MHA shape:", aligned_arr.shape)

# Check if aligned_arr is literally orig_arr[::-1, :, :]
is_z_flipped = np.array_equal(aligned_arr, orig_arr[::-1, :, :])
print("Is aligned_arr literally orig_arr[::-1, :, :]? ->", is_z_flipped)

# Now let's check where the maxilla teeth vs mandible teeth are in orig_arr:
# In orig_arr:
# Slice 0 vs Slice 639
# Where is the maxilla?
# Slice 150 vs Slice 450
print("Orig slice 150 mean:", orig_arr[150].mean(), "bright:", np.sum(orig_arr[150] > 800))
print("Orig slice 450 mean:", orig_arr[450].mean(), "bright:", np.sum(orig_arr[450] > 800))
