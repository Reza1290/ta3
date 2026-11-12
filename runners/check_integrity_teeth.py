import SimpleITK as sitk
import numpy as np
from pathlib import Path

p = Path("data/DICOM_OPENSOURCE/Integrity/Labeled/Mask/Integrity_L_001.nii.gz")
if p.exists():
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img)
    u = np.unique(arr)
    print("Integrity_L_001 unique labels:", u.tolist())
    for fdi in [17, 18, 27, 28, 37, 38, 47, 48]:
        print(f"  FDI {fdi}: {np.sum(arr == fdi)} voxels")
