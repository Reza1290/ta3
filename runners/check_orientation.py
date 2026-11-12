import SimpleITK as sitk
from pathlib import Path
import numpy as np

# Check original DICOM file using SimpleITK
dcm_path = Path("data/DICOMM/000.dcm")
if dcm_path.exists():
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(dcm_path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    print("=== Original DICOM Metadata (000.dcm) ===")
    for k in ["0020|0032", "0020|0037", "0028|0030", "0018|0050", "0018|5100"]:
        if reader.HasMetaDataKey(k):
            print(f"Tag {k}: {reader.GetMetaData(k)}")

# Check output_aligned/volume.mha
mha_path = Path("data/DICOMM/output_aligned/volume.mha")
if mha_path.exists():
    img = sitk.ReadImage(str(mha_path))
    print("\n=== output_aligned/volume.mha ===")
    print("Size (X, Y, Z):", img.GetSize())
    print("Spacing:", img.GetSpacing())
    print("Origin:", img.GetOrigin())
    print("Direction:", img.GetDirection())

# Check pred_fdi_mask.mha
mask_path = Path("data/DICOMM/output_aligned/pred_fdi_mask.mha")
if mask_path.exists():
    mask = sitk.ReadImage(str(mask_path))
    print("\n=== output_aligned/pred_fdi_mask.mha ===")
    print("Size (X, Y, Z):", mask.GetSize())
    print("Spacing:", mask.GetSpacing())
    print("Origin:", mask.GetOrigin())
    print("Direction:", mask.GetDirection())
