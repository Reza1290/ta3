"""Export a 3D CBCT MetaImage (.mha) into a standard clinical DICOM slice series (.dcm).

Creates a DICOM directory with consistent PatientID, SeriesUID, PixelSpacing,
and SliceLocation tags for clinical viewer compatibility.
"""

from pathlib import Path
import time
import numpy as np
import SimpleITK as sitk


def export_mha_to_dicom_series(
    mha_path: Path,
    output_dicom_dir: Path,
    patient_name: str = "CASE_008_COMPLETE_32TEETH",
    patient_id: str = "TF2_008",
):
    output_dicom_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Reading MHA volume: {mha_path}...")
    img = sitk.ReadImage(str(mha_path))
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    
    num_slices = arr.shape[0]
    print(f"[+] Loaded volume: {arr.shape} (ZYX), Spacing: {spacing}")
    print(f"[*] Exporting {num_slices} DICOM slices to {output_dicom_dir}...")
    
    series_uid = f"1.2.826.0.1.3680043.2.1125.8.{int(time.time())}"
    study_uid = f"1.2.826.0.1.3680043.2.1125.9.{int(time.time())}"
    
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    
    for i in range(num_slices):
        slice_arr = np.clip(arr[i, :, :], -1000, 3000).astype(np.int16)
        slice_img = sitk.GetImageFromArray(slice_arr.reshape(1, slice_arr.shape[0], slice_arr.shape[1]))
        slice_img.SetSpacing((spacing[0], spacing[1], spacing[2]))
        
        # Calculate physical slice location
        slice_z = origin[2] + i * spacing[2]
        slice_origin = (origin[0], origin[1], slice_z)
        slice_img.SetOrigin(slice_origin)
        slice_img.SetDirection(direction)
        
        # Add DICOM metadata
        slice_img.SetMetaData("0010|0010", patient_name)
        slice_img.SetMetaData("0010|0020", patient_id)
        slice_img.SetMetaData("0008|0060", "CT")
        slice_img.SetMetaData("0008|1030", "Dental CBCT Full Dentition")
        slice_img.SetMetaData("0020|000d", study_uid)
        slice_img.SetMetaData("0020|000e", series_uid)
        slice_img.SetMetaData("0020|0013", str(i + 1))  # Instance Number
        slice_img.SetMetaData("0020|1041", f"{slice_z:.3f}")  # Slice Location
        slice_img.SetMetaData("0020|0032", f"{origin[0]:.3f}\\{origin[1]:.3f}\\{slice_z:.3f}")
        
        out_dcm = output_dicom_dir / f"slice_{i:04d}.dcm"
        writer.SetFileName(str(out_dcm))
        writer.Execute(slice_img)
        
    print(f"[+] Successfully exported {num_slices} DICOM slices to: {output_dicom_dir}")


if __name__ == "__main__":
    export_mha_to_dicom_series(
        mha_path=Path(r"D:\Workspace\TA3\data\raw\ToothFairy2F_008_0000.mha"),
        output_dicom_dir=Path(r"D:\Workspace\TA3\data\DICOM_CASE_008"),
    )
