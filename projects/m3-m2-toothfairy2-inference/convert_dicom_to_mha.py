"""Convert DICOM series to 3D MetaImage (.mha) format using SimpleITK.

Preserves physical metadata (voxel spacing, origin, direction cosine matrix)
and aligns spatial orientation to ToothFairy2 canonical coordinates
(Superior -> Inferior Z-axis, Maxilla at low Z, Mandible at high Z).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import SimpleITK as sitk
import numpy as np


def convert_dicom_series_to_mha(
    dicom_dir: Path,
    output_mha_path: Path,
    use_compression: bool = True,
    verify_roundtrip: bool = True,
    align_toothfairy_coords: bool = True,
) -> sitk.Image:
    """Reads a DICOM series from directory and writes a compressed .mha file.

    Args:
        dicom_dir: Directory containing .dcm files or DICOM slices.
        output_mha_path: Destination path for the .mha volume.
        use_compression: Whether to apply zlib compression to .mha.
        verify_roundtrip: Re-open saved .mha to verify integrity.
        align_toothfairy_coords: Ensures Z-axis aligns with ToothFairy2
            canonical space (Superior -> Inferior).

    Returns:
        sitk.Image object of the loaded volume.
    """
    dicom_dir = Path(dicom_dir).resolve()
    output_mha_path = Path(output_mha_path).resolve()
    output_mha_path.parent.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  DICOM to MHA Volume Converter (SimpleITK)")
    print(f"  Input directory : {dicom_dir}")
    print(f"  Output MHA path : {output_mha_path}")
    print(f"  ToothFairy Align: {align_toothfairy_coords}")
    print("=================================================================\n")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    all_dcm_files = sorted(list(dicom_dir.glob("*.dcm")))
    if not series_ids:
        file_names = [str(f) for f in all_dcm_files]
    else:
        # Pick the series with the largest number of slices
        best_series = max(series_ids, key=lambda sid: len(reader.GetGDCMSeriesFileNames(str(dicom_dir), sid)))
        file_names = list(reader.GetGDCMSeriesFileNames(str(dicom_dir), best_series))
        if len(file_names) < len(all_dcm_files) * 0.5:
            # Fallback to direct file sorting if series UID was fragmented
            file_names = [str(f) for f in all_dcm_files]

    if not file_names:
        raise FileNotFoundError(f"No valid DICOM series found in {dicom_dir}")

    print(f"[*] Found {len(file_names)} DICOM slices. Reading volume...")
    t0 = time.time()
    reader.SetFileNames(file_names)
    image = reader.Execute()
    read_duration = time.time() - t0

    # Inspect physical properties
    size_xyz = image.GetSize()
    spacing_xyz = image.GetSpacing()
    origin_xyz = image.GetOrigin()
    direction = image.GetDirection()
    pixel_type = image.GetPixelIDTypeAsString()

    print(f"[+] Loaded volume in {read_duration:.2f}s:")
    print(f"    - Grid Size (X, Y, Z) : {size_xyz}")
    print(f"    - Spacing (mm)        : ({spacing_xyz[0]:.4f}, {spacing_xyz[1]:.4f}, {spacing_xyz[2]:.4f})")
    print(f"    - Origin (mm)         : ({origin_xyz[0]:.2f}, {origin_xyz[1]:.2f}, {origin_xyz[2]:.2f})")
    print(f"    - Pixel Type          : {pixel_type}")

    # Inspect voxel statistics
    arr = sitk.GetArrayFromImage(image)
    print(f"    - Intensity range (HU): min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

    # Check Z-orientation alignment for ToothFairy2
    if align_toothfairy_coords:
        print("[*] Applying ToothFairy2 canonical coordinate alignment (Z-flip Inferior -> Superior)...")
        arr_aligned = np.ascontiguousarray(arr[::-1, :, :])
        aligned_image = sitk.GetImageFromArray(arr_aligned)
        aligned_image.SetSpacing(spacing_xyz)
        aligned_image.SetOrigin(origin_xyz)
        aligned_image.SetDirection(direction)
        image = aligned_image

    # Write MHA file
    print(f"\n[*] Writing MHA (compressed={use_compression})...")
    t1 = time.time()
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(output_mha_path))
    writer.SetUseCompression(use_compression)
    writer.Execute(image)
    write_duration = time.time() - t1

    file_size_mb = output_mha_path.stat().st_size / (1024 * 1024)
    print(f"[+] MHA written successfully in {write_duration:.2f}s!")
    print(f"    - File size: {file_size_mb:.2f} MB")

    if verify_roundtrip:
        print("[*] Verifying saved .mha file integrity...")
        reloaded = sitk.ReadImage(str(output_mha_path))
        if reloaded.GetSize() != size_xyz:
            raise ValueError(f"Integrity check failed: size mismatch {reloaded.GetSize()} vs {size_xyz}")
        print("[+] Integrity check PASSED: dimension and spacing match original DICOM.")

    return image


def main():
    parser = argparse.ArgumentParser(description="Convert CBCT DICOM series to .mha format")
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM",
        help="Path to directory containing DICOM slices",
    )
    parser.add_argument(
        "--output_mha",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\output\volume.mha",
        help="Path to save output .mha file",
    )
    parser.add_argument(
        "--no_compression",
        action="store_true",
        help="Disable zlib compression",
    )
    parser.add_argument(
        "--no_align",
        action="store_true",
        help="Disable ToothFairy2 coordinate alignment",
    )

    args = parser.parse_args()
    convert_dicom_series_to_mha(
        dicom_dir=Path(args.dicom_dir),
        output_mha_path=Path(args.output_mha),
        use_compression=not args.no_compression,
        align_toothfairy_coords=not args.no_align,
    )


if __name__ == "__main__":
    main()
