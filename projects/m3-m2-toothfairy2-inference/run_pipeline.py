"""Unified Pipeline: DICOM -> MHA -> ToothFairy2 Model -> FDI Tooth Classification.

Runs the complete clinical analysis workflow for CBCT scans.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from convert_dicom_to_mha import convert_dicom_series_to_mha
from download_models import download_toothseg_dataset121
from predict_toothfairy2 import predict_cbct_fdi
from visualize_fdi_results import visualize_fdi_mpr


def run_pipeline(
    dicom_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    skip_dicom: bool = False,
    device: str = "cuda",
):
    """Executes the full end-to-end dental AI pipeline."""
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("       CLINICAL CBCT FDI CLASSIFICATION & SEGMENTATION")
    print("                   POWERED BY TOOTHFAIRY2")
    print("=================================================================")
    print(f"  DICOM Directory    : {dicom_dir}")
    print(f"  Output Directory   : {output_dir}")
    print(f"  Device             : {device}")
    print("=================================================================\n")

    mha_volume_path = output_dir / "volume.mha"
    pred_mask_path = output_dir / "pred_fdi_mask.mha"
    report_json_path = output_dir / "fdi_clinical_report.json"
    visualization_png_path = output_dir / "fdi_mpr_overview.png"

    # Step 1: DICOM to MHA
    if not skip_dicom or not mha_volume_path.exists():
        print("[STEP 1/4] Converting DICOM series to .mha volume...")
        convert_dicom_series_to_mha(
            dicom_dir=dicom_dir,
            output_mha_path=mha_volume_path,
            use_compression=True,
        )
    else:
        print(f"[STEP 1/4] Using existing MHA volume: {mha_volume_path}")

    # Step 2: Ensure Weights are Ready
    print("\n[STEP 2/4] Verifying ToothFairy2 model checkpoint...")
    if not checkpoint_path.exists():
        print(f"[*] Checkpoint not found at {checkpoint_path}. Downloading...")
        download_toothseg_dataset121(checkpoint_path.parent)
    else:
        print(f"[+] Checkpoint verified: {checkpoint_path} ({checkpoint_path.stat().st_size / 1024**2:.1f} MB)")

    # Step 3: Run Inference & FDI Classification
    print("\n[STEP 3/4] Running 3D deep learning inference & FDI tooth identification...")
    summary = predict_cbct_fdi(
        mha_input_path=mha_volume_path,
        output_pred_mha=pred_mask_path,
        checkpoint_path=checkpoint_path,
        report_json_path=report_json_path,
        device_str=device,
    )

    # Step 4: Render Visual Overlays
    print("\n[STEP 4/4] Rendering Multi-Planar Reconstruction (MPR) overlays...")
    visualize_fdi_mpr(
        cbct_mha_path=mha_volume_path,
        pred_mask_mha_path=pred_mask_path,
        output_png_path=visualization_png_path,
        report_json_path=report_json_path,
    )

    total_time = time.time() - t_start
    print("\n=================================================================")
    print("                  PIPELINE EXECUTION COMPLETED")
    print(f"  Total processing time : {total_time:.2f} seconds")
    print(f"  CBCT Volume MHA       : {mha_volume_path}")
    print(f"  FDI Segmentation Mask : {pred_mask_path}")
    print(f"  Clinical JSON Report  : {report_json_path}")
    print(f"  MPR Visualization PNG : {visualization_png_path}")
    print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Run complete ToothFairy2 CBCT inference pipeline")
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM",
        help="Directory with input DICOM slices",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\output",
        help="Directory to save generated outputs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"D:\Workspace\TA3\weights\toothseg\checkpoint_best.pth",
        help="Path to ToothSeg model checkpoint",
    )
    parser.add_argument(
        "--skip_dicom",
        action="store_true",
        help="Skip DICOM conversion if volume.mha already exists",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device (cuda or cpu)",
    )

    args = parser.parse_args()
    run_pipeline(
        dicom_dir=Path(args.dicom_dir),
        output_dir=Path(args.output_dir),
        checkpoint_path=Path(args.checkpoint),
        skip_dicom=args.skip_dicom,
        device=args.device,
    )


if __name__ == "__main__":
    main()
