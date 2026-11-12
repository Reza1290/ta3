"""Run Inference on Official ToothFairy2 Benchmark Cases & Compare with Ground Truth.

Evaluates Dice score, FDI tooth accuracy, and generates visual validation comparisons.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import SimpleITK as sitk
from predict_toothfairy2 import (
    ARCH_KWARGS,
    TARGET_SPACING,
    load_toothseg_model,
    postprocess_and_extract_teeth,
    resample_image,
    run_sliding_window_inference,
)
from visualize_fdi_results import visualize_fdi_mpr


def evaluate_benchmark_case(
    case_name: str = "ToothFairy2F_001",
    raw_dir: Path = Path(r"D:\Workspace\TA3\data\raw"),
    gt_dir: Path = Path(r"D:\Workspace\TA3\data\toothfairy2\Dataset112_ToothFairy2\labelsTr"),
    checkpoint_path: Path = Path(r"D:\Workspace\TA3\weights\toothseg\checkpoint_best.pth"),
    output_dir: Path = Path(r"D:\Workspace\TA3\data\benchmark_eval"),
    device_str: str = "cuda",
):
    """Runs inference on a benchmark case and computes validation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mha_img_path = raw_dir / f"{case_name}_0000.mha"
    gt_mask_path = gt_dir / f"{case_name}.mha"

    if not mha_img_path.exists():
        raise FileNotFoundError(f"Benchmark raw image not found: {mha_img_path}")

    print("=================================================================")
    print(f"  OFFICIAL TOOTHFAIRY2 BENCHMARK EVALUATION: {case_name}")
    print(f"  Raw Image   : {mha_img_path}")
    print(f"  Ground Truth: {gt_mask_path}")
    print("=================================================================\n")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = load_toothseg_model(checkpoint_path, device)

    itk_img = sitk.ReadImage(str(mha_img_path))
    itk_resampled = resample_image(itk_img, TARGET_SPACING)
    vol_zyx = sitk.GetArrayFromImage(itk_resampled)

    print(f"[*] Running sliding-window inference on {case_name}...")
    t0 = time.time()
    pred_resampled = run_sliding_window_inference(vol_zyx, model, device)
    print(f"[+] Inference completed in {time.time()-t0:.2f}s.")

    cleaned_mask, detected_teeth = postprocess_and_extract_teeth(pred_resampled, TARGET_SPACING)

    # Project back to original grid
    pred_itk = sitk.GetImageFromArray(cleaned_mask)
    pred_itk.SetSpacing(TARGET_SPACING)
    pred_itk.SetOrigin(itk_img.GetOrigin())
    pred_itk.SetDirection(itk_img.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(itk_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    final_pred = resampler.Execute(pred_itk)

    pred_out_path = output_dir / f"{case_name}_pred.mha"
    sitk.WriteImage(final_pred, str(pred_out_path), True)

    # Calculate Dice scores against Ground Truth if available
    if gt_mask_path.exists():
        gt_itk = sitk.ReadImage(str(gt_mask_path))
        gt_arr = sitk.GetArrayFromImage(gt_itk)
        pred_arr = sitk.GetArrayFromImage(final_pred)

        print("\n=================================================================")
        print("  DICE SIMILARITY COEFFICIENT (DSC) VALIDATION MATRIX")
        print("=================================================================")
        dices = []
        for t in detected_teeth:
            fdi = t["fdi"]
            name_id = t["name_id"]
            pred_b = (pred_arr == fdi)
            gt_b = (gt_arr == fdi)
            intersection = np.logical_and(pred_b, gt_b).sum()
            total = pred_b.sum() + gt_b.sum()
            dice = (2.0 * intersection) / max(total, 1) if total > 0 else 1.0
            dices.append(dice)
            print(f"  FDI {fdi:2d} ({name_id}): Dice = {dice:.4f} (GT={gt_b.sum()} vx, Pred={pred_b.sum()} vx)")

        mean_dice = np.mean(dices) if dices else 0.0
        print(f"\n[+] Mean Tooth Dice Score: {mean_dice:.4f} ({len(dices)} evaluated teeth)")

    # Render Visual MPR
    png_path = output_dir / f"{case_name}_mpr_validation.png"
    visualize_fdi_mpr(mha_img_path, pred_out_path, png_path)
    print(f"\n[+] Validation MPR visual saved to: {png_path}")


if __name__ == "__main__":
    import torch
    evaluate_benchmark_case()
