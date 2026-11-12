"""Inference and FDI Tooth Classification Engine for ToothFairy2 / CBCT volumes.

Runs volumetric sliding-window inference with Gaussian blending using
PlainConvUNet (ToothSeg / ToothFairy2 benchmark) on CBCT .mha scans.
Outputs 3D segmentation masks, FDI tooth metrics, and M2-M3 clinical relations.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from scipy.ndimage import label as ndi_label

from fdi_taxonomy import (
    FDI_METADATA,
    M2_M3_PAIRS,
    TOOTHSEG_LABEL_TO_FDI,
    get_fdi_info,
    get_quadrant_name,
)

# ToothSeg Model Architecture Definition (nnU-Net v2 3d_fullres)
ARCH_KWARGS = {
    "input_channels": 1,
    "n_stages": 6,
    "features_per_stage": [32, 64, 128, 256, 320, 320],
    "conv_op": torch.nn.Conv3d,
    "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
    "conv_bias": True,
    "norm_op": torch.nn.InstanceNorm3d,
    "norm_op_kwargs": {"eps": 1e-05, "affine": True},
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": torch.nn.LeakyReLU,
    "nonlin_kwargs": {"inplace": True},
    "num_classes": 33,  # 0: Background, 1-32: FDI Teeth
    "deep_supervision": False,
}

PATCH_SIZE = (80, 160, 160)  # (D, H, W)
TARGET_SPACING = (0.3, 0.3, 0.3)  # mm


def build_gaussian_importance_map(patch_size: Tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    """Generate a 3D Gaussian importance map for seamless patch blending."""
    d, h, w = patch_size
    zc = (d - 1) / 2.0
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0

    z = torch.linspace(-zc, zc, steps=d)
    y = torch.linspace(-yc, yc, steps=h)
    x = torch.linspace(-xc, xc, steps=w)

    gz = torch.exp(-0.5 * (z / (zc * sigma_scale * 2.0)) ** 2)
    gy = torch.exp(-0.5 * (y / (yc * sigma_scale * 2.0)) ** 2)
    gx = torch.exp(-0.5 * (x / (xc * sigma_scale * 2.0)) ** 2)

    g = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    g = g / g.max()
    return g.clamp(min=1e-4)


def load_toothseg_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Instantiate PlainConvUNet and load weights from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    print(f"[*] Loading ToothSeg model from: {checkpoint_path}")
    model = PlainConvUNet(**ARCH_KWARGS)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["network_weights"] if "network_weights" in ckpt else ckpt

    # Strip prefixes if necessary
    clean_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace("_orig_mod.", "").replace("module.", "")
        clean_dict[k_clean] = v

    model.load_state_dict(clean_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"[+] ToothSeg model loaded successfully on {device} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params).")
    return model


def resample_image(itk_image: sitk.Image, new_spacing: Tuple[float, float, float]) -> sitk.Image:
    """Resample SimpleITK image to target isotropic spacing."""
    orig_spacing = itk_image.GetSpacing()
    orig_size = itk_image.GetSize()

    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    return resample.Execute(itk_image)


def run_sliding_window_inference(
    volume_zyx: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    overlap: float = 0.5,
) -> np.ndarray:
    """Run sliding-window 3D inference with Gaussian importance weighting."""
    D, H, W = volume_zyx.shape
    pD, pH, pW = PATCH_SIZE
    num_classes = 33

    # Normalize CT volume (ToothFairy2 convention)
    # Clip extreme values and standardize
    v_min, v_max = -1000.0, 3000.0
    norm_vol = np.clip(volume_zyx.astype(np.float32), v_min, v_max)
    # Z-score normalization based on foreground
    fg_mask = norm_vol > -500.0
    if fg_mask.sum() > 0:
        mean, std = norm_vol[fg_mask].mean(), norm_vol[fg_mask].std()
        norm_vol = (norm_vol - mean) / max(std, 1e-5)
    else:
        norm_vol = (norm_vol - norm_vol.mean()) / max(norm_vol.std(), 1e-5)

    # Pad if volume is smaller than patch size
    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        norm_vol = np.pad(norm_vol, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        D, H, W = norm_vol.shape

    step_d = int(pD * (1.0 - overlap))
    step_h = int(pH * (1.0 - overlap))
    step_w = int(pW * (1.0 - overlap))

    z_starts = list(range(0, D - pD + 1, step_d))
    if z_starts[-1] != D - pD:
        z_starts.append(D - pD)

    y_starts = list(range(0, H - pH + 1, step_h))
    if y_starts[-1] != H - pH:
        y_starts.append(H - pH)

    x_starts = list(range(0, W - pW + 1, step_w))
    if x_starts[-1] != W - pW:
        x_starts.append(W - pW)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    print(f"[*] Sliding window: Volume={D}x{H}x{W}, Patch={pD}x{pH}x{pW}, Total patches={total_patches}")

    gaussian = build_gaussian_importance_map(PATCH_SIZE).to(device)

    # Accumulator tensors on GPU or CPU
    acc_probs = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device="cpu")
    acc_weights = torch.zeros((1, D, H, W), dtype=torch.float32, device="cpu")

    t0 = time.time()
    patch_count = 0

    with torch.no_grad():
        for zs in z_starts:
            for ys in y_starts:
                for xs in x_starts:
                    patch = norm_vol[zs:zs+pD, ys:ys+pH, xs:xs+pW]
                    x_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

                    with torch.amp.autocast("cuda"):
                        logits = model(x_t)  # (1, 33, pD, pH, pW)
                        probs = F.softmax(logits, dim=1)  # Softmax probability blending

                    weighted_probs = probs.squeeze(0) * gaussian  # (33, pD, pH, pW)

                    acc_probs[:, zs:zs+pD, ys:ys+pH, xs:xs+pW] += weighted_probs.cpu()
                    acc_weights[:, zs:zs+pD, ys:ys+pH, xs:xs+pW] += gaussian.cpu()

                    patch_count += 1
                    if patch_count % 10 == 0 or patch_count == total_patches:
                        speed = patch_count / max(time.time() - t0, 0.01)
                        print(f"    Inference progress: {patch_count}/{total_patches} patches ({speed:.1f} patch/s)", end="\r")

    print(f"\n[+] Sliding window complete in {time.time()-t0:.2f}s.")
    acc_weights = acc_weights.clamp(min=1e-5)
    acc_probs /= acc_weights

    # Slice back to original dimensions if padded
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        acc_probs = acc_probs[:, :D-pad_d, :H-pad_h, :W-pad_w]

    pred_mask = torch.argmax(acc_probs, dim=0).numpy().astype(np.uint8)
    return pred_mask


def postprocess_and_extract_teeth(
    pred_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    min_volume_mm3: float = 30.0,
) -> Tuple[np.ndarray, List[Dict]]:
    """Clean mask with connected components and calculate clinical FDI statistics."""
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    min_voxels = int(round(min_volume_mm3 / voxel_vol_mm3))

    cleaned_mask = np.zeros_like(pred_mask)
    detected_teeth = []

    unique_labels = np.unique(pred_mask)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        fdi_code = TOOTHSEG_LABEL_TO_FDI.get(int(lbl), None)
        if fdi_code is None:
            continue

        binary = (pred_mask == lbl)
        labeled_cc, num_cc = ndi_label(binary)
        if num_cc == 0:
            continue

        # Keep largest connected component
        cc_sizes = np.bincount(labeled_cc.flat)[1:]
        largest_cc_idx = np.argmax(cc_sizes) + 1
        largest_size = cc_sizes[largest_cc_idx - 1]

        if largest_size < min_voxels:
            continue

        tooth_voxels = (labeled_cc == largest_cc_idx)
        cleaned_mask[tooth_voxels] = fdi_code

        # Spatial coordinates (Z, Y, X)
        coords = np.argwhere(tooth_voxels)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        centroid_zyx = coords.mean(axis=0)

        # Centroid in physical mm
        centroid_mm = [
            round(float(centroid_zyx[0] * spacing[0]), 2),
            round(float(centroid_zyx[1] * spacing[1]), 2),
            round(float(centroid_zyx[2] * spacing[2]), 2),
        ]

        info = get_fdi_info(fdi_code)
        volume_mm3 = round(float(largest_size * voxel_vol_mm3), 2)

        detected_teeth.append({
            "fdi": fdi_code,
            "quadrant": info.quadrant if info else fdi_code // 10,
            "name_id": info.name_id if info else f"FDI {fdi_code}",
            "name_en": info.name_en if info else f"FDI {fdi_code}",
            "is_m2": info.is_m2 if info else False,
            "is_m3": info.is_m3 if info else False,
            "volume_mm3": volume_mm3,
            "voxel_count": int(largest_size),
            "centroid_zyx": [round(float(c), 1) for c in centroid_zyx],
            "centroid_mm": centroid_mm,
            "bbox_zyx": [int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)],
        })

    # Sort teeth by FDI code
    detected_teeth.sort(key=lambda t: t["fdi"])
    return cleaned_mask, detected_teeth


def summarize_clinical_report(detected_teeth: List[Dict]) -> Dict:
    """Generate structured clinical summary for M2-M3 molar relations."""
    detected_fdis = {t["fdi"]: t for t in detected_teeth}
    quad_reports = {}

    for quad_name, (m2_code, m3_code) in M2_M3_PAIRS.items():
        m2_found = m2_code in detected_fdis
        m3_found = m3_code in detected_fdis

        status = "None"
        if m2_found and m3_found:
            status = "Eligible M2-M3 Pair (Processable)"
        elif m2_found and not m3_found:
            status = "M2 Present, M3 Agenesis/Extracted"
        elif not m2_found and m3_found:
            status = "M3 Present, M2 Missing"
        else:
            status = "Both Molars Absent"

        quad_reports[quad_name] = {
            "m2_fdi": m2_code,
            "m2_detected": m2_found,
            "m2_volume_mm3": detected_fdis[m2_code]["volume_mm3"] if m2_found else None,
            "m3_fdi": m3_code,
            "m3_detected": m3_found,
            "m3_volume_mm3": detected_fdis[m3_code]["volume_mm3"] if m3_found else None,
            "status": status,
        }

    return {
        "total_teeth_detected": len(detected_teeth),
        "detected_fdi_list": list(detected_fdis.keys()),
        "molar_pair_relations": quad_reports,
    }


def predict_cbct_fdi(
    mha_input_path: Path,
    output_pred_mha: Path,
    checkpoint_path: Path,
    report_json_path: Optional[Path] = None,
    device_str: str = "cuda",
) -> Dict:
    """Full pipeline: Load MHA -> Resample -> Inference -> FDI Clean -> Save."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("  ToothFairy2 / FDI Tooth Segmentation & Classification Pipeline")
    print(f"  Input volume : {mha_input_path}")
    print(f"  Output mask  : {output_pred_mha}")
    print(f"  Device       : {device}")
    print("=================================================================\n")

    # 1. Load MHA Volume
    print(f"[*] Reading input volume from {mha_input_path}...")
    itk_orig = sitk.ReadImage(str(mha_input_path))
    orig_spacing = itk_orig.GetSpacing()
    orig_size = itk_orig.GetSize()
    print(f"[+] Original size (XYZ): {orig_size}, spacing: {orig_spacing}")

    # 2. Resample to 0.3mm isotropic spacing for ToothSeg
    print(f"[*] Resampling to {TARGET_SPACING} mm isotropic spacing...")
    itk_resampled = resample_image(itk_orig, TARGET_SPACING)
    vol_zyx = sitk.GetArrayFromImage(itk_resampled)
    print(f"[+] Resampled size (ZYX): {vol_zyx.shape}")

    # 3. Load Model
    model = load_toothseg_model(checkpoint_path, device)

    # 4. Run Sliding-Window Inference
    pred_mask_resampled = run_sliding_window_inference(vol_zyx, model, device)

    # 5. Extract and Clean Teeth
    print("[*] Filtering connected components and mapping FDI codes...")
    cleaned_mask_resampled, detected_teeth = postprocess_and_extract_teeth(
        pred_mask_resampled, TARGET_SPACING
    )

    # 6. Resample mask back to original resolution
    print("[*] Projecting predicted mask back to original input grid...")
    pred_itk_resampled = sitk.GetImageFromArray(cleaned_mask_resampled)
    pred_itk_resampled.SetSpacing(TARGET_SPACING)
    pred_itk_resampled.SetOrigin(itk_orig.GetOrigin())
    pred_itk_resampled.SetDirection(itk_orig.GetDirection())

    resampler_back = sitk.ResampleImageFilter()
    resampler_back.SetReferenceImage(itk_orig)
    resampler_back.SetInterpolator(sitk.sitkNearestNeighbor)
    final_pred_itk = resampler_back.Execute(pred_itk_resampled)

    # 7. Save output predicted mask MHA
    output_pred_mha.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving predicted FDI segmentation mask to {output_pred_mha}...")
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(output_pred_mha))
    writer.SetUseCompression(True)
    writer.Execute(final_pred_itk)
    print(f"[+] Output mask saved successfully ({output_pred_mha.stat().st_size / 1024**2:.1f} MB).")

    # 8. Clinical Summary
    summary = summarize_clinical_report(detected_teeth)
    summary["detected_teeth"] = detected_teeth
    summary["input_volume"] = str(mha_input_path)
    summary["output_pred_mask"] = str(output_pred_mha)

    print("\n=================================================================")
    print("  CLINICAL TOOTH INVENTORY & M2-M3 RELATION REPORT")
    print("=================================================================")
    print(f"  Total permanent teeth identified : {summary['total_teeth_detected']} / 32")
    print(f"  Identified FDI list              : {summary['detected_fdi_list']}")
    print("\n  Quadrant M2-M3 Relations:")
    for quad, q_data in summary["molar_pair_relations"].items():
        print(f"    - {quad:24s}: {q_data['status']}")
        if q_data["m2_detected"]:
            print(f"        M2 (FDI {q_data['m2_fdi']}): {q_data['m2_volume_mm3']} mm3")
        if q_data["m3_detected"]:
            print(f"        M3 (FDI {q_data['m3_fdi']}): {q_data['m3_volume_mm3']} mm3")

    if report_json_path:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[+] Saved detailed clinical JSON report to: {report_json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Segment and classify teeth into FDI labels using ToothFairy2")
    parser.add_argument(
        "--input_mha",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA.mha",
        help="Input .mha CBCT volume path",
    )
    parser.add_argument(
        "--output_mha",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA_pred_fdi.mha",
        help="Output .mha predicted mask path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"D:\Workspace\TA3\weights\toothseg\checkpoint_best.pth",
        help="ToothSeg model checkpoint path",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA_fdi_report.json",
        help="Output JSON summary report path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (cuda or cpu)",
    )

    args = parser.parse_args()
    predict_cbct_fdi(
        mha_input_path=Path(args.input_mha),
        output_pred_mha=Path(args.output_mha),
        checkpoint_path=Path(args.checkpoint),
        report_json_path=Path(args.report_json) if args.report_json else None,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
