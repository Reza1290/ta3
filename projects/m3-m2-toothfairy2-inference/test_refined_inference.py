"""Test Refined Sliding Window Inference with Softmax Blending and Overlap=0.6.

Tests whether proper probability accumulation and increased overlap resolves
the tooth 26/27 split and missed tooth 27.
"""

from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import label as ndi_label

from fdi_taxonomy import FDI_METADATA, TOOTHSEG_LABEL_TO_FDI, FDI_TO_TOOTHSEG_LABEL
from predict_toothfairy2 import (
    load_toothseg_model,
    resample_image,
    build_gaussian_importance_map,
    PATCH_SIZE,
    TARGET_SPACING,
)


def run_refined_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_toothseg_model(r"D:\Workspace\TA3\weights\toothseg\checkpoint_best.pth", device)

    img_itk = sitk.ReadImage(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
    print("[*] Resampling to target spacing (0.3mm)...")
    resampled = resample_image(img_itk, TARGET_SPACING)
    vol_zyx = sitk.GetArrayFromImage(resampled)  # (D, H, W)
    print(f"[*] Resampled volume shape: {vol_zyx.shape}")

    D, H, W = vol_zyx.shape
    pD, pH, pW = PATCH_SIZE
    num_classes = 33
    overlap = 0.6  # 60% overlap for seamless patches

    # Normalization (standard CT/CBCT Z-score on foreground)
    norm_vol = np.clip(vol_zyx.astype(np.float32), -1000.0, 3000.0)
    fg_mask = norm_vol > -500.0
    mean, std = norm_vol[fg_mask].mean(), norm_vol[fg_mask].std()
    norm_vol = ((norm_vol - mean) / max(std, 1e-5)).astype(np.float32)

    # Pad if necessary
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
    print(f"[*] Sliding window: Volume={D}x{H}x{W}, Patch={pD}x{pH}x{pW}, Overlap={overlap}, Total patches={total_patches}")

    gaussian = build_gaussian_importance_map(PATCH_SIZE).to(device)

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
                        logits = model(x_t)
                        # CRUCIAL FIX: Apply softmax before Gaussian accumulation
                        probs = F.softmax(logits, dim=1)

                    weighted_probs = probs.squeeze(0) * gaussian  # (33, pD, pH, pW)

                    acc_probs[:, zs:zs+pD, ys:ys+pH, xs:xs+pW] += weighted_probs.cpu()
                    acc_weights[:, zs:zs+pD, ys:ys+pH, xs:xs+pW] += gaussian.cpu()

                    patch_count += 1
                    if patch_count % 20 == 0 or patch_count == total_patches:
                        speed = patch_count / max(time.time() - t0, 0.01)
                        print(f"    Inference: {patch_count}/{total_patches} patches ({speed:.1f} p/s)")

    print(f"\n[+] Inference finished in {time.time()-t0:.2f}s.")
    acc_weights = acc_weights.clamp(min=1e-5)
    acc_probs /= acc_weights

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        acc_probs = acc_probs[:, :D-pad_d, :H-pad_h, :W-pad_w]

    # Raw argmax mask (before connected component removal)
    raw_pred_mask = torch.argmax(acc_probs, dim=0).numpy().astype(np.uint8)

    # Convert ToothSeg classes (1..32) to FDI codes (11..48)
    fdi_mask_resampled = np.zeros_like(raw_pred_mask)
    for seg_idx, fdi_code in TOOTHSEG_LABEL_TO_FDI.items():
        fdi_mask_resampled[raw_pred_mask == seg_idx] = fdi_code

    # Inspect Kuadran 2 molars
    print("\n--- Kuadran 2 Tooth Stats in Raw Refined Prediction ---")
    for fdi in [24, 25, 26, 27, 28]:
        b = (fdi_mask_resampled == fdi)
        print(f"FDI {fdi}: {b.sum():6d} voxels", end="")
        if b.sum() > 0:
            pts = np.argwhere(b)
            cz, cy, cx = pts.mean(axis=0)
            print(f" | Centroid Z={cz:.1f}, Y={cy:.1f}, X={cx:.1f} | Y range: {pts[:,1].min()}..{pts[:,1].max()}")
        else:
            print(" | NOT FOUND")

    # Save resampled refined mask for testing
    out_mha = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_refined_resampled.mha")
    res_itk = sitk.GetImageFromArray(fdi_mask_resampled)
    res_itk.SetSpacing(TARGET_SPACING)
    sitk.WriteImage(res_itk, str(out_mha), True)
    print(f"\n[+] Saved refined resampled mask to: {out_mha}")


if __name__ == "__main__":
    run_refined_inference()
