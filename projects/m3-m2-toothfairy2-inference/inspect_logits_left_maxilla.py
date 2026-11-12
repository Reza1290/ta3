"""Inspect Raw Model Probabilities and Logits at Left Maxilla Molar Region.

Checks whether the model predicts Tooth 27 (FDI 27) with high logit/probability
under different normalization schemes (CTNormalization vs ZScoreNormalization)
and with test-time augmentation (TTA).
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from fdi_taxonomy import TOOTHSEG_LABEL_TO_FDI, FDI_TO_TOOTHSEG_LABEL
from predict_toothfairy2 import (
    load_toothseg_model,
    resample_image,
    build_gaussian_importance_map,
    PATCH_SIZE,
    TARGET_SPACING,
)


def inspect_molar_logits():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_toothseg_model(r"D:\Workspace\TA3\weights\toothseg\checkpoint_best.pth", device)

    img_itk = sitk.ReadImage(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
    resampled = resample_image(img_itk, TARGET_SPACING)
    vol = sitk.GetArrayFromImage(resampled)  # (D, H, W)
    print("Resampled volume shape:", vol.shape)

    # Let's test two normalizations:
    # Scheme A: Current ZScore (clipping -1000..3000, foreground mean/std)
    # Scheme B: Official CTNormalization (clipping -113.8..4021.0, fixed mean=1921.52, std=686.88)
    # Scheme C: Whole volume Z-Score (nnUNet ZScoreNormalization with use_mask_for_norm=False)
    
    schemes = {
        "A_current": lambda v: (np.clip(v, -1000, 3000) - np.mean(v[v > -500])) / np.std(v[v > -500]),
        "B_ctnorm": lambda v: (np.clip(v, -113.796, 4021.0) - 1921.52) / 686.88,
        "C_zscore": lambda v: (v - np.mean(v)) / np.std(v),
    }

    # Focus patch containing Left Maxilla Molars (FDI 25, 26, 27, 28):
    # In 0.3mm grid:
    # Maxilla Z is ~ 100..200 (Patch: 80 slices -> 90..170)
    # Left Y is ~ 120..280 (Patch: 160 slices -> 120..280)
    # Left X is ~ 60..220 (Patch: 160 slices -> 70..230)
    
    # Let's find exact bounding box of Kuadran 2 in resampled grid
    pD, pH, pW = PATCH_SIZE
    # We can run full sliding window for each scheme with step overlap 0.5
    # or compare the patch directly
    for s_name, norm_fn in schemes.items():
        print(f"\n================ Testing Scheme {s_name} ================")
        norm_v = norm_fn(vol.astype(np.float32))
        print(f"Norm range: min={norm_v.min():.2f}, max={norm_v.max():.2f}, mean={norm_v.mean():.2f}")

        # Extract patch around left maxilla molars
        zs, ys, xs = 90, 120, 80
        patch = norm_v[zs:zs+pD, ys:ys+pH, xs:xs+pW]
        x_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits = model(x_t)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (33, pD, pH, pW)

        # Check max prob and detected volume for FDI 26 (label 14), 27 (label 15), 28 (label 16)
        l26 = FDI_TO_TOOTHSEG_LABEL[26]  # 14
        l27 = FDI_TO_TOOTHSEG_LABEL[27]  # 15
        l28 = FDI_TO_TOOTHSEG_LABEL[28]  # 16

        argmax_p = np.argmax(probs, axis=0)
        v26 = (argmax_p == l26).sum()
        v27 = (argmax_p == l27).sum()
        v28 = (argmax_p == l28).sum()

        print(f"  FDI 26 (M1) voxels in patch: {v26:6d} | Max prob: {probs[l26].max():.4f}")
        print(f"  FDI 27 (M2) voxels in patch: {v27:6d} | Max prob: {probs[l27].max():.4f}")
        print(f"  FDI 28 (M3) voxels in patch: {v28:6d} | Max prob: {probs[l28].max():.4f}")


if __name__ == "__main__":
    inspect_molar_logits()
