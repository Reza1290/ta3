"""Visualization Engine for FDI Tooth Predictions & Multi-Planar Reconstruction (MPR).

Generates side-by-side CBCT slice views with colored FDI tooth contours,
centroids, and clinical M2-M3 relation highlights.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap

from fdi_taxonomy import (
    FDI_METADATA,
    M2_M3_PAIRS,
    TOOTHSEG_LABEL_TO_FDI,
    get_fdi_info,
)


def get_distinct_fdi_colormap() -> Tuple[ListedColormap, Dict[int, Tuple[float, float, float]]]:
    """Generate high-contrast color palette for all 32 teeth."""
    cmap_base = plt.get_cmap("tab20")
    colors = {}
    # Use distinct hues for each quadrant
    for fdi in FDI_METADATA.keys():
        quad = fdi // 10
        t_num = fdi % 10
        # Color variations
        idx = ((quad - 1) * 8 + (t_num - 1)) % 20
        colors[fdi] = cmap_base(idx)
    return colors


def visualize_fdi_mpr(
    cbct_mha_path: Path,
    pred_mask_mha_path: Path,
    output_png_path: Path,
    report_json_path: Optional[Path] = None,
):
    """Generate a multi-view visual overview of FDI tooth segmentation."""
    print("=================================================================")
    print("  Generating FDI Multi-Planar Reconstruction (MPR) Overlays")
    print(f"  CBCT volume : {cbct_mha_path}")
    print(f"  Mask volume : {pred_mask_mha_path}")
    print(f"  Output PNG  : {output_png_path}")
    print("=================================================================\n")

    itk_img = sitk.ReadImage(str(cbct_mha_path))
    itk_mask = sitk.ReadImage(str(pred_mask_mha_path))

    vol = sitk.GetArrayFromImage(itk_img)  # (Z, Y, X)
    mask = sitk.GetArrayFromImage(itk_mask)  # (Z, Y, X)

    # Intensity windowing for dental CBCT bone view
    # Window: 300 to 2500 HU (teeth and cortical bone)
    vmin, vmax = 200, 2200
    vol_disp = np.clip(vol, vmin, vmax)
    vol_disp = (vol_disp - vmin) / (vmax - vmin)

    # Find center of dental arch from mask
    teeth_voxels = np.argwhere(mask > 0)
    if len(teeth_voxels) > 0:
        z_mid, y_mid, x_mid = np.median(teeth_voxels, axis=0).astype(int)
    else:
        z_mid, y_mid, x_mid = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2

    # Create figure with 4 subplots:
    # 1. Axial view (arch overview)
    # 2. Coronal view (anterior/posterior teeth)
    # 3. Sagittal view (Right M2-M3)
    # 4. Sagittal view (Left M2-M3)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor="#1a1a1a")

    color_map = get_distinct_fdi_colormap()

    def overlay_slice(ax, slice_img, slice_mask, title):
        ax.imshow(slice_img, cmap="gray", origin="lower")
        unique_teeth = np.unique(slice_mask)
        for t_fdi in unique_teeth:
            if t_fdi == 0:
                continue
            t_binary = (slice_mask == t_fdi)
            color = color_map.get(int(t_fdi), (1.0, 0.2, 0.2, 1.0))
            ax.contour(t_binary, levels=[0.5], colors=[color], linewidths=1.5, origin="lower")

            # Label centroid
            pts = np.argwhere(t_binary)
            if len(pts) > 20:
                cy, cx = pts.mean(axis=0)
                ax.text(cx, cy, str(int(t_fdi)), color="yellow", fontsize=9, fontweight="bold",
                        ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="yellow", alpha=0.6))

        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")

    # 1. Axial slice
    overlay_slice(axes[0, 0], vol_disp[z_mid, :, :], mask[z_mid, :, :], f"Axial Slice (Z={z_mid}) - Dental Arch")

    # 2. Coronal slice
    overlay_slice(axes[0, 1], vol_disp[:, y_mid, :], mask[:, y_mid, :], f"Coronal Slice (Y={y_mid}) - Maxilla/Mandible")

    # Find M2-M3 sagittal coordinates
    # Right Molars (FDI 47, 48 or 17, 18)
    r_molars = np.argwhere(np.isin(mask, [17, 18, 47, 48]))
    x_r = int(np.median(r_molars[:, 2])) if len(r_molars) > 0 else x_mid - 80

    # Left Molars (FDI 27, 28 or 37, 38)
    l_molars = np.argwhere(np.isin(mask, [27, 28, 37, 38]))
    x_l = int(np.median(l_molars[:, 2])) if len(l_molars) > 0 else x_mid + 80

    # 3. Right Molar Sagittal
    overlay_slice(axes[1, 0], vol_disp[:, :, x_r], mask[:, :, x_r], f"Right Molars Sagittal (X={x_r}) - M2/M3 Q1 & Q4")

    # 4. Left Molar Sagittal
    overlay_slice(axes[1, 1], vol_disp[:, :, x_l], mask[:, :, x_l], f"Left Molars Sagittal (X={x_l}) - M2/M3 Q2 & Q3")

    # Add subtitle & metadata
    fig.suptitle("ToothFairy2 / CBCT FDI Tooth Classification & Segmentation\nPatient: PRATIWI SISWI DIANA (38 Thn)",
                 color="white", fontsize=15, y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print(f"[+] Multi-planar reconstruction visualization saved to: {output_png_path}")


def main():
    parser = argparse.ArgumentParser(description="Render FDI tooth segmentation overlays on CBCT slices")
    parser.add_argument(
        "--cbct_mha",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA.mha",
        help="Input CBCT .mha path",
    )
    parser.add_argument(
        "--mask_mha",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA_pred_fdi.mha",
        help="Predicted FDI mask .mha path",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA_fdi_mpr.png",
        help="Output PNG visualization path",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=r"D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA_fdi_report.json",
        help="Optional clinical report JSON",
    )

    args = parser.parse_args()
    visualize_fdi_mpr(
        cbct_mha_path=Path(args.cbct_mha),
        pred_mask_mha_path=Path(args.mask_mha),
        output_png_path=Path(args.output_png),
        report_json_path=Path(args.report_json) if args.report_json else None,
    )


if __name__ == "__main__":
    main()
