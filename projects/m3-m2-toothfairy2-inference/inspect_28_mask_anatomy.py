"""Inspect Tooth 28 Mask Accuracy against Raw CBCT Anatomy for Patient Pratiwi.

Generates a high-resolution multi-slice sagittal montage zooming into FDI 28 and FDI 27
to pinpoint exactly where the mask under-segments, over-segments, or misaligns.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def inspect_28_slices():
    vol_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
    mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask_anatomical_refined.mha")
    
    vol_itk = sitk.ReadImage(str(vol_path))
    mask_itk = sitk.ReadImage(str(mask_path))
    
    vol = sitk.GetArrayFromImage(vol_itk)  # (Z, Y, X)
    mask = sitk.GetArrayFromImage(mask_itk)
    
    # Bounding box of 28: X: 168..246, Y: 335..434, Z: 130..190
    # Let's inspect sagittal cuts across X: 185, 195, 205, 215, 225
    x_cuts = [185, 195, 205, 215, 225]
    
    # Bone window
    vmin, vmax = -200, 1500
    
    fig, axes = plt.subplots(len(x_cuts), 3, figsize=(16, 20), dpi=200)
    
    for row_idx, x_val in enumerate(x_cuts):
        # Sagittal slice: vol[:, :, x_val]
        # Crop region of interest around Q2 molars: Z in [100, 310], Y in [300, 470]
        z_s, z_e = 100, 310
        y_s, y_e = 300, 470
        
        slice_cbct = vol[z_s:z_e, y_s:y_e, x_val]
        slice_mask = mask[z_s:z_e, y_s:y_e, x_val]
        
        # Colormap for teeth
        # 26: Cyan, 27: Green, 28: Red, others: Yellow
        cmap_teeth = mcolors.ListedColormap(["none", "#00ffff", "#00ff00", "#ff0000", "#ffff00"])
        
        mask_viz = np.zeros_like(slice_mask, dtype=int)
        mask_viz[slice_mask == 26] = 1
        mask_viz[slice_mask == 27] = 2
        mask_viz[slice_mask == 28] = 3
        
        # Column 1: Raw CBCT
        ax1 = axes[row_idx, 0]
        ax1.imshow(slice_cbct, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax1.set_title(f"Sagital X={x_val}: Raw CBCT", fontsize=10, fontweight="bold")
        ax1.axis("off")
        
        # Column 2: Raw CBCT + Mask Overlay
        ax2 = axes[row_idx, 1]
        ax2.imshow(slice_cbct, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        m_show = np.ma.masked_where(mask_viz == 0, mask_viz)
        ax2.imshow(m_show, cmap=cmap_teeth, alpha=0.5, origin="lower", interpolation="nearest")
        ax2.set_title(f"Overlay (Hijau=27, Merah=28, Biru=26)", fontsize=10, fontweight="bold")
        ax2.axis("off")
        
        # Column 3: Contour Outlines
        ax3 = axes[row_idx, 2]
        ax3.imshow(slice_cbct, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        # Draw contour of 28
        b28 = (mask_viz == 3).astype(float)
        b27 = (mask_viz == 2).astype(float)
        if b28.sum() > 0:
            ax3.contour(b28, levels=[0.5], colors=["red"], linewidths=2.0)
        if b27.sum() > 0:
            ax3.contour(b27, levels=[0.5], colors=["lime"], linewidths=1.5)
        ax3.set_title("Kontur Batas Mask (Merah: 28, Hijau: 27)", fontsize=10, fontweight="bold")
        ax3.axis("off")
        
    plt.suptitle(
        "AUDIT DETIL MORFOLOGI GIGI 28 & 27 (PATIENT PRATIWI)\n"
        "Menampilkan Irisan Sagital Melintang dari Lateral ke Medial (X=185 hingga X=225)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    out_png = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\audit_tooth28_slices.png")
    plt.savefig(str(out_png), dpi=200)
    plt.close()
    print(f"[+] Saved slice audit to: {out_png}")


if __name__ == "__main__":
    inspect_28_slices()
