"""Generate Ultra-Detailed MPR Tri-Planar View of Tooth 28 in Patient Pratiwi.

Shows Axial, Coronal, and Sagittal cross-sections passing through the true centroid
of FDI 28 (Z=163, Y=383, X=207) with and without segmentation overlays.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def zoom_tooth_28():
    vol_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
    raw_mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")
    ref_mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask_anatomical_refined.mha")
    
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(vol_path)))
    m_raw = sitk.GetArrayFromImage(sitk.ReadImage(str(raw_mask_path)))
    m_ref = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_mask_path)))
    
    cz, cy, cx = 163, 383, 207
    
    # Bone window
    vmin, vmax = -200, 1400
    
    # Bounding boxes for zooming
    # Z: [100, 230], Y: [320, 450], X: [150, 260]
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), dpi=200)
    
    # Row 1: Sagittal Cut (X=207)
    # Raw CBCT
    ax = axes[0, 0]
    ax.imshow(vol[100:230, 320:450, cx], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(f"Sagital X={cx}: Raw CBCT", fontsize=11, fontweight="bold")
    ax.set_ylabel("Superior -> Inferior (Z)", fontsize=9)
    ax.set_xlabel("Anterior -> Posterior (Y)", fontsize=9)
    
    # Raw Model Mask
    ax = axes[0, 1]
    ax.imshow(vol[100:230, 320:450, cx], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    m1 = m_raw[100:230, 320:450, cx]
    m1_show = np.ma.masked_where(m1 == 0, m1)
    ax.imshow(m1_show, cmap="Set1", alpha=0.55, origin="lower", interpolation="nearest")
    ax.set_title(f"Sagital X={cx}: Raw ToothFairy2 Mask", fontsize=11, fontweight="bold", color="blue")
    
    # Refined Mask
    ax = axes[0, 2]
    ax.imshow(vol[100:230, 320:450, cx], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    m2 = m_ref[100:230, 320:450, cx]
    m2_show = np.ma.masked_where(m2 == 0, m2)
    ax.imshow(m2_show, cmap="Set1", alpha=0.55, origin="lower", interpolation="nearest")
    ax.set_title(f"Sagital X={cx}: Refined Mask (Dilated)", fontsize=11, fontweight="bold", color="purple")
    
    # Row 2: Coronal Cut (Y=383)
    ax = axes[1, 0]
    ax.imshow(vol[100:230, cy, 150:260], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(f"Koronal Y={cy}: Raw CBCT", fontsize=11, fontweight="bold")
    ax.set_ylabel("Superior -> Inferior (Z)", fontsize=9)
    ax.set_xlabel("Lateral -> Medial (X)", fontsize=9)
    
    ax = axes[1, 1]
    ax.imshow(vol[100:230, cy, 150:260], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    m1_cor = m_raw[100:230, cy, 150:260]
    ax.imshow(np.ma.masked_where(m1_cor == 0, m1_cor), cmap="Set1", alpha=0.55, origin="lower", interpolation="nearest")
    ax.set_title(f"Koronal Y={cy}: Raw ToothFairy2 Mask", fontsize=11, fontweight="bold", color="blue")
    
    ax = axes[1, 2]
    ax.imshow(vol[100:230, cy, 150:260], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    m2_cor = m_ref[100:230, cy, 150:260]
    ax.imshow(np.ma.masked_where(m2_cor == 0, m2_cor), cmap="Set1", alpha=0.55, origin="lower", interpolation="nearest")
    ax.set_title(f"Koronal Y={cy}: Refined Mask", fontsize=11, fontweight="bold", color="purple")
    
    # Row 3: Axial Cut (Z=163)
    ax = axes[2, 0]
    ax.imshow(vol[cz, 320:450, 150:260], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"Aksial Z={cz}: Raw CBCT", fontsize=11, fontweight="bold")
    ax.set_ylabel("Anterior -> Posterior (Y)", fontsize=9)
    ax.set_xlabel("Lateral -> Medial (X)", fontsize=9)
    
    ax = axes[2, 1]
    ax.imshow(vol[cz, 320:450, 150:260], cmap="gray", vmin=vmin, vmax=vmax)
    m1_ax = m_raw[cz, 320:450, 150:260]
    ax.imshow(np.ma.masked_where(m1_ax == 0, m1_ax), cmap="Set1", alpha=0.55, interpolation="nearest")
    ax.set_title(f"Aksial Z={cz}: Raw ToothFairy2 Mask", fontsize=11, fontweight="bold", color="blue")
    
    ax = axes[2, 2]
    ax.imshow(vol[cz, 320:450, 150:260], cmap="gray", vmin=vmin, vmax=vmax)
    m2_ax = m_ref[cz, 320:450, 150:260]
    ax.imshow(np.ma.masked_where(m2_ax == 0, m2_ax), cmap="Set1", alpha=0.55, interpolation="nearest")
    ax.set_title(f"Aksial Z={cz}: Refined Mask", fontsize=11, fontweight="bold", color="purple")
    
    plt.suptitle(
        "ANALISIS ANATOMI & PERBANDINGAN MASK GIGI 28 (PRATIWI)\n"
        "Menampilkan Irisan 3D Tepat pada Titik Pusat FDI 28 (Z=163, Y=383, X=207)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_png = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\zoom_tooth_28_triplanar.png")
    plt.savefig(str(out_png), dpi=200)
    plt.close()
    print(f"[+] Triplanar view saved to: {out_png}")


if __name__ == "__main__":
    zoom_tooth_28()
