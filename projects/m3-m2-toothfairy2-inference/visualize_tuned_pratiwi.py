"""Generate Before vs After Visual Comparison for Tooth 28 and 27 Tuning in Patient Pratiwi.

Shows how the false leakage and overlap under Tooth 28 were eliminated,
leaving two anatomically clean, separated teeth (FDI 27 vertical, FDI 28 horizontal).
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def compare_tuned_molars():
    vol_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
    old_mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask_anatomical_refined.mha")
    new_mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask_clinically_tuned.mha")
    
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(vol_path)))
    m_old = sitk.GetArrayFromImage(sitk.ReadImage(str(old_mask_path)))
    m_new = sitk.GetArrayFromImage(sitk.ReadImage(str(new_mask_path)))
    
    # Bone window
    vmin, vmax = -200, 1400
    
    # Custom colormap:
    # 26: Cyan (#00d2ff), 27: Green (#00ff44), 28: Red/Orange (#ff3b30)
    cmap = mcolors.ListedColormap(["none", "#00d2ff", "#00ff44", "#ff3b30"])
    
    def prep_viz(m_slice):
        v = np.zeros_like(m_slice, dtype=int)
        v[m_slice == 26] = 1
        v[m_slice == 27] = 2
        v[m_slice == 28] = 3
        return np.ma.masked_where(v == 0, v)
        
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), dpi=200)
    
    # 1. Sagittal Cut X=207 (Through Tooth 28)
    x1 = 207
    z_s, z_e = 110, 270
    y_s, y_e = 320, 440
    
    # Raw CBCT
    axes[0, 0].imshow(vol[z_s:z_e, y_s:y_e, x1], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, 0].set_title(f"Sagital X={x1} (Melalui Gigi 28): Raw CBCT", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Superior -> Inferior (Z)", fontsize=9)
    axes[0, 0].set_xlabel("Anterior -> Posterior (Y)", fontsize=9)
    
    # Old Mask
    axes[0, 1].imshow(vol[z_s:z_e, y_s:y_e, x1], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, 1].imshow(prep_viz(m_old[z_s:z_e, y_s:y_e, x1]), cmap=cmap, alpha=0.55, origin="lower", interpolation="nearest")
    axes[0, 1].set_title("Sebelum Tuning: Ada Kebocoran Label 27 di Bawah 28", fontsize=11, fontweight="bold", color="darkred")
    
    # Tuned Mask
    axes[0, 2].imshow(vol[z_s:z_e, y_s:y_e, x1], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0, 2].imshow(prep_viz(m_new[z_s:z_e, y_s:y_e, x1]), cmap=cmap, alpha=0.55, origin="lower", interpolation="nearest")
    axes[0, 2].set_title("Setelah Tuning: Gigi 28 Tidur Utuh & Bersih", fontsize=11, fontweight="bold", color="darkgreen")
    
    # 2. Sagittal Cut X=198 (Melalui Gigi 27 M2)
    x2 = 198
    axes[1, 0].imshow(vol[z_s:z_e, y_s:y_e, x2], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1, 0].set_title(f"Sagital X={x2} (Melalui Gigi 27): Raw CBCT", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Superior -> Inferior (Z)", fontsize=9)
    axes[1, 0].set_xlabel("Anterior -> Posterior (Y)", fontsize=9)
    
    axes[1, 1].imshow(vol[z_s:z_e, y_s:y_e, x2], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1, 1].imshow(prep_viz(m_old[z_s:z_e, y_s:y_e, x2]), cmap=cmap, alpha=0.55, origin="lower", interpolation="nearest")
    axes[1, 1].set_title("Sebelum Tuning: Gigi 27 Melebar ke Posterior", fontsize=11, fontweight="bold", color="darkred")
    
    axes[1, 2].imshow(vol[z_s:z_e, y_s:y_e, x2], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1, 2].imshow(prep_viz(m_new[z_s:z_e, y_s:y_e, x2]), cmap=cmap, alpha=0.55, origin="lower", interpolation="nearest")
    axes[1, 2].set_title("Setelah Tuning: Gigi 27 Berdiri Tegak & Terpisah", fontsize=11, fontweight="bold", color="darkgreen")
    
    # 3. Coronal Cut Y=375 (Melalui Bidang Kontak)
    y3 = 375
    axes[2, 0].imshow(vol[z_s:z_e, y3, 160:250], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[2, 0].set_title(f"Koronal Y={y3}: Raw CBCT", fontsize=11, fontweight="bold")
    axes[2, 0].set_ylabel("Superior -> Inferior (Z)", fontsize=9)
    axes[2, 0].set_xlabel("Lateral -> Medial (X)", fontsize=9)
    
    axes[2, 1].imshow(vol[z_s:z_e, y3, 160:250], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[2, 1].imshow(prep_viz(m_old[z_s:z_e, y3, 160:250]), cmap=cmap, alpha=0.55, origin="lower", interpolation="nearest")
    axes[2, 1].set_title("Sebelum Tuning (Koronal)", fontsize=11, fontweight="bold", color="darkred")
    
    axes[2, 2].imshow(vol[z_s:z_e, y3, 160:250], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[2, 2].imshow(prep_viz(m_new[z_s:z_e, y3, 160:250]), cmap=cmap, alpha=0.55, origin="lower", interpolation="nearest")
    axes[2, 2].set_title("Setelah Tuning (Koronal: Batas Anatomis Rapi)", fontsize=11, fontweight="bold", color="darkgreen")
    
    plt.suptitle(
        "HASIL PENYEMPURNAAN (TUNING) MASK GIGI 27 & 28 PASIEN PRATIWI\n"
        "Menghilangkan Tumpang Tindih Palsu & Memulihkan Morfologi Gigi 28 yang Impaksi Horizontal",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_png = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\tuned_molar_comparison.png")
    plt.savefig(str(out_png), dpi=200)
    plt.close()
    print(f"[+] Comparison saved to: {out_png}")


if __name__ == "__main__":
    compare_tuned_molars()
