"""Render Side-by-Side Comparison of Before vs After Mask Refinement.

Shows:
1. Raw CBCT (0.16mm, Sagittal X=207)
2. Old ToothFairy2 Mask (with under-segmentation gaps)
3. New Morphologically-Refined Mask (full tooth coverage)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

img_itk = sitk.ReadImage(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
old_mask_itk = sitk.ReadImage(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")
new_mask_itk = sitk.ReadImage(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask_anatomical_refined.mha")

arr_img = sitk.GetArrayFromImage(img_itk)
arr_old = sitk.GetArrayFromImage(old_mask_itk)
arr_new = sitk.GetArrayFromImage(new_mask_itk)

x = 207
sl_img = np.clip(arr_img[:, :, x], 200, 2400)
sl_img = (sl_img - 200) / 2200

sl_old = arr_old[:, :, x]
sl_new = arr_new[:, :, x]

fig, (ax_raw, ax_old, ax_new) = plt.subplots(1, 3, figsize=(21, 8), facecolor="#0e1117")

for ax in (ax_raw, ax_old, ax_new):
    ax.set_xlim(260, 440)
    ax.set_ylim(390, 110)
    ax.axis("off")

# 1. Raw
ax_raw.imshow(sl_img, cmap="gray", origin="upper")
ax_raw.set_title("1. Citra CBCT Asli (Sagittal X=207)\nResolusi Isotropik Asli 0.16 mm", color="white", fontsize=12, fontweight="bold", pad=8)

# Color palette
colors = {
    25: [0.0, 0.9, 1.0, 0.45],   # Cyan
    26: [1.0, 0.85, 0.0, 0.45],  # Yellow
    27: [1.0, 0.55, 0.0, 0.45],  # Orange
    28: [1.0, 0.1, 0.25, 0.45],  # Red
    36: [0.65, 0.45, 0.95, 0.45],# Purple
    37: [0.0, 0.9, 0.45, 0.45],  # Green
    38: [0.2, 0.6, 1.0, 0.45],   # Blue
}

def draw_overlay(ax, mask_slice, title, subtitle):
    ax.imshow(sl_img, cmap="gray", origin="upper")
    overlay = np.zeros((*sl_img.shape, 4))
    for fdi, col in colors.items():
        b = (mask_slice == fdi)
        if b.sum() > 0:
            overlay[b] = col
            ax.contour(b, levels=[0.5], colors=[col[:3]], linewidths=1.8, origin="upper")
            pts = np.argwhere(b)
            cz, cy = pts.mean(axis=0)
            ax.text(cy, cz, str(fdi), color="white", fontsize=10.5, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.2", fc=col[:3], ec="black", lw=1.0))
    ax.imshow(overlay, origin="upper")
    ax.set_title(f"{title}\n{subtitle}", color="#58A6FF", fontsize=12, fontweight="bold", pad=8)

# 2. Old Mask
draw_overlay(ax_old, sl_old, "2. Mask Lama (Sebelum Refinement)", "Gigi 28 & 27 Masih Terpotong / Bolong Sebagian")

# 3. New Mask
draw_overlay(ax_new, sl_new, "3. Mask Baru (Sesudah Refinement)", "Cakupan Penuh Mahkota, Pulpa & Akar Gigi")

fig.suptitle("EVALUASI PERBAIKAN GEOMETRI MASK TOOTHFAIRY2 (KUADRAN KIRI ATAS)\nKonfirmasi Batas Mahkota, Email & Akar Gigi 26, 27, dan 28",
             color="#58A6FF", fontsize=14, fontweight="bold", y=0.98)

out_png = Path(r"C:\Users\rezam\.gemini\antigravity\brain\5376da8e-88a6-42f7-99ed-9736ffae41be\compare_mask_refinement_q2.png")
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"[+] Saved comparison visual to: {out_png}")
