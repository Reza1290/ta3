"""Generate Multi-View Clinical Zoom Explaining the Overlapping M2-M3 Impaction.

Shows:
1. High-Res Sagittal Zoom (X=207)
2. High-Res Coronal Zoom (Y=385)
3. High-Res Axial Zoom (Z=165)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

img_itk = sitk.ReadImage(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
mask_itk = sitk.ReadImage(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")

arr_img = sitk.GetArrayFromImage(img_itk)
arr_mask = sitk.GetArrayFromImage(mask_itk)

# Physical bone display window
vmin, vmax = 200, 2400
disp = np.clip(arr_img, vmin, vmax)
disp = (disp - vmin) / (vmax - vmin)

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor="#0e1117")

# 1. Sagittal View (X = 207)
ax_sag = axes[0]
ax_sag.imshow(disp[:, :, 207], cmap="gray", origin="upper")
ax_sag.contour(arr_mask[:, :, 207] == 27, levels=[0.5], colors=["#FF9100"], linewidths=2.0, origin="upper")
ax_sag.contour(arr_mask[:, :, 207] == 28, levels=[0.5], colors=["#FF1744"], linewidths=2.0, origin="upper")
ax_sag.set_title("1. Potongan Sagittal (X=207)\nTampak Samping (Anterior <-> Posterior)", color="white", fontsize=11, fontweight="bold")
ax_sag.set_xlim(280, 460)
ax_sag.set_ylim(320, 100)
ax_sag.axis("off")

# Annotations Sagittal
ax_sag.annotate("Molar 3 (FDI 28)\nIMPAKSI HORIZONTAL\n(Gigi Bungsu Atas)",
                xy=(410, 155), xytext=(320, 125),
                arrowprops=dict(facecolor="#FF1744", shrink=0.08, width=1.8, headwidth=6),
                color="#FF7B72", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#21262d", ec="#FF1744", lw=1.2))

ax_sag.annotate("Molar 2 (FDI 27)\nAkar Gigi Berdampingan/Tertindih",
                xy=(370, 230), xytext=(290, 270),
                arrowprops=dict(facecolor="#FF9100", shrink=0.08, width=1.8, headwidth=6),
                color="#FFA657", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#21262d", ec="#FF9100", lw=1.2))


# 2. Coronal View (Y = 385)
ax_cor = axes[1]
ax_cor.imshow(disp[:, 385, :], cmap="gray", origin="upper")
ax_cor.contour(arr_mask[:, 385, :] == 27, levels=[0.5], colors=["#FF9100"], linewidths=2.0, origin="upper")
ax_cor.contour(arr_mask[:, 385, :] == 28, levels=[0.5], colors=["#FF1744"], linewidths=2.0, origin="upper")
ax_cor.set_title("2. Potongan Coronal (Y=385)\nTampak Depan-Belakang (Superior <-> Inferior)", color="white", fontsize=11, fontweight="bold")
ax_cor.set_xlim(140, 280)
ax_cor.set_ylim(320, 100)
ax_cor.axis("off")

ax_cor.annotate("M3 (28) di Tulang Atas",
                xy=(200, 150), xytext=(150, 120),
                arrowprops=dict(facecolor="#FF1744", shrink=0.08, width=1.5, headwidth=5),
                color="#FF7B72", fontsize=9.5, fontweight="bold")

ax_cor.annotate("M2 (27) Erupsi Bawah",
                xy=(210, 270), xytext=(150, 300),
                arrowprops=dict(facecolor="#FF9100", shrink=0.08, width=1.5, headwidth=5),
                color="#FFA657", fontsize=9.5, fontweight="bold")


# 3. Axial View (Z = 165 - level of impaction)
ax_ax = axes[2]
ax_ax.imshow(disp[165, :, :], cmap="gray", origin="upper")
ax_ax.contour(arr_mask[165, :, :] == 27, levels=[0.5], colors=["#FF9100"], linewidths=2.0, origin="upper")
ax_ax.contour(arr_mask[165, :, :] == 28, levels=[0.5], colors=["#FF1744"], linewidths=2.0, origin="upper")
ax_ax.set_title("3. Potongan Axial (Z=165)\nTampak Horisontal Tulang Tuber Maksila", color="white", fontsize=11, fontweight="bold")
ax_ax.set_xlim(140, 280)
ax_ax.set_ylim(450, 310)
ax_ax.axis("off")

ax_ax.annotate("M3 (28) Melintang Horisontal",
               xy=(205, 410), xytext=(150, 440),
               arrowprops=dict(facecolor="#FF1744", shrink=0.08, width=1.5, headwidth=5),
               color="#FF7B72", fontsize=9.5, fontweight="bold")

fig.suptitle("PENJELASAN ANATOMI KLINIS: FENOMENA 'TUMPANG TINDIH' GIGI 27 & 28\nPasien PRATIWI SISWI DIANA - Impaksi Tulang Kompleks Molar 3 (FDI 28) di Atas Molar 2 (FDI 27)",
             color="#58A6FF", fontsize=13, fontweight="bold", y=0.98)

out_png = Path(r"C:\Users\rezam\.gemini\antigravity\brain\5376da8e-88a6-42f7-99ed-9736ffae41be\clinical_explanation_overlapping.png")
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"[+] Saved explanation visual to: {out_png}")
