"""Generate Detailed Clinical Explanation for FDI 27 & 28 Classification.

Explains:
1. Why FDI 28 is clinically genuine (High Horizontal Impacted Third Molar).
2. Why FDI 27 was erroneously over-split from FDI 26 crown.
3. Where the genuine physical M2 (FDI 27) tooth sits in the dental arch.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

img_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")
out_png = Path(r"C:\Users\rezam\.gemini\antigravity\brain\5376da8e-88a6-42f7-99ed-9736ffae41be\clinical_audit_27_28.png")

img_itk = sitk.ReadImage(str(img_path))
mask_itk = sitk.ReadImage(str(mask_path))

arr_img = sitk.GetArrayFromImage(img_itk)
arr_mask = sitk.GetArrayFromImage(mask_itk)

# Slice X = 207 (Left Maxilla Molars)
sl_img = np.clip(arr_img[:, :, 207], 200, 2400)
sl_img = (sl_img - 200) / 2200
sl_mask = arr_mask[:, :, 207]

fig, (ax_raw, ax_diag) = plt.subplots(1, 2, figsize=(18, 9), facecolor="#0e1117")

# Panel 1: Pure Anatomy
ax_raw.imshow(sl_img, cmap="gray", origin="upper")
ax_raw.set_title("1. Anatomi Asli Citra CBCT (Kiri Atas / Kuadran 2)\nPotongan Sagittal X=207",
                 color="#58A6FF", fontsize=12, fontweight="bold", pad=10)
ax_raw.set_xlim(160, 470)
ax_raw.set_ylim(410, 110)
ax_raw.axis("off")

# Anatomical annotations on raw
ax_raw.text(250, 140, "Sinus Maksilaris", color="#79C0FF", fontsize=10, style="italic")
ax_raw.annotate("Gigi Premolar (P2)\nFDI 25", xy=(260, 270), xytext=(210, 230),
                arrowprops=dict(facecolor="#58A6FF", shrink=0.08, width=1.5, headwidth=6),
                color="white", fontsize=9.5, fontweight="bold")
ax_raw.annotate("Molar 1 Asli\n(Trifurkasi Lengkap)", xy=(310, 260), xytext=(270, 180),
                arrowprops=dict(facecolor="#58A6FF", shrink=0.08, width=1.5, headwidth=6),
                color="white", fontsize=9.5, fontweight="bold")
ax_raw.annotate("Molar 2 Asli\n(Erupsi Normal di Lengkung)", xy=(365, 270), xytext=(350, 210),
                arrowprops=dict(facecolor="#3FB950", shrink=0.08, width=2.0, headwidth=7),
                color="#3FB950", fontsize=10, fontweight="bold")
ax_raw.annotate("Molar 3 Asli (M3)\nIMPAKSI TINGGI TUBER MAKSILA", xy=(430, 165), xytext=(330, 130),
                arrowprops=dict(facecolor="#F85149", shrink=0.08, width=2.0, headwidth=7),
                color="#F85149", fontsize=10.5, fontweight="bold")


# Panel 2: Model Prediction & Audit
ax_diag.imshow(sl_img, cmap="gray", origin="upper")
ax_diag.set_title("2. Hasil Segmentasi AI ToothFairy2 & Analisis Kesalahan\nIdentifikasi Split Error & Missed Segmentation",
                  color="#FF7B72", fontsize=12, fontweight="bold", pad=10)
ax_diag.set_xlim(160, 470)
ax_diag.set_ylim(410, 110)
ax_diag.axis("off")

# Draw model contours
color_map = {25: "#00E5FF", 26: "#FFD600", 27: "#FF9100", 28: "#FF1744", 36: "#A371F7", 37: "#00E676"}
for fdi in [25, 26, 27, 28, 36, 37]:
    b = (sl_mask == fdi)
    if b.sum() > 0:
        c = color_map.get(fdi, "white")
        ax_diag.contour(b, levels=[0.5], colors=[c], linewidths=2.2, origin="upper")

# Explanatory Callouts on AI predictions
ax_diag.annotate("FDI 28: BENAR!\nIni memang Gigi Bungsu (M3)\nyang impaksi di tuber maksila.",
                 xy=(430, 165), xytext=(300, 135),
                 arrowprops=dict(facecolor="#FF1744", shrink=0.08, width=2.0, headwidth=7),
                 color="#FF7B72", fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#21262d", ec="#FF1744", lw=1.5))

ax_diag.annotate("KESALAHAN 1 (VERTICAL OVER-SPLIT):\nSatu gigi Molar 1 dibelah dua:\n- Akar dilabel FDI 26 (Kuning)\n- Mahkota dilabel FDI 27 (Oranye)",
                 xy=(310, 270), xytext=(170, 340),
                 arrowprops=dict(facecolor="#FFD600", shrink=0.08, width=2.0, headwidth=7),
                 color="#FFD600", fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#21262d", ec="#FFD600", lw=1.5))

ax_diag.annotate("KESALAHAN 2 (MISSED M2):\nGigi Molar 2 Asli (FDI 27) ada di sini,\ntetapi TERLEWAT oleh model AI\nkarena oklusi rapat dengan gigi bawah 37.",
                 xy=(365, 270), xytext=(320, 360),
                 arrowprops=dict(facecolor="#3FB950", shrink=0.08, width=2.0, headwidth=7),
                 color="#7EE787", fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#21262d", ec="#3FB950", lw=1.5))

fig.suptitle("AUDIT KLINIS KLASIFIKASI GIGI 27 & 28 (KUADRAN KIRI ATAS)\nKonfirmasi Temuan Visual vs Kenyataan Anatomi CBCT Pasien Pratiwi",
             color="#58A6FF", fontsize=14, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"[+] Saved clinical audit image to: {out_png}")
