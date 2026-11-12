"""
render_roi_and_masks.py
=======================
Renders side-by-side comparisons of:
1. Pure 2D Sagittal CBCT Crop (224x224)
2. Ground-truth Predicted FDI Mask (224x224)
3. Overlay of Mask Contours & Contact Points on CBCT
For all 4 quadrants (17-18, 27-28, 37-38, 47-48) of patient PRATIWI SISWI DIANA.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

from test_dicom import crop_window_2d, extract_surface_voxels

def main():
    root_dir = Path("D:/Workspace/TA3")
    dicom_out = root_dir / "data/DICOMM/output"
    vol_path = dicom_out / "volume.mha"
    mask_path = dicom_out / "pred_fdi_mask.mha"
    
    out_dir = root_dir / "runners/outputs/mask_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    brain_dir = Path("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5")

    print("Loading 3D Volume and 3D Mask...")
    vol_img = sitk.ReadImage(str(vol_path))
    mask_img = sitk.ReadImage(str(mask_path))
    vol_arr = sitk.GetArrayFromImage(vol_img)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    spacing = np.array(mask_img.GetSpacing())
    spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
    sz = spacing[2]

    pairs = [
        {"qname": "Q1_Upper_Right_17_18", "title": "Kuadran 1 (Kanan Atas): FDI 17 - 18", "m2": 17, "m3": 18, "is_mand": False},
        {"qname": "Q2_Upper_Left_27_28", "title": "Kuadran 2 (Kiri Atas): FDI 27 - 28", "m2": 27, "m3": 28, "is_mand": False},
        {"qname": "Q3_Lower_Left_37_38", "title": "Kuadran 3 (Kiri Bawah): FDI 37 - 38", "m2": 37, "m3": 38, "is_mand": True},
        {"qname": "Q4_Lower_Right_47_48", "title": "Kuadran 4 (Kanan Bawah): FDI 47 - 48", "m2": 47, "m3": 48, "is_mand": True},
    ]

    for p in pairs:
        qname = p["qname"]
        title = p["title"]
        m2 = p["m2"]
        m3 = p["m3"]
        is_mand = p["is_mand"]
        print(f"\nProcessing {title}...")

        mask_m2 = (mask_arr == m2)
        mask_m3 = (mask_arr == m3)

        surf_m2 = extract_surface_voxels(mask_m2)
        surf_m3 = extract_surface_voxels(mask_m3)

        sp = spacing_zyx.astype(np.float64)
        phys_m2 = surf_m2.astype(np.float64) * sp
        phys_m3 = surf_m3.astype(np.float64) * sp

        tree = cKDTree(phys_m2)
        dists, nearest_m2_idxs = tree.query(phys_m3)
        best_m3_idx = int(np.argmin(dists))
        dmin_mm = float(dists[best_m3_idx])

        best_m3_voxel = surf_m3[best_m3_idx]
        best_m2_voxel = surf_m2[nearest_m2_idxs[best_m3_idx]]

        center_z = int(np.round((best_m2_voxel[0] + best_m3_voxel[0]) / 2.0))
        center_y = int(np.round((best_m2_voxel[1] + best_m3_voxel[1]) / 2.0))
        center_x = int(np.round((best_m2_voxel[2] + best_m3_voxel[2]) / 2.0))
        sagittal_slice_idx = max(0, min(vol_arr.shape[2] - 1, center_x))

        # Extract 2D sagittal slice from volume and mask
        cbct_2d = vol_arr[:, :, sagittal_slice_idx]
        mask_2d = mask_arr[:, :, sagittal_slice_idx]

        # Crop 224x224
        cbct_crop, (r_min, r_max, c_min, c_max) = crop_window_2d(cbct_2d, center_yx=(center_z, center_y), crop_size=(224, 224))
        mask_crop, _ = crop_window_2d(mask_2d, center_yx=(center_z, center_y), crop_size=(224, 224), pad_value=0)

        # 1. Normalize CBCT crop
        c_min_val, c_max_val = cbct_crop.min(), cbct_crop.max()
        cbct_norm = (cbct_crop - c_min_val) / (c_max_val - c_min_val + 1e-7)
        cbct_u8 = np.clip(cbct_norm * 255.0, 0, 255).astype(np.uint8)
        img_cbct = Image.fromarray(cbct_u8).convert("RGB")

        # 2. Build Colored Mask Crop
        # Colors:
        # M2: Cyan (0, 220, 220)
        # M3: Orange/Red (255, 120, 0)
        # Other teeth: Gray (160, 160, 160)
        # Background: Black (0, 0, 0)
        mask_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        other_teeth = (mask_crop > 0) & (mask_crop != m2) & (mask_crop != m3)
        mask_rgb[other_teeth] = [120, 120, 120]
        mask_rgb[mask_crop == m2] = [0, 220, 220]  # M2 in Cyan
        mask_rgb[mask_crop == m3] = [255, 120, 0]  # M3 in Orange
        img_mask = Image.fromarray(mask_rgb)

        # 3. Build Overlay (CBCT + Colored Contours)
        overlay_rgb = np.stack([cbct_u8, cbct_u8, cbct_u8], axis=-1)

        # Extract 2D boundary of M2 and M3
        struct2d = ndimage.generate_binary_structure(2, 1)
        m2_bin = (mask_crop == m2)
        m3_bin = (mask_crop == m3)
        other_bin = other_teeth

        edge_m2 = m2_bin ^ ndimage.binary_erosion(m2_bin, structure=struct2d)
        edge_m3 = m3_bin ^ ndimage.binary_erosion(m3_bin, structure=struct2d)
        edge_other = other_bin ^ ndimage.binary_erosion(other_bin, structure=struct2d)

        # Dilate edge slightly for visibility
        edge_m2 = ndimage.binary_dilation(edge_m2, structure=struct2d)
        edge_m3 = ndimage.binary_dilation(edge_m3, structure=struct2d)

        overlay_rgb[edge_other] = [160, 160, 160]
        overlay_rgb[edge_m2] = [0, 255, 255] # Cyan contour for M2
        overlay_rgb[edge_m3] = [255, 100, 0] # Orange contour for M3
        img_overlay = Image.fromarray(overlay_rgb)

        # Draw contact marker on overlay
        draw_ov = ImageDraw.Draw(img_overlay)
        cp_r = center_z - r_min
        cp_c = center_y - c_min
        if 0 <= cp_r < 224 and 0 <= cp_c < 224:
            r_rad = 4
            draw_ov.ellipse(
                [(cp_c - r_rad, cp_r - r_rad), (cp_c + r_rad, cp_r + r_rad)],
                outline=(255, 255, 0), width=2
            )
            draw_ov.text((cp_c + 6, max(0, cp_r - 6)), f"Dmin: {dmin_mm:.2f}mm", fill=(255, 255, 0))

        # Annotate labels on each panel
        draw_cbct = ImageDraw.Draw(img_cbct)
        draw_cbct.rectangle([(0, 0), (224, 20)], fill=(0, 0, 0))
        draw_cbct.text((5, 4), "1. Citra Asli CBCT (2D Crop)", fill=(255, 255, 255))

        draw_mask = ImageDraw.Draw(img_mask)
        draw_mask.rectangle([(0, 0), (224, 20)], fill=(0, 0, 0))
        draw_mask.text((5, 4), f"2. Mask Segmentasi (M2:{m2}, M3:{m3})", fill=(255, 255, 255))

        draw_ov.rectangle([(0, 0), (224, 20)], fill=(0, 0, 0))
        draw_ov.text((5, 4), "3. Overlay Kontur Mask + Titik Kontak", fill=(255, 255, 255))

        # Combine into 1 side-by-side composite image: (224 x 3 + 20 spacing) x (224 + 40 banner)
        panel_w, panel_h = 224, 224
        header_h = 36
        margin = 10
        comp_w = panel_w * 3 + margin * 4
        comp_h = panel_h + header_h + margin * 2

        comp = Image.new("RGB", (comp_w, comp_h), (25, 25, 25))
        draw_comp = ImageDraw.Draw(comp)

        # Header banner
        draw_comp.text((margin, 8), f"{title} | Irisan Sagital X = {sagittal_slice_idx} | D_min = {dmin_mm:.2f} mm", fill=(255, 255, 255))

        # Paste 3 panels
        comp.paste(img_cbct, (margin, header_h))
        comp.paste(img_mask, (margin * 2 + panel_w, header_h))
        comp.paste(img_overlay, (margin * 3 + panel_w * 2, header_h))

        out_img_path = out_dir / f"inspection_{qname}.png"
        comp.save(out_img_path)
        comp.save(brain_dir / f"inspection_{qname}.png")
        print(f"Saved comparison to: {out_img_path}")

    print("\nAll 4 quadrant mask inspections generated successfully!")

if __name__ == "__main__":
    main()
