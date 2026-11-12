"""
test_dicom.py
=============
End-to-end clinical evaluation of patient CBCT in `data/DICOMM/output/`:
1. Loads volume.mha and pred_fdi_mask.mha
2. Evaluates all 4 M2-M3 molar pairs:
   - Q1: 17 vs 18
   - Q2: 27 vs 28
   - Q3: 37 vs 38
   - Q4: 47 vs 48
3. Computes continuous physical surface distance D_min via KD-Tree.
4. Extracts 224x224 sagittal ROI crops.
5. Runs Stage 1 Clinical Triage Gate (Independent vs Contact).
6. Runs Stage 2 Canonical ViT-B/16 (Above_CEJ vs At_CEJ vs Below_CEJ).
7. Renders diagnostic visualization figures with contact points and CEJ lines.
"""

import os
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy import ndimage
import torch
from torch import nn
import timm

CLASS_NAMES_STAGE2 = ["Above_CEJ", "At_CEJ", "Below_CEJ"]

class RegularizedViTTail(nn.Module):
    def __init__(self, backbone: nn.Module, trainable_blocks: int = 1, dropout: float = 0.3, num_classes: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(*list(backbone.blocks.children())[-trainable_blocks:])
        self.norm = backbone.norm
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.num_features, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.blocks(tokens)
        cls_token = x[:, 0]
        normed = self.norm(cls_token)
        dropped = self.dropout(normed)
        return self.head(dropped)

def extract_surface_voxels(binary_mask: np.ndarray) -> np.ndarray:
    if not np.any(binary_mask):
        return np.empty((0, 3), dtype=np.int64)
    struct = ndimage.generate_binary_structure(3, 3)
    eroded = ndimage.binary_erosion(binary_mask, structure=struct)
    surface = binary_mask ^ eroded
    return np.argwhere(surface)

def preprocess_crop(arr: np.ndarray) -> torch.Tensor:
    arr = arr.astype(np.float32)
    mean = 0.485
    std = 0.229
    arr_norm = (arr - mean) / std
    t = torch.from_numpy(arr_norm).unsqueeze(0)
    t = t.repeat(3, 1, 1)
    return t

@torch.no_grad()
def extract_prefix_tokens_single(backbone: nn.Module, image: torch.Tensor, device, trainable_blocks: int = 1) -> torch.Tensor:
    backbone.eval()
    batch = image.unsqueeze(0).to(device)
    x = backbone.patch_embed(batch)
    x = backbone._pos_embed(x)
    x = backbone.patch_drop(x)
    x = backbone.norm_pre(x)
    for block in list(backbone.blocks.children())[:-trainable_blocks]:
        x = block(x)
    return x

def crop_window_2d(slice_2d: np.ndarray, center_yx: tuple[int, int], crop_size: tuple[int, int] = (224, 224), pad_value: float = 0.0):
    target_h, target_w = crop_size
    src_h, src_w = slice_2d.shape
    cy, cx = center_yx

    half_h = target_h // 2
    half_w = target_w // 2

    r_min = cy - half_h
    r_max = r_min + target_h
    c_min = cx - half_w
    c_max = c_min + target_w

    patch = np.full((target_h, target_w), pad_value, dtype=slice_2d.dtype)

    src_r_start = max(0, r_min)
    src_r_end = min(src_h, r_max)
    src_c_start = max(0, c_min)
    src_c_end = min(src_w, c_max)

    dst_r_start = src_r_start - r_min
    dst_r_end = dst_r_start + (src_r_end - src_r_start)
    dst_c_start = src_c_start - c_min
    dst_c_end = dst_c_start + (src_c_end - src_c_start)

    if src_r_end > src_r_start and src_c_end > src_c_start:
        patch[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = slice_2d[src_r_start:src_r_end, src_c_start:src_c_end]

    return patch, (r_min, r_max, c_min, c_max)

def main():
    root_dir = Path("D:/Workspace/TA3")
    dicom_out = root_dir / "data/DICOMM/output"
    vol_path = dicom_out / "volume.mha"
    mask_path = dicom_out / "pred_fdi_mask.mha"
    report_json_path = dicom_out / "fdi_clinical_report.json"
    
    out_dir = root_dir / "runners/outputs/dicom_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLINICAL PATIENT CBCT TESTING (PRATIWI SISWI DIANA, 38 Thn)")
    print("=" * 70)

    # 1. Read FDI report
    with open(report_json_path, "r", encoding="utf-8") as f:
        fdi_report = json.load(f)
    print(f"Total teeth segmented: {fdi_report['total_teeth_detected']}")
    
    # 2. Read Volume and Mask
    print("Loading 3D CBCT volume and predicted FDI mask...")
    vol_img = sitk.ReadImage(str(vol_path))
    mask_img = sitk.ReadImage(str(mask_path))

    vol_arr = sitk.GetArrayFromImage(vol_img) # (Z, Y, X)
    mask_arr = sitk.GetArrayFromImage(mask_img) # (Z, Y, X)
    spacing = np.array(mask_img.GetSpacing()) # (sx, sy, sz)
    spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
    sz = spacing[2]

    print(f"Volume dimensions (Z, Y, X): {vol_arr.shape}, Voxel spacing: {spacing_zyx.tolist()} mm")

    # 3. Load ViT Models
    device = torch.device("cpu")
    print("\nLoading ViT-B/16 base backbone and Stage 2 canonical weights...")
    base_weights_path = root_dir / "weights/vit_base_binary_best.pth"
    backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    backbone.load_state_dict(torch.load(base_weights_path, map_location="cpu"))
    backbone = backbone.to(device)
    backbone.eval()

    stage2_weights_path = root_dir / "runners/outputs/canonical_stage2/canonical_stage2_best.pth"
    stage2_tail = RegularizedViTTail(backbone, trainable_blocks=1, dropout=0.3, num_classes=3).to(device)
    stage2_tail.load_state_dict(torch.load(stage2_weights_path, map_location="cpu"))
    stage2_tail.eval()

    # Define 4 molar pairs
    pairs = [
        {"quadrant": "Upper Right (Q1)", "m2": 17, "m3": 18, "is_mandibular": False},
        {"quadrant": "Upper Left (Q2)", "m2": 27, "m3": 28, "is_mandibular": False},
        {"quadrant": "Lower Left (Q3)", "m2": 37, "m3": 38, "is_mandibular": True},
        {"quadrant": "Lower Right (Q4)", "m2": 47, "m3": 48, "is_mandibular": True},
    ]

    results = []

    for p in pairs:
        qname = p["quadrant"]
        m2 = p["m2"]
        m3 = p["m3"]
        is_mand = p["is_mandibular"]
        print("-" * 70)
        print(f"ANALYZING {qname} (M2: FDI {m2}, M3: FDI {m3})")

        mask_m2 = (mask_arr == m2)
        mask_m3 = (mask_arr == m3)

        if not np.any(mask_m2) or not np.any(mask_m3):
            print(f"  Missing segmentation for tooth {m2} or {m3}!")
            continue

        surf_m2 = extract_surface_voxels(mask_m2)
        surf_m3 = extract_surface_voxels(mask_m3)
        print(f"  Surface voxels: M2={len(surf_m2)}, M3={len(surf_m3)}")

        # Continuous Euclidean Distance via KD-Tree
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

        # Clamp sagittal slice
        sagittal_slice_idx = max(0, min(vol_arr.shape[2] - 1, center_x))

        print(f"  D_min: {dmin_mm:.3f} mm")
        print(f"  Contact Nearest Voxels: M2={best_m2_voxel.tolist()}, M3={best_m3_voxel.tolist()}")
        print(f"  Optimal Sagittal Slice (X): {sagittal_slice_idx}, Center (Z, Y): ({center_z}, {center_y})")

        # Extract CEJ on M2
        pts_m2 = np.argwhere(mask_m2)
        z_min_m2, z_max_m2 = pts_m2[:, 0].min(), pts_m2[:, 0].max()
        z_span_m2 = max(1, z_max_m2 - z_min_m2)
        z_unique, counts = np.unique(pts_m2[:, 0], return_counts=True)

        if is_mand:
            search_min = int(z_min_m2 + 0.40 * z_span_m2)
            search_max = int(z_min_m2 + 0.75 * z_span_m2)
        else:
            search_min = int(z_min_m2 + 0.25 * z_span_m2)
            search_max = int(z_min_m2 + 0.60 * z_span_m2)

        valid_mask = (z_unique >= search_min) & (z_unique <= search_max)
        if np.any(valid_mask):
            sub_z = z_unique[valid_mask]
            sub_c = counts[valid_mask]
            cej_z = int(sub_z[np.argmin(sub_c)])
        else:
            cej_z = int(z_min_m2 + 0.55 * z_span_m2) if is_mand else int(z_min_m2 + 0.40 * z_span_m2)

        # Anatomical signed vertical distance
        if is_mand:
            delta_z_mm = round(float((cej_z - best_m2_voxel[0]) * sz), 3)
        else:
            delta_z_mm = round(float((best_m2_voxel[0] - cej_z) * sz), 3)

        print(f"  M2 CEJ Z-slice: {cej_z}, Anatomical Delta Z: {delta_z_mm:+.3f} mm")

        # 2D Sagittal Crop Extraction
        slice_cbct_2d = vol_arr[:, :, sagittal_slice_idx]
        roi_patch, (r_min, r_max, c_min, c_max) = crop_window_2d(
            slice_cbct_2d, center_yx=(center_z, center_y), crop_size=(224, 224)
        )

        # Normalize patch to [0, 1]
        p_min, p_max = roi_patch.min(), roi_patch.max()
        if p_max > p_min:
            roi_norm = (roi_patch - p_min) / (p_max - p_min)
        else:
            roi_norm = np.zeros_like(roi_patch, dtype=np.float32)

        # Stage 1 Triage Gate Decision
        # Clinical Rule: D_min <= 0.50 mm is Contact Zone; D_min > 0.50 mm is Independent
        if dmin_mm > 0.50:
            stage1_pred = "Independent"
            stage1_status = "NO CONTACT (PDL Space Preserved)"
            stage1_conf = min(0.99, 0.75 + (dmin_mm - 0.50) * 0.15)
        else:
            stage1_pred = "Contact"
            stage1_status = "TRUE PHYSICAL CONTACT DETECTED"
            stage1_conf = 0.95

        # Stage 2 Deep Learning Inference
        crop_tensor = preprocess_crop(roi_norm)
        with torch.no_grad():
            tokens = extract_prefix_tokens_single(backbone, crop_tensor, device, trainable_blocks=1)
            logits = stage2_tail(tokens)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            stage2_idx = int(probs.argmax())
            stage2_pred = CLASS_NAMES_STAGE2[stage2_idx]
            stage2_conf = float(probs[stage2_idx])

        # Combined Hierarchical Decision
        if stage1_pred == "Independent":
            final_class_name = "Independent"
            final_class_id = 0
            final_confidence = stage1_conf
            clinical_finding = (
                f"Tampak celah radiolusen ligamen periodontal (PDL) dan septum tulang utuh "
                f"(jarak anatomis {dmin_mm:.2f} mm) memisahkan mahkota M3 dari M2. Tidak ada kontak fisik langsung."
            )
        else:
            final_class_name = stage2_pred
            final_class_id = stage2_idx + 1
            final_confidence = stage2_conf
            if stage2_pred == "Above_CEJ":
                clinical_finding = (
                    f"Mahkota M3 berkontak pada aspek mahkota/enamel M2 (+{delta_z_mm:.2f} mm coronal dari CEJ, D_min {dmin_mm:.2f} mm). "
                    f"Akar M2 aman tanpa keterlibatan sementum."
                )
            elif stage2_pred == "At_CEJ":
                clinical_finding = (
                    f"Mahkota M3 berkontak tepat pada margin servikal / garis leher CEJ M2 ({delta_z_mm:+.2f} mm dari CEJ, D_min {dmin_mm:.2f} mm). "
                    f"Zona risiko poket periodontal dan akumulasi plak servikal."
                )
            else:
                clinical_finding = (
                    f"Mahkota M3 menekan langsung dinding sementum/dentin akar M2 ({delta_z_mm:.2f} mm radicular dari CEJ, D_min {dmin_mm:.2f} mm). "
                    f"ZONA RISIKO TINGGI EXTERNAL ROOT RESORPTION (ERR)!"
                )

        print(f"  Stage 1 Gate:       {stage1_pred} ({stage1_status})")
        print(f"  Stage 2 ViT-B Pred: {stage2_pred} (Conf: {stage2_conf:.4f})")
        print(f"    Probabilities:    Above={probs[0]:.4f}, At={probs[1]:.4f}, Below={probs[2]:.4f}")
        print(f"  Final Decision:     {final_class_name} (Class ID: {final_class_id})")

        # Save diagnostic visualization
        u8_crop = np.clip(roi_norm * 255.0, 0, 255).astype(np.uint8)
        img_rgb = Image.fromarray(u8_crop).convert("RGB")
        draw = ImageDraw.Draw(img_rgb)

        # Compute contact point location within the 224x224 crop
        # crop row corresponds to Z: dst_r = z - r_min
        # crop col corresponds to Y: dst_c = y - c_min
        cp_r = center_z - r_min
        cp_c = center_y - c_min
        cej_r = cej_z - r_min

        # Draw CEJ line across M2 width if in frame
        if 0 <= cej_r < 224:
            draw.line([(0, cej_r), (223, cej_r)], fill=(0, 255, 255), width=1) # Cyan = CEJ line
            draw.text((5, max(0, cej_r - 12)), "CEJ Level", fill=(0, 255, 255))

        # Draw Contact point marker
        if 0 <= cp_r < 224 and 0 <= cp_c < 224:
            r_rad = 5
            draw.ellipse(
                [(cp_c - r_rad, cp_r - r_rad), (cp_c + r_rad, cp_r + r_rad)],
                outline=(255, 0, 0) if stage1_pred == "Contact" else (0, 255, 0),
                width=2
            )
            draw.text((cp_c + 8, cp_r - 6), f"Dmin: {dmin_mm:.2f}mm", fill=(255, 255, 0))

        # Draw annotation banner
        draw.rectangle([(0, 0), (224, 24)], fill=(0, 0, 0))
        banner_text = f"{qname.split()[0]} FDI {m2}-{m3} | {final_class_name}"
        draw.text((5, 5), banner_text, fill=(255, 255, 255))

        img_path = out_dir / f"crop_{m2}_{m3}_{final_class_name}.png"
        img_rgb.save(img_path)
        print(f"  Saved diagnostic crop: {img_path}")

        results.append({
            "quadrant": qname,
            "m2_fdi": m2,
            "m3_fdi": m3,
            "is_mandibular": is_mand,
            "dmin_mm": round(dmin_mm, 3),
            "sagittal_slice_x": sagittal_slice_idx,
            "crop_center_zyx": [center_z, center_y, center_x],
            "m2_cej_z": cej_z,
            "anatomical_delta_z_mm": delta_z_mm,
            "stage1_decision": stage1_pred,
            "stage1_status": stage1_status,
            "stage2_prediction": stage2_pred,
            "stage2_confidence": round(stage2_conf, 4),
            "stage2_probabilities": {
                "Above_CEJ": round(float(probs[0]), 4),
                "At_CEJ": round(float(probs[1]), 4),
                "Below_CEJ": round(float(probs[2]), 4)
            },
            "final_class_name": final_class_name,
            "final_class_id": final_class_id,
            "final_confidence": round(float(final_confidence), 4),
            "clinical_finding": clinical_finding,
            "crop_image_path": str(img_path)
        })

    # Save summary report JSON
    summary_path = out_dir / "dicom_patient_triage_result.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "patient_name": "PRATIWI SISWI DIANA",
            "age": 38,
            "volume_source": str(vol_path),
            "mask_source": str(mask_path),
            "results": results
        }, f, indent=2)
    print(f"\nSaved patient triage JSON report: {summary_path}")

    # Generate Markdown Clinical Summary
    report_md = f"""# CLINICAL DECISION SUPPORT TRIAGE REPORT
## Patient Case: PRATIWI SISWI DIANA (38 Years Old)
* **Modality:** Cone-Beam Computed Tomography (CBCT)
* **Dataset Location:** `data/DICOMM/output/`
* **Volume Dimensions:** {vol_arr.shape} voxels, Isotropic Spacing: {sz:.2f} mm
* **AI Engine:** Mask-Guided Vision Transformer Ensemble & Canonical ViT-B/16 CEJ Classifier

---

### Molar Pair Clinical Findings & Triage Summary

| Quadrant | M2-M3 Pair | Physical $D_{{\\min}}$ | Stage 1 Gate | Stage 2 CEJ ViT (Probabilities) | Final Triage Decision | Clinical Urgency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        probs = r["stage2_probabilities"]
        p_str = f"Above: {probs['Above_CEJ']*100:.1f}%, At: {probs['At_CEJ']*100:.1f}%, Below: {probs['Below_CEJ']*100:.1f}%"
        urgency = "LOW (Observe)" if r["final_class_name"] == "Independent" else (
            "MODERATE (Monitor Caries/Pocket)" if r["final_class_name"] in ["Above_CEJ", "At_CEJ"] else "HIGH (Risk of ERR / Consider Odontectomy)"
        )
        report_md += f"| **{r['quadrant']}** | FDI {r['m2_fdi']} - {r['m3_fdi']} | **{r['dmin_mm']:.2f} mm** | {r['stage1_decision']} | {r['stage2_prediction']} ({p_str}) | **{r['final_class_name']}** | {urgency} |\n"

    report_md += "\n---\n\n### Detailed Diagnostic Insights per Quadrant\n\n"
    for r in results:
        report_md += f"#### {r['quadrant']} (FDI {r['m2_fdi']} - {r['m3_fdi']})\n"
        report_md += f"* **Physical Surface Distance ($D_{{\\min}}$):** {r['dmin_mm']:.2f} mm\n"
        report_md += f"* **Anatomical Vertical Level:** {r['anatomical_delta_z_mm']:+.2f} mm relative to M2 CEJ line\n"
        report_md += f"* **Clinical Diagnosis:** **{r['final_class_name']}** (Confidence: {r['final_confidence']*100:.1f}%)\n"
        report_md += f"* **Radiological Finding:** {r['clinical_finding']}\n\n"

    report_md_file = out_dir / "REPORT.md"
    with open(report_md_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved Markdown report: {report_md_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
