# ToothFairy2 CBCT Inference & FDI Tooth Classification Pipeline

Monorepo subproject for converting clinical dental CBCT DICOM series to 3D MetaImage (`.mha`) format and performing automated tooth instance segmentation and FDI (ISO 3950) classification using the **ToothFairy2 Benchmark / ToothSeg** deep learning model.

---

## 1. Overview & Capabilities

* **DICOM to MHA Conversion (`convert_dicom_to_mha.py`):**
  * Reads multi-slice DICOM series with SimpleITK (`sitk.ImageSeriesReader`).
  * Preserves native isotropic spatial resolution ($0.16 \times 0.16 \times 0.16$ mm), orientation cosine direction matrix, and origin coordinates.
  * Writes compressed `.mha` files to reduce disk footprint (~60% space reduction).

* **ToothFairy2 / ToothSeg Architecture (`predict_toothfairy2.py`):**
  * Built on **PlainConvUNet** (`dynamic-network-architectures`) trained on the MICCAI ToothFairy2 dataset (Zenodo DOI: [10.5281/zenodo.14893540](https://doi.org/10.5281/zenodo.14893540)).
  * Operates with 3D sliding-window inference and Gaussian importance blending on an NVIDIA GPU (RTX 4050 with CUDA 12.1).
  * Segment and classify **all 32 adult permanent teeth** into international standard **FDI two-digit codes** (11–18, 21–28, 31–38, 41–48).

* **Clinical Inventory & M2-M3 Relation Analysis:**
  * Computes physical volume ($mm^3$), 3D bounding box, and centroid for every detected tooth.
  * Evaluates presence and eligibility of Second Molar (M2) and Third Molar (M3) across all 4 quadrants (UR, UL, LL, LR) for clinical triage gate processing.

* **Multi-Planar Reconstruction (MPR) Visualization (`visualize_fdi_results.py`):**
  * Renders Axial, Coronal, and Quadrant-specific Sagittal slices with color-coded tooth contours and FDI numbering.

---

## 2. Directory Structure

```text
projects/m3-m2-toothfairy2-inference/
├── README.md                      # Pipeline documentation & clinical usage guide
├── fdi_taxonomy.py                # FDI (ISO 3950) notation mapping & tooth metadata
├── convert_dicom_to_mha.py        # SimpleITK DICOM series to .mha converter
├── download_models.py             # Automatic weights downloader (Zenodo / Hugging Face)
├── predict_toothfairy2.py         # 3D sliding-window inference & FDI classification engine
├── visualize_fdi_results.py       # Multi-planar reconstruction (MPR) visualization
└── run_pipeline.py                # Unified end-to-end CLI pipeline
```

---

## 3. Quick Start & Execution

### Run Complete End-to-End Pipeline
```powershell
# Navigate to monorepo root
cd D:\Workspace\TA3

# Run the complete pipeline on data/DICOMM
& "D:\Workspace\TA3\.venv_cuda\Scripts\python.exe" projects\m3-m2-toothfairy2-inference\run_pipeline.py `
    --dicom_dir "D:\Workspace\TA3\data\DICOMM" `
    --output_dir "D:\Workspace\TA3\data\DICOMM\output" `
    --device "cuda"
```

### Run Step-by-Step

#### 1. Convert DICOM to MHA
```powershell
& "D:\Workspace\TA3\.venv_cuda\Scripts\python.exe" projects\m3-m2-toothfairy2-inference\convert_dicom_to_mha.py `
    --dicom_dir "D:\Workspace\TA3\data\DICOMM" `
    --output_mha "D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA.mha"
```

#### 2. Run FDI Inference
```powershell
& "D:\Workspace\TA3\.venv_cuda\Scripts\python.exe" projects\m3-m2-toothfairy2-inference\predict_toothfairy2.py `
    --input_mha "D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA.mha" `
    --output_mha "D:\Workspace\TA3\data\DICOMM\output\pred_fdi_mask.mha" `
    --report_json "D:\Workspace\TA3\data\DICOMM\output\fdi_report.json"
```

#### 3. Render MPR Overlays
```powershell
& "D:\Workspace\TA3\.venv_cuda\Scripts\python.exe" projects\m3-m2-toothfairy2-inference\visualize_fdi_results.py `
    --cbct_mha "D:\Workspace\TA3\data\DICOMM\PRATIWI_SISWI_DIANA.mha" `
    --mask_mha "D:\Workspace\TA3\data\DICOMM\output\pred_fdi_mask.mha" `
    --output_png "D:\Workspace\TA3\data\DICOMM\output\fdi_mpr.png"
```

---

## 4. FDI Tooth Mapping Schema

| Quadrant | FDI Range | Tooth Types | Key Research Pairs (M2 - M3) |
| :--- | :---: | :--- | :---: |
| **Q1 (Upper Right)** | 11 – 18 | Central Incisor (11) to Wisdom Tooth (18) | 17 & 18 |
| **Q2 (Upper Left)** | 21 – 28 | Central Incisor (21) to Wisdom Tooth (28) | 27 & 28 |
| **Q3 (Lower Left)** | 31 – 38 | Central Incisor (31) to Wisdom Tooth (38) | 37 & 38 |
| **Q4 (Lower Right)** | 41 – 48 | Central Incisor (41) to Wisdom Tooth (48) | 47 & 48 |
