# AGENTS.md — TA3 Workspace Guidelines & Operating Protocols

## 1. Workspace Identity & Purpose

This workspace (`D:\Workspace\TA3`) is the master monorepo workspace for the clinical research project:
* **Title:** Rancang Bangun Sistem Pendukung Keputusan Klinis Klasifikasi Relasi Gigi Molar Ketiga dan Kedua (M3-M2) pada Citra CBCT Menggunakan Mask-Guided Vision Transformer Ensemble dan Clinical Triage Gate
* **Author / Candidate:** M. Reza (NRP: 3126640060)
* **Advisor:** Nana Ramadijanti, S.Kom., M.Kom. (NIP: 197111091998022001)
* **Institution:** Program Studi Sarjana Terapan Teknik Informatika, PENS, 2025.

The workspace is organized into modular directories to cleanly decouple:
1. **Shared Volumetric Data** (`data/`),
2. **Academic Documents & Presentations** (`documents/`),
3. **Machine Learning & Geometry Research** (`projects/m3-m2-experiments/`),
4. **Clinical Decision Support Web App** (`projects/m3-m2-cdss-app/`).

---

## 2. Data Position & Zero-Duplication Architecture

> [!CAUTION]
> **Data Duplication Is Strictly Prohibited.**
> The root `data/` directory is the **single source of truth** (~126.5 GB). Never copy, move, duplicate, or re-download datasets into subprojects.

### Data Layout at Root (`D:\Workspace\TA3\data`)
* `data/raw/` (~90.4 GB): Raw 3D CBCT volumes (.nii.gz, DICOM).
* `data/toothfairy2/` (~36.0 GB): Benchmark 3D tooth segmentation dataset.
* `data/processed/` (~150 MB): Extracted 2D sagittal crops ($224 \times 224$), `metadata.csv`, and data split manifests.
* `data/masks/`, `data/manifests/`, `data/custom/`, `data/review/`.

### Shared Model Weights at Root (`D:\Workspace\TA3\weights`)
* `weights/` (~800 MB): Exported PyTorch model checkpoints (`swin_small_binary_best.pth`, `vit_base_binary_best.pth`, `stage2_dl_best.pth`, etc.).

### How Subprojects Access Shared Data & Weights
Subprojects in `projects/` access `data/` and `weights/` through two complementary mechanisms:
1. **Windows NTFS Directory Junctions (`da---l`):**
   - `projects/m3-m2-experiments/data` $\rightarrow$ `D:\Workspace\TA3\data`
   - `projects/m3-m2-experiments/weights` $\rightarrow$ `D:\Workspace\TA3\weights`
   - `projects/m3-m2-cdss-app/data` $\rightarrow$ `D:\Workspace\TA3\data`
   - `projects/m3-m2-cdss-app/weights` $\rightarrow$ `D:\Workspace\TA3\weights`
   - `projects/m3-m2-cdss-app/src` $\rightarrow$ `D:\Workspace\TA3\projects\m3-m2-experiments\src`
   *(Junction points operate at the filesystem level with 0 byte extra storage).*
2. **Dynamic Configuration Fallback:**
   - In `src/config.py`, path resolution checks the `TA3_DATA_DIR` environment variable, then local `data/` junction, and falls back to `../../data`.

---

## 3. Directory Layout & Roles

```text
D:\Workspace\TA3/
├── data/                               # [SHARED] Single source of truth for CBCT data (~126.5 GB)
├── weights/                            # [SHARED] Trained PyTorch checkpoints (~800 MB)
│
├── documents/                          # [REPO] Academic documents, proposals & presentations
│   ├── drafts/                         # Naskah PA / proposal (.docx, .pdf)
│   ├── presentations/                  # Seminar proposal slides (.pptx, .pdf)
│   ├── figures/                        # System diagrams, architecture & 3D visualizations
│   ├── references/                     # Base research papers & technical feasibility audits
│   └── slides_generator/               # Python scripts for PowerPoint generation & preview renders
│
└── projects/                           # [REPOS] Source code and software implementations
    │
    ├── m3-m2-experiments/              # [REPO] Research pipeline & model training
    │   ├── src/                        # Core library (loader, FDI filter, KD-Tree, ViT, CEJ, report)
    │   ├── scripts/                    # Research scripts (audit, CEJ annotation, training, eval)
    │   ├── notebooks/                  # Spatial analysis notebooks (.ipynb)
    │   ├── reports/                    # Benchmark metrics, confusion matrices, audit tables
    │   ├── tests/                      # Pytest test suite (32 unit tests)
    │   ├── pytest.ini                  # Config with pythonpath = .
    │   ├── AGENTS.md                   # Detailed research & clinical classification rules
    │   └── requirements.txt
    │
    └── m3-m2-cdss-app/                 # [REPO] Interactive Streamlit Clinical Web CDSS
        ├── streamlit_app.py            # Main Streamlit clinical application
        ├── components/                 # UI components and MPR slice viewer
        └── README.md
```

---

## 4. Agent Operating Protocols

### 1. Working Directory Awareness
When executing commands, **always navigate to the respective subproject folder**:
* For ML experiments, data processing, and tests:
  ```powershell
  cd D:\Workspace\TA3\projects\m3-m2-experiments
  pytest tests -v
  ```
* For Web CDSS development:
  ```powershell
  cd D:\Workspace\TA3\projects\m3-m2-cdss-app
  streamlit run streamlit_app.py
  ```

### 2. Multi-Repo Independence
Each folder inside `documents/`, `projects/m3-m2-experiments/`, and `projects/m3-m2-cdss-app/` is designed to be an independent Git repository if initialized:
* Never commit `data/`, `weights/`, `.venv*/`, or cache files to any repository.
* Ensure all `.gitignore` files remain active and enforced.

### 3. Detailed Research Rules
For all deep learning tasks, model architectures (Swin-S, ViT-B/16, CEJ classifier), KD-Tree surface distance ($D_{\min}$), clinical triage gate ($\tau = 0.70$), and evaluation metrics (Macro-F1, Macro-AUC, ESR, DTAC):
* Refer strictly to [`projects/m3-m2-experiments/AGENTS.md`](projects/m3-m2-experiments/AGENTS.md).

### 4. Integrity and Safety
* **No Fabricated Labels:** Never infer clinical labels (e.g. `Contact` vs `Resorption`) purely from geometric distance without clinical validation.
* **Test Verification:** Always run `pytest tests -v` in `projects/m3-m2-experiments` after any change to geometry or preprocessing code.
* **Preserve Documentation:** Do not delete academic files or reference materials in `documents/`.
