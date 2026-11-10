# AGENTS.md — TA3 Workspace Guidelines & Operating Protocols

## 1. Workspace Identity & Purpose

This workspace (`D:\Workspace\TA3` or `<workspace-root>`) is the master monorepo workspace for the clinical research project:
* **Title:** Rancang Bangun Sistem Pendukung Keputusan Klinis Klasifikasi Relasi Gigi Molar Ketiga dan Kedua (M3-M2) pada Citra CBCT Menggunakan Mask-Guided Vision Transformer Ensemble dan Clinical Triage Gate
* **Author / Candidate:** M. Reza (NRP: 3126640060)
* **Advisor:** Nana Ramadijanti, S.Kom., M.Kom. (NIP: 197111091998022001)
* **Institution:** Program Studi Sarjana Terapan Teknik Informatika, PENS, 2025.

The workspace is organized into modular directories to cleanly decouple:
1. **Shared Volumetric Data** (`data/`),
2. **Academic Documents & Presentations** (`documents/`),
3. **Machine Learning & Geometry Research** (`projects/m3-m2-experiments/` and `projects/m3-m2-rebuild/`),
4. **Clinical Decision Support Web App** (`projects/m3-m2-cdss-app/`).

---

## 2. Workspace Setup & Exact File Alignment Protocol

When starting in a fresh environment or cloning this workspace to another machine, **agents MUST ensure file positions match this exact workspace architecture**:

### Step 1: Clone Repository with Submodules
```bash
git clone --recurse-submodules git@github.com:Reza1290/ta3.git
cd ta3
```

### Step 2: Establish NTFS Directory Junctions
To ensure zero data duplication while keeping local paths identical in every subproject, run:
```powershell
# Set root dynamically based on current working directory
$wsRoot = (Get-Location).Path

# Link data and weights to m3-m2-experiments
New-Item -ItemType Junction -Path "projects\m3-m2-experiments\data" -Target "$wsRoot\data"
New-Item -ItemType Junction -Path "projects\m3-m2-experiments\weights" -Target "$wsRoot\weights"

# Link data, weights, and src to m3-m2-cdss-app
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\data" -Target "$wsRoot\data"
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\weights" -Target "$wsRoot\weights"
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\src" -Target "$wsRoot\projects\m3-m2-experiments\src"
```
*(Alternatively, simply run `.\setup_workspace.ps1` at the root).*

---

## 3. Data Architecture & Zero-Duplication Policy

> [!CAUTION]
> **Data Duplication Is Strictly Prohibited.**
> The root `data/` directory is the **single source of truth** (~126.5 GB). Never copy, move, duplicate, or re-download datasets into subprojects.

### Data Layout at Root (`data/`)
* `data/processed/` (~150 MB): **[TRACKED IN GIT]** Extracted 2D sagittal crops ($224 \times 224$), `crops_2d/*.png`, and ground-truth label CSVs (`m2_m3_cej_4class_labels.csv`, `toothfairy_binary_labels.csv`, `toothfairy_classified_labels.csv`).
* `data/manifests/` (< 1 MB): **[TRACKED IN GIT]** Master spatial metadata `dataset_4class_manifest.csv` and patient splits `patient_splits_seed42.json`.
* `data/custom/` (~40 MB): **[TRACKED IN GIT]** Custom classification taxonomies.
* `data/raw/` (~90.4 GB): **[IGNORED IN GIT]** Raw 3D CBCT volumes (.nii.gz, DICOM).
* `data/toothfairy2/` (~36.0 GB): **[IGNORED IN GIT]** Benchmark 3D tooth segmentation dataset.

### Labeled Data Readiness
Because `data/processed/` and `data/manifests/` are tracked directly in Git, **model training (Stage 1 & Stage 2) and evaluation can run immediately out of the box** without needing the heavy 3D volumetric raw files!

---

## 4. ToothFairy2 Raw Dataset Discovery Protocol (MANDATORY AGENT RULE)

> [!IMPORTANT]
> **DO NOT SILENTLY DOWNLOAD 24–36 GB OVER THE INTERNET.**
> When setting up the workspace or executing any task that requires raw 3D CBCT volumes or ToothFairy2 3D segmentation labels (`data/toothfairy2/Dataset112_ToothFairy2/` or `data/raw/`):
> 
> 1. The agent **MUST FIRST check** if `data/toothfairy2/Dataset112_ToothFairy2/` exists and contains data.
> 2. **IF NOT FOUND: THE AGENT MUST STOP AND ASK THE USER:**
>    > *"File dataset mentah ToothFairy2 belum ditemukan di folder `data/toothfairy2/`. Di mana lokasi file zip ToothFairy2 (`ToothFairy2_Dataset.zip` atau `ToothFairy2.zip`) tersimpan di storage/drive komputer Anda?"*
> 3. Once the user provides the local zip path (e.g. `D:\Downloads\ToothFairy2.zip` or an external drive), the agent should extract it into `data/toothfairy2/`.
> 4. Only if the user explicitly instructs the agent that they do not have the zip locally and wish to download it, may the agent invoke `scripts/download_toothfairy.py`.

---

## 5. Directory Layout & Roles

```text
D:\Workspace\TA3/
├── data/                               # [SHARED] Single source of truth for CBCT data
│   ├── processed/                      # [GIT TRACKED] 2D crops (224x224) & label CSVs
│   ├── manifests/                      # [GIT TRACKED] Master manifest & patient splits
│   ├── custom/                         # [GIT TRACKED] Custom classification taxonomies
│   ├── raw/                            # [LOCAL ONLY] Raw 3D CBCT volumes (~90.4 GB)
│   └── toothfairy2/                    # [LOCAL ONLY] 3D ToothFairy2 segmentation (~36.0 GB)
│
├── weights/                            # [SHARED] Exported PyTorch checkpoints (~800 MB)
│
├── documents/                          # [SUBMODULE: Reza1290/m3-m2-documents]
│   ├── drafts/                         # Naskah PA / proposal (.docx, .pdf)
│   ├── presentations/                  # Seminar proposal slides (.pptx, .pdf)
│   ├── figures/                        # System diagrams, architecture & 3D visualizations
│   ├── references/                     # Base research papers & technical feasibility audits
│   └── slides_generator/               # Python scripts for PowerPoint generation
│
└── projects/                           # [SUBMODULE REPOSITORIES]
    ├── m3-m2-experiments/              # [SUBMODULE: Reza1290/m3-m2-experiments]
    │   ├── src/                        # Core library (loader, FDI filter, KD-Tree, ViT, CEJ, report)
    │   ├── scripts/                    # Research scripts (audit, CEJ annotation, training, eval)
    │   ├── notebooks/                  # Spatial analysis notebooks (.ipynb)
    │   ├── reports/                    # Benchmark metrics, confusion matrices, audit tables
    │   ├── tests/                      # Pytest test suite (32 unit tests passing)
    │   ├── pytest.ini                  # Config with pythonpath = .
    │   └── AGENTS.md                   # Detailed research & clinical classification rules
    │
    ├── m3-m2-cdss-app/                 # [SUBMODULE: Reza1290/m3-m2-cdss-app]
    │   ├── streamlit_app.py            # Main Streamlit clinical application
    │   ├── components/                 # UI components and MPR slice viewer
    │   └── README.md
    │
    └── m3-m2-rebuild/                  # [SUBMODULE: Reza1290/m3-m2-rebuild]
        └── ...                         # Independent development pipeline & physical geometry
```

---

## 6. Agent Operating Protocols

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
Each subfolder (`documents/`, `projects/m3-m2-experiments/`, `projects/m3-m2-cdss-app/`, `projects/m3-m2-rebuild/`) is an independent Git repository tracked via Git Submodules in the parent repo:
* When changing code in a subproject, commit and push inside that subproject first.
* Then commit the updated submodule commit pointer in the root workspace.

### 3. Detailed Research Rules
For all deep learning tasks, model architectures (Swin-S, ViT-B/16, CEJ classifier), KD-Tree surface distance ($D_{\min}$), clinical triage gate ($\tau = 0.70$), and evaluation metrics (Macro-F1, Macro-AUC, ESR, DTAC):
* Refer strictly to [`projects/m3-m2-experiments/AGENTS.md`](projects/m3-m2-experiments/AGENTS.md).

### 4. Integrity and Safety
* **No Fabricated Labels:** Never infer clinical labels purely from geometric distance without clinical validation.
* **Test Verification:** Always run `pytest tests -v` in `projects/m3-m2-experiments` after any change to geometry or preprocessing code.
* **Preserve Documentation:** Do not delete academic files or reference materials in `documents/`.
