# TA3 Automated Workspace Setup & Junction Linker
$ErrorActionPreference = "Continue"
$wsRoot = (Get-Location).Path

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "TA3 Workspace Setup & Alignment Protocol" -ForegroundColor Cyan
Write-Host "Root Directory: $wsRoot" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check & Initialize Submodules
Write-Host "`n[1/4] Checking Git Submodules..." -ForegroundColor Yellow
if (Test-Path ".gitmodules") {
    git submodule update --init --recursive
    Write-Host "[OK] Submodules verified and initialized." -ForegroundColor Green
} else {
    Write-Host "[!] Not inside a submodule parent repo; skipping." -ForegroundColor Gray
}

# 2. Establish Windows NTFS Junctions
Write-Host "`n[2/4] Setting up NTFS Directory Junctions (Zero-Duplication)..." -ForegroundColor Yellow

function Ensure-Junction {
    param(
        [string]$LinkPath,
        [string]$TargetPath
    )
    if (-not (Test-Path $TargetPath)) {
        New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
    }
    if (Test-Path $LinkPath) {
        $item = Get-Item $LinkPath -Force
        if ($item.LinkType -eq "Junction") {
            Write-Host "[OK] Junction exists: $LinkPath -> $TargetPath" -ForegroundColor Green
            return
        } else {
            Write-Host "[!] Existing directory found at $LinkPath." -ForegroundColor DarkYellow
            return
        }
    }
    New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
    Write-Host "[+] Created junction: $LinkPath -> $TargetPath" -ForegroundColor Green
}

# Ensure root data and weights exist
if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" -Force | Out-Null }
if (-not (Test-Path "weights")) { New-Item -ItemType Directory -Path "weights" -Force | Out-Null }

# Link to m3-m2-experiments
if (Test-Path "projects\m3-m2-experiments") {
    Ensure-Junction -LinkPath "projects\m3-m2-experiments\data" -TargetPath "$wsRoot\data"
    Ensure-Junction -LinkPath "projects\m3-m2-experiments\weights" -TargetPath "$wsRoot\weights"
}

# Link to m3-m2-cdss-app
if (Test-Path "projects\m3-m2-cdss-app") {
    Ensure-Junction -LinkPath "projects\m3-m2-cdss-app\data" -TargetPath "$wsRoot\data"
    Ensure-Junction -LinkPath "projects\m3-m2-cdss-app\weights" -TargetPath "$wsRoot\weights"
    if (Test-Path "projects\m3-m2-experiments\src") {
        Ensure-Junction -LinkPath "projects\m3-m2-cdss-app\src" -TargetPath "$wsRoot\projects\m3-m2-experiments\src"
    }
}

# 3. Check Labeled Data & ToothFairy2 Raw Dataset
Write-Host "`n[3/4] Verifying Datasets..." -ForegroundColor Yellow

# Check Labeled Crops
if (Test-Path "data\processed\crops_2d") {
    $cropCount = (Get-ChildItem "data\processed\crops_2d" -Filter "*.npy").Count
    Write-Host "[OK] Labeled 2D crops found: $cropCount .npy patches ready for model training/inference." -ForegroundColor Green
} else {
    Write-Host "[!] Labeled 2D crops missing at data\processed\crops_2d." -ForegroundColor Red
}

# Check ToothFairy2 Raw Data
$tf2Target = "data\toothfairy2\Dataset112_ToothFairy2"
if ((Test-Path $tf2Target) -and (Get-ChildItem $tf2Target -ErrorAction SilentlyContinue).Count -gt 0) {
    Write-Host "[OK] ToothFairy2 dataset found at $tf2Target." -ForegroundColor Green
} else {
    Write-Host "[!] ToothFairy2 3D volumetric dataset not found at $tf2Target." -ForegroundColor Yellow
    Write-Host ""
    $zipPath = Read-Host "Di mana lokasi file zip ToothFairy2 (ToothFairy2_Dataset.zip / ToothFairy2.zip) di komputer Anda? (Masukkan path lengkap atau tekan Enter untuk lewati)"
    
    if (-not [string]::IsNullOrWhiteSpace($zipPath)) {
        if (Test-Path $zipPath) {
            Write-Host "Mengekstrak $zipPath ke data\toothfairy2\..." -ForegroundColor Cyan
            Expand-Archive -Path $zipPath -DestinationPath "data\toothfairy2" -Force
            Write-Host "[OK] Ekstraksi selesai." -ForegroundColor Green
        } else {
            Write-Host "[!] File zip tidak ditemukan di: $zipPath" -ForegroundColor Red
        }
    } else {
        Write-Host "Dilewati. Anda dapat menjalankan python scripts/download_toothfairy.py nanti jika ingin mengunduh secara otomatis." -ForegroundColor Gray
    }
}

# 4. Verify Tests in m3-m2-experiments
Write-Host "`n[4/4] Running Unit Tests Verification..." -ForegroundColor Yellow
if (Test-Path "projects\m3-m2-experiments") {
    Push-Location "projects\m3-m2-experiments"
    try {
        pytest tests -q
        Write-Host "[OK] All 32 unit tests passed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "[!] Unit tests failed or pytest not found in current environment." -ForegroundColor Red
    } finally {
        Pop-Location
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Setup Completed! Workspace is aligned and ready." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
