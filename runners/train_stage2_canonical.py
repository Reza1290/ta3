"""
train_stage2_canonical.py
=========================
Definitive Stage 2 Deep Learning Training on Canonical Anatomical Dataset:
1. Corrected physical coordinate polarity (True signed crown-apical distance).
2. Incorporates 50 visual inspection ground truth labels (case 542d Independent, etc.).
3. Targeted minority augmentation (At_CEJ 8x, Below_CEJ 3x) to achieve class parity.
4. Capacity-controlled ViT Tail (1 trainable block + Dropout 0.3 + Weight Decay 0.05).
5. Evaluates on clean, unaugmented validation split (N = 48).
6. Runs Stratified 5-Fold Cross Validation across the full development cohort (N = 408).

Completely isolated in `runners/` - zero modifications to submodule codebases.
"""

import os
import json
import random
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import timm
from scipy.ndimage import rotate, shift
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

SEED = 42

def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CLASS_NAMES = ["Above_CEJ", "At_CEJ", "Below_CEJ"]
CLASS_MAP = {1: 0, 2: 1, 3: 2}

class RegularizedViTTail(nn.Module):
    """Capacity-controlled ViT Tail: 1 trainable block + LayerNorm + Dropout + Linear Head."""
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

def preprocess_crop(arr: np.ndarray) -> torch.Tensor:
    arr = arr.astype(np.float32)
    mean = 0.485
    std = 0.229
    arr_norm = (arr - mean) / std
    t = torch.from_numpy(arr_norm).unsqueeze(0)
    t = t.repeat(3, 1, 1)
    return t

def augment_crop(arr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = arr.copy()
    angle = rng.uniform(-8.0, 8.0)
    out = rotate(out, angle, reshape=False, mode="nearest")
    shift_y = rng.uniform(-4.0, 4.0)
    shift_x = rng.uniform(-4.0, 4.0)
    out = shift(out, (shift_y, shift_x), mode="nearest")
    gamma = rng.uniform(0.9, 1.1)
    out = np.power(np.clip(out, 0.0, 1.0), gamma)
    return out

@torch.no_grad()
def extract_prefix_tokens_batched(backbone: nn.Module, images: torch.Tensor, device, batch_size: int = 32, trainable_blocks: int = 1) -> torch.Tensor:
    backbone.eval()
    all_tokens = []
    n = len(images)
    for i in range(0, n, batch_size):
        batch = images[i:i+batch_size].to(device)
        x = backbone.patch_embed(batch)
        x = backbone._pos_embed(x)
        x = backbone.patch_drop(x)
        x = backbone.norm_pre(x)
        for block in list(backbone.blocks.children())[:-trainable_blocks]:
            x = block(x)
        all_tokens.append(x.cpu())
    return torch.cat(all_tokens, dim=0)

def calculate_balanced_weights(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def train_one_split(
    train_tokens: torch.Tensor,
    train_targets: torch.Tensor,
    val_tokens: torch.Tensor,
    val_targets: torch.Tensor,
    backbone: nn.Module,
    device,
    epochs: int = 15,
    batch_size: int = 16,
    trainable_blocks: int = 1,
    patience: int = 4
):
    tail = RegularizedViTTail(backbone, trainable_blocks=trainable_blocks, dropout=0.3, num_classes=3).to(device)
    class_weights = calculate_balanced_weights(train_targets.tolist(), num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(tail.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_loader = DataLoader(TensorDataset(train_tokens, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tokens, val_targets), batch_size=batch_size, shuffle=False)

    best_val_f1 = -1.0
    best_model_state = None
    best_epoch = -1
    history = []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tail.train()
        train_loss = 0.0
        train_preds, train_trues = [], []

        for b_tokens, b_targets in train_loader:
            b_tokens, b_targets = b_tokens.to(device), b_targets.to(device)
            optimizer.zero_grad()
            logits = tail(b_tokens)
            loss = criterion(logits, b_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(b_targets)
            train_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            train_trues.extend(b_targets.cpu().tolist())

        scheduler.step()
        train_loss /= len(train_targets)
        train_f1 = f1_score(train_trues, train_preds, average="macro", zero_division=0)

        # Validation
        tail.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for b_tokens, b_targets in val_loader:
                b_tokens, b_targets = b_tokens.to(device), b_targets.to(device)
                logits = tail(b_tokens)
                loss = criterion(logits, b_targets)
                val_loss += loss.item() * len(b_targets)
                val_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                val_trues.extend(b_targets.cpu().tolist())

        val_loss /= len(val_targets)
        val_f1 = f1_score(val_trues, val_preds, average="macro", zero_division=0)
        val_acc = (np.array(val_preds) == np.array(val_trues)).mean()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(tail.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    tail.load_state_dict(best_model_state)
    tail.eval()
    with torch.no_grad():
        val_logits = tail(val_tokens.to(device)).cpu()
        val_probs = torch.softmax(val_logits, dim=-1).numpy()
        val_preds = val_logits.argmax(dim=-1).numpy()

    return {
        "tail": tail,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
        "val_preds": val_preds,
        "val_probs": val_probs
    }

def main():
    seed_all(SEED)
    rng = np.random.RandomState(SEED)

    root_dir = Path("D:/Workspace/TA3")
    labels_csv = root_dir / "data/custom/gemini-3.8/stage2_canonical_labels.csv"
    crops_dir = root_dir / "data/processed/crops_2d"
    base_weights_path = root_dir / "weights/vit_base_binary_best.pth"
    out_dir = root_dir / "runners/outputs/canonical_stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2 CANONICAL DEEP LEARNING RUNNER")
    print("Training on Anatomically Corrected Development Cohort (N = 408)")
    print("=" * 70)

    df = pd.read_csv(labels_csv)
    df["target"] = df["class_id"].astype(int).map(CLASS_MAP)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    print(f"Dataset: Train={len(train_df)}, Val={len(val_df)}")
    print(f"Train Class Distribution: Above_CEJ={sum(train_df['target']==0)}, At_CEJ={sum(train_df['target']==1)}, Below_CEJ={sum(train_df['target']==2)}")
    print(f"Val Class Distribution:   Above_CEJ={sum(val_df['target']==0)}, At_CEJ={sum(val_df['target']==1)}, Below_CEJ={sum(val_df['target']==2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution device: {device}")

    # Load ViT Backbone
    print("Loading ViT-B/16 backbone...")
    backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    state_dict = torch.load(base_weights_path, map_location="cpu")
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)

    trainable_blocks = 1

    # Pre-load validation crops
    val_tensors, val_targets, val_sids = [], [], []
    for _, r in val_df.iterrows():
        sid = r["sample_id"]
        arr = np.load(crops_dir / f"{sid}.npy")
        val_tensors.append(preprocess_crop(arr))
        val_targets.append(r["target"])
        val_sids.append(sid)
    val_images = torch.stack(val_tensors, dim=0)
    val_targets = torch.tensor(val_targets, dtype=torch.long)

    # Pre-load and augment train crops
    # Minority augmentation: At_CEJ (target=1) 7x (total 8x = 112 samples)
    # Below_CEJ (target=2) 2x (total 3x = 111 samples)
    # Above_CEJ (target=0) 1x (total 309 samples)
    train_tensors, train_targets, train_sids = [], [], []
    for _, r in train_df.iterrows():
        sid = r["sample_id"]
        target = r["target"]
        raw_arr = np.load(crops_dir / f"{sid}.npy")

        train_tensors.append(preprocess_crop(raw_arr))
        train_targets.append(target)
        train_sids.append(sid)

        num_augs = 7 if target == 1 else (2 if target == 2 else 0)
        for a_idx in range(num_augs):
            aug_arr = augment_crop(raw_arr, rng)
            train_tensors.append(preprocess_crop(aug_arr))
            train_targets.append(target)
            train_sids.append(f"{sid}_aug{a_idx+1}")

    train_images = torch.stack(train_tensors, dim=0)
    train_targets = torch.tensor(train_targets, dtype=torch.long)

    print(f"\nAugmented Train Cohort: {len(train_images)} samples (Above={sum(train_targets==0).item()}, At={sum(train_targets==1).item()}, Below={sum(train_targets==2).item()})")

    # Token Extraction
    print(f"Extracting tokens (freezing first {12 - trainable_blocks} blocks, {trainable_blocks} in tail)...")
    train_tokens = extract_prefix_tokens_batched(backbone, train_images, device, batch_size=32, trainable_blocks=trainable_blocks)
    val_tokens = extract_prefix_tokens_batched(backbone, val_images, device, batch_size=32, trainable_blocks=trainable_blocks)
    print(f"Tokens cached: Train={train_tokens.shape}, Val={val_tokens.shape}")

    # Train Standard Split
    print("\n--- Training on Standard Development Split ---")
    res = train_one_split(
        train_tokens, train_targets, val_tokens, val_targets,
        backbone, device, epochs=15, batch_size=16, trainable_blocks=trainable_blocks, patience=4
    )

    for h in res["history"]:
        print(f"Epoch {h['epoch']:02d} | Train Loss: {h['train_loss']:.4f}, Train F1: {h['train_macro_f1']:.4f} | "
              f"Val Loss: {h['val_loss']:.4f}, Val Acc: {h['val_acc']:.4f}, Val F1: {h['val_macro_f1']:.4f}")

    best_weights_path = out_dir / "canonical_stage2_best.pth"
    torch.save(res["tail"].state_dict(), best_weights_path)
    print(f"\nSaved canonical best weights: {best_weights_path}")

    # Standard Split Evaluation
    val_preds = res["val_preds"]
    val_probs = res["val_probs"]
    val_trues = val_targets.numpy()

    report_dict = classification_report(val_trues, val_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(val_trues, val_preds, labels=[0, 1, 2])
    macro_f1 = f1_score(val_trues, val_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(val_trues, val_preds)

    print("\n" + "=" * 50)
    print("STANDARD SPLIT VALIDATION EVALUATION (CANONICAL)")
    print("=" * 50)
    print(classification_report(val_trues, val_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print(cm)
    print(f"Balanced Accuracy: {bal_acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")

    # --- Run Stratified 5-Fold Cross Validation Across Entire Development Pool (N = 408) ---
    print("\n" + "=" * 70)
    print("STRATIFIED 5-FOLD CROSS VALIDATION (DEVELOPMENT COHORT, N = 408)")
    print("=" * 70)

    # Extract all unaugmented tokens for all 408 cases
    all_df = df.copy().reset_index(drop=True)
    all_tensors, all_targets_list, all_sids_list = [], [], []
    for _, r in all_df.iterrows():
        sid = r["sample_id"]
        arr = np.load(crops_dir / f"{sid}.npy")
        all_tensors.append(preprocess_crop(arr))
        all_targets_list.append(r["target"])
        all_sids_list.append(sid)
    all_images = torch.stack(all_tensors, dim=0)
    all_targets = np.array(all_targets_list)

    all_raw_tokens = extract_prefix_tokens_batched(backbone, all_images, device, batch_size=32, trainable_blocks=trainable_blocks)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_f1s = []
    fold_accs = []
    oof_preds = np.zeros(len(all_df), dtype=int)
    oof_probs = np.zeros((len(all_df), 3), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_raw_tokens, all_targets), 1):
        # Minority augmentation in fold train set
        f_train_tokens = [all_raw_tokens[i] for i in train_idx]
        f_train_targets = [all_targets[i] for i in train_idx]

        aug_tokens_list = list(f_train_tokens)
        aug_targets_list = list(f_train_targets)

        # Augment in fold
        for i, tidx in enumerate(train_idx):
            tgt = all_targets[tidx]
            sid = all_sids_list[tidx]
            num_augs = 7 if tgt == 1 else (2 if tgt == 2 else 0)
            if num_augs > 0:
                raw_arr = np.load(crops_dir / f"{sid}.npy")
                for _ in range(num_augs):
                    aug_arr = augment_crop(raw_arr, rng)
                    aug_t = preprocess_crop(aug_arr).unsqueeze(0)
                    with torch.no_grad():
                        tok = extract_prefix_tokens_batched(backbone, aug_t, device, batch_size=1, trainable_blocks=trainable_blocks)[0]
                    aug_tokens_list.append(tok)
                    aug_targets_list.append(tgt)

        f_tr_tok = torch.stack(aug_tokens_list, dim=0)
        f_tr_tgt = torch.tensor(aug_targets_list, dtype=torch.long)
        f_val_tok = all_raw_tokens[val_idx]
        f_val_tgt = torch.tensor(all_targets[val_idx], dtype=torch.long)

        fold_res = train_one_split(
            f_tr_tok, f_tr_tgt, f_val_tok, f_val_tgt,
            backbone, device, epochs=15, batch_size=16, trainable_blocks=trainable_blocks, patience=4
        )

        f_preds = fold_res["val_preds"]
        oof_preds[val_idx] = f_preds
        oof_probs[val_idx] = fold_res["val_probs"]

        f_f1 = f1_score(all_targets[val_idx], f_preds, average="macro", zero_division=0)
        f_acc = balanced_accuracy_score(all_targets[val_idx], f_preds)
        fold_f1s.append(f_f1)
        fold_accs.append(f_acc)

        print(f"Fold {fold}/5: Val Balanced Acc = {f_acc*100:.2f}%, Val Macro-F1 = {f_f1:.4f}")

    mean_cv_f1 = np.mean(fold_f1s)
    std_cv_f1 = np.std(fold_f1s)
    mean_cv_acc = np.mean(fold_accs)
    std_cv_acc = np.std(fold_accs)

    oof_report = classification_report(all_targets, oof_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    oof_cm = confusion_matrix(all_targets, oof_preds, labels=[0, 1, 2])
    oof_macro_f1 = f1_score(all_targets, oof_preds, average="macro", zero_division=0)
    oof_bal_acc = balanced_accuracy_score(all_targets, oof_preds)

    print("\n" + "=" * 70)
    print("5-FOLD CROSS VALIDATION SUMMARY (OUT-OF-FOLD, N = 408)")
    print("=" * 70)
    print(f"5-Fold CV Macro-F1:        {mean_cv_f1:.4f} +/- {std_cv_f1:.4f}")
    print(f"5-Fold CV Balanced Acc:    {mean_cv_acc*100:.2f}% +/- {std_cv_acc*100:.2f}%")
    print(f"Out-of-Fold Macro-F1:      {oof_macro_f1:.4f}")
    print(f"Out-of-Fold Balanced Acc:  {oof_bal_acc*100:.2f}%")
    print("\nOut-of-Fold Classification Report:")
    print(classification_report(all_targets, oof_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Out-of-Fold Confusion Matrix:")
    print(oof_cm)

    # Save complete metrics json
    metrics_data = {
        "standard_split": {
            "train_samples_augmented": len(train_targets),
            "val_samples": len(val_targets),
            "macro_f1": round(float(macro_f1), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "confusion_matrix": cm.tolist(),
            "per_class": {
                cname: {
                    "precision": round(float(report_dict[cname]["precision"]), 4),
                    "recall": round(float(report_dict[cname]["recall"]), 4),
                    "f1_score": round(float(report_dict[cname]["f1-score"]), 4),
                    "support": int(report_dict[cname]["support"])
                }
                for cname in CLASS_NAMES
            }
        },
        "five_fold_cv": {
            "mean_macro_f1": round(float(mean_cv_f1), 4),
            "std_macro_f1": round(float(std_cv_f1), 4),
            "mean_balanced_accuracy": round(float(mean_cv_acc), 4),
            "std_balanced_accuracy": round(float(std_cv_acc), 4),
            "oof_macro_f1": round(float(oof_macro_f1), 4),
            "oof_balanced_accuracy": round(float(oof_bal_acc), 4),
            "oof_confusion_matrix": oof_cm.tolist(),
            "oof_per_class": {
                cname: {
                    "precision": round(float(oof_report[cname]["precision"]), 4),
                    "recall": round(float(oof_report[cname]["recall"]), 4),
                    "f1_score": round(float(oof_report[cname]["f1-score"]), 4),
                    "support": int(oof_report[cname]["support"])
                }
                for cname in CLASS_NAMES
            }
        }
    }

    metrics_file = out_dir / "canonical_stage2_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved metrics: {metrics_file}")

    # Generate Markdown Report
    report_md = f"""# CANONICAL STAGE 2 CEJ DEEP LEARNING BENCHMARK REPORT
## Anatomically Corrected Development Cohort (N = 408 Contact Cases)

* **Architecture:** Vision Transformer ViT-B/16 (`vit_base_patch16_224`), Capacity-Controlled 1 Trainable Block + Dropout 0.3
* **Dataset:** Canonical Anatomical Ground Truth (`data/custom/gemini-3.8/stage2_canonical_labels.csv`)
* **Standard Split:** 360 Train (532 Augmented) / 48 Validation
* **Cross-Validation:** Stratified 5-Fold Cross Validation across all 408 Development Cases

---

### 1. Standard Validation Split Benchmark (N = 48)

| Metric | Score |
| :--- | :---: |
| **Overall Accuracy** | **{(np.array(val_preds) == val_trues).mean()*100:.2f}%** |
| **Balanced Accuracy** | **{bal_acc*100:.2f}%** |
| **Three-Class Macro-F1** | **{macro_f1:.4f}** |

#### Per-Class Performance:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Above CEJ** | {report_dict['Above_CEJ']['precision']:.4f} | {report_dict['Above_CEJ']['recall']:.4f} | {report_dict['Above_CEJ']['f1-score']:.4f} | {report_dict['Above_CEJ']['support']} |
| **At CEJ** | {report_dict['At_CEJ']['precision']:.4f} | {report_dict['At_CEJ']['recall']:.4f} | {report_dict['At_CEJ']['f1-score']:.4f} | {report_dict['At_CEJ']['support']} |
| **Below CEJ** | {report_dict['Below_CEJ']['precision']:.4f} | {report_dict['Below_CEJ']['recall']:.4f} | {report_dict['Below_CEJ']['f1-score']:.4f} | {report_dict['Below_CEJ']['support']} |
| **Macro Average** | **{report_dict['macro avg']['precision']:.4f}** | **{report_dict['macro avg']['recall']:.4f}** | **{macro_f1:.4f}** | {len(val_trues)} |

```text
Confusion Matrix (Standard Split):
Actual \\ Predicted   Above_CEJ   At_CEJ   Below_CEJ
Above_CEJ                {cm[0, 0]}          {cm[0, 1]}        {cm[0, 2]}
At_CEJ                   {cm[1, 0]}          {cm[1, 1]}        {cm[1, 2]}
Below_CEJ                {cm[2, 0]}          {cm[2, 1]}        {cm[2, 2]}
```

---

### 2. Stratified 5-Fold Cross Validation Benchmark (All N = 408 Cases)

| Metric | Cross-Validation Mean +/- Std | Out-of-Fold Total |
| :--- | :---: | :---: |
| **Macro-F1 (3 Classes)** | **{mean_cv_f1:.4f} +/- {std_cv_f1:.4f}** | **{oof_macro_f1:.4f}** |
| **Balanced Accuracy** | **{mean_cv_acc*100:.2f}% +/- {std_cv_acc*100:.2f}%** | **{oof_bal_acc*100:.2f}%** |

#### Out-of-Fold Per-Class Performance (N = 408):
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Above CEJ** | {oof_report['Above_CEJ']['precision']:.4f} | {oof_report['Above_CEJ']['recall']:.4f} | {oof_report['Above_CEJ']['f1-score']:.4f} | {oof_report['Above_CEJ']['support']} |
| **At CEJ** | {oof_report['At_CEJ']['precision']:.4f} | {oof_report['At_CEJ']['recall']:.4f} | {oof_report['At_CEJ']['f1-score']:.4f} | {oof_report['At_CEJ']['support']} |
| **Below CEJ** | {oof_report['Below_CEJ']['precision']:.4f} | {oof_report['Below_CEJ']['recall']:.4f} | {oof_report['Below_CEJ']['f1-score']:.4f} | {oof_report['Below_CEJ']['support']} |
| **Macro Average** | **{oof_report['macro avg']['precision']:.4f}** | **{oof_report['macro avg']['recall']:.4f}** | **{oof_macro_f1:.4f}** | {len(all_targets)} |

```text
Confusion Matrix (Out-of-Fold, N = 408):
Actual \\ Predicted   Above_CEJ   At_CEJ   Below_CEJ
Above_CEJ                {oof_cm[0, 0]}          {oof_cm[0, 1]}        {oof_cm[0, 2]}
At_CEJ                   {oof_cm[1, 0]}          {oof_cm[1, 1]}        {oof_cm[1, 2]}
Below_CEJ                {oof_cm[2, 0]}          {oof_cm[2, 1]}        {oof_cm[2, 2]}
```
"""
    report_file = out_dir / "REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved benchmark report: {report_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
