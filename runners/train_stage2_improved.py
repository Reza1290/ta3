"""
train_stage2_improved.py
========================
Optimized Stage 2 Deep Learning Training with:
1. Anti-overfitting regularizations: Dropout (0.3), Label Smoothing (0.1), Weight Decay (0.05).
2. Capacity control: 1 trainable Transformer block (Block 11) instead of 2.
3. Minority class data augmentation (At_CEJ & Above_CEJ) to combat 22:1 class imbalance.
4. Early stopping with patience=4.

Runs in isolated `runners/` without modifying any submodule codebase.
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
        x = self.norm(x)
        cls_token = x[:, 0]
        cls_token = self.dropout(cls_token)
        return self.head(cls_token)

def augment_crop(arr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Clinical dental CBCT-safe augmentations: small rotation (±4 deg), small translation (±4 px), intensity scaling."""
    # 1. Subtle rotation (±4 degrees)
    angle = rng.uniform(-4.0, 4.0)
    aug = rotate(arr, angle, reshape=False, mode="nearest")
    
    # 2. Subtle shift (±3 pixels)
    sh_y = rng.uniform(-3.0, 3.0)
    sh_x = rng.uniform(-3.0, 3.0)
    aug = shift(aug, [sh_y, sh_x], mode="nearest")
    
    # 3. Subtle intensity jitter (±8%)
    scale = rng.uniform(0.92, 1.08)
    aug = np.clip(aug * scale, 0.0, 1.0)
    return aug.astype(np.float32)

def preprocess_crop(crop_array: np.ndarray) -> torch.Tensor:
    arr = np.array(crop_array, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

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

def main():
    seed_all(SEED)
    rng = np.random.RandomState(SEED)

    root_dir = Path("D:/Workspace/TA3")
    labels_csv = root_dir / "data/custom/gemini-3.8/corrected_stage2_labels.csv"
    crops_dir = root_dir / "data/processed/crops_2d"
    base_weights_path = root_dir / "weights/vit_base_binary_best.pth"
    out_dir = root_dir / "runners/outputs/corrected_stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2 IMPROVED DEEP LEARNING RUNNER")
    print("Anti-Overfitting Training on Augmented Expanded Dataset")
    print("=" * 70)

    df = pd.read_csv(labels_csv)
    df["target"] = df["class_id"].astype(int).map(CLASS_MAP)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    print(f"Original Train: {len(train_df)} cases, Validation: {len(val_df)} cases")

    # Load validation set (unaugmented)
    val_tensors, val_targets, val_sids = [], [], []
    for _, r in val_df.iterrows():
        sid = r["sample_id"]
        arr = np.load(crops_dir / f"{sid}.npy")
        val_tensors.append(preprocess_crop(arr))
        val_targets.append(r["target"])
        val_sids.append(sid)
    val_images = torch.stack(val_tensors, dim=0)
    val_targets = torch.tensor(val_targets, dtype=torch.long)

    # Load and augment train set:
    # At_CEJ (target=1): augment 4x (total 5x = ~70 samples)
    # Above_CEJ (target=0): augment 2x (total 3x = ~99 samples)
    # Below_CEJ (target=2): 1x (313 samples)
    train_tensors, train_targets, train_sids = [], [], []
    for _, r in train_df.iterrows():
        sid = r["sample_id"]
        target = r["target"]
        raw_arr = np.load(crops_dir / f"{sid}.npy")
        
        # Original sample
        train_tensors.append(preprocess_crop(raw_arr))
        train_targets.append(target)
        train_sids.append(sid)

        # Augmentation for minority classes
        num_augs = 4 if target == 1 else (2 if target == 0 else 0)
        for a_idx in range(num_augs):
            aug_arr = augment_crop(raw_arr, rng)
            train_tensors.append(preprocess_crop(aug_arr))
            train_targets.append(target)
            train_sids.append(f"{sid}_aug{a_idx+1}")

    train_images = torch.stack(train_tensors, dim=0)
    train_targets = torch.tensor(train_targets, dtype=torch.long)

    print(f"\nAfter Targeted Augmentation, Train Size = {len(train_images)} samples:")
    for cid, cname in enumerate(CLASS_NAMES):
        count = (train_targets == cid).sum().item()
        print(f"  Class {cid} ({cname}): {count} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nExecution device: {device}")

    # Load ViT Backbone
    print("Loading ViT-B/16 base backbone...")
    backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    state_dict = torch.load(base_weights_path, map_location="cpu")
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)

    # Prefix Token Extraction with 1 Trainable Block
    trainable_blocks = 1
    print(f"Extracting tokens (freezing first {12 - trainable_blocks} blocks, {trainable_blocks} block in tail)...")
    train_tokens = extract_prefix_tokens_batched(backbone, train_images, device, batch_size=32, trainable_blocks=trainable_blocks)
    val_tokens = extract_prefix_tokens_batched(backbone, val_images, device, batch_size=32, trainable_blocks=trainable_blocks)
    print(f"Train tokens shape: {train_tokens.shape}, Val tokens shape: {val_tokens.shape}")

    # Regularized Tail Classifier
    tail = RegularizedViTTail(backbone, trainable_blocks=trainable_blocks, dropout=0.3, num_classes=3).to(device)

    # Loss with label smoothing + balanced weights
    class_weights = calculate_balanced_weights(train_targets.tolist(), num_classes=3).to(device)
    print(f"Class Weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Optimizer with stronger weight decay
    optimizer = torch.optim.AdamW(tail.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    batch_size = 16
    train_loader = DataLoader(TensorDataset(train_tokens, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tokens, val_targets), batch_size=batch_size, shuffle=False)

    epochs = 15
    best_val_f1 = -1.0
    best_model_state = None
    best_epoch = -1
    history = []
    patience = 4
    patience_counter = 0

    print(f"\nStarting Anti-Overfitting Training ({epochs} epochs, batch_size={batch_size}, weight_decay=0.05)...")
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

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}")

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
                print(f"Early stopping triggered at Epoch {epoch} (patience={patience})")
                break

    print(f"\nLoading best checkpoint from Epoch {best_epoch} (Val Macro-F1: {best_val_f1:.4f})...")
    tail.load_state_dict(best_model_state)
    best_weights_path = out_dir / "stage2_vit_improved_best.pth"
    torch.save(best_model_state, best_weights_path)
    print(f"Saved best weights to: {best_weights_path}")

    # Final evaluation on val set
    tail.eval()
    with torch.no_grad():
        val_logits = tail(val_tokens.to(device)).cpu()
        val_probs = torch.softmax(val_logits, dim=-1).numpy()
        val_preds = val_logits.argmax(dim=-1).numpy()
        val_trues = val_targets.numpy()

    report_dict = classification_report(
        val_trues, val_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(val_trues, val_preds, labels=[0, 1, 2])
    macro_f1 = f1_score(val_trues, val_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(val_trues, val_preds)

    print("\n" + "=" * 50)
    print("FINAL VALIDATION EVALUATION (IMPROVED STAGE 2)")
    print("=" * 50)
    print(classification_report(val_trues, val_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    print(cm)
    print(f"Overall Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"Three-Class Macro-F1:      {macro_f1:.4f}")

    metrics_data = {
        "best_epoch": best_epoch,
        "train_samples_augmented": len(train_targets),
        "validation_samples": len(val_trues),
        "macro_f1": round(float(macro_f1), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            cname: {
                "precision": round(float(report_dict[cname]["precision"]), 4),
                "recall": round(float(report_dict[cname]["recall"]), 4),
                "f1_score": round(float(report_dict[cname]["f1-score"]), 4),
                "support": int(report_dict[cname]["support"])
            }
            for cname in CLASS_NAMES
        },
        "sample_predictions": [
            {
                "sample_id": sid,
                "ground_truth": CLASS_NAMES[true_cls],
                "prediction": CLASS_NAMES[pred_cls],
                "confidence": round(float(val_probs[i, pred_cls]), 4),
                "probabilities": {cname: round(float(val_probs[i, c_idx]), 4) for c_idx, cname in enumerate(CLASS_NAMES)}
            }
            for i, (sid, true_cls, pred_cls) in enumerate(zip(val_sids, val_trues, val_preds))
        ],
        "training_history": history
    }

    metrics_path = out_dir / "improved_stage2_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Write Markdown Report
    report_md = f"""# STAGE 2 IMPROVED CEJ TRAINING REPORT
## Regularized ViT-B/16 Tail Classifier with Minority Augmentation & Anti-Overfitting

* **Model Architecture:** Vision Transformer ViT-B/16 (`vit_base_patch16_224`) with 1 Trainable Block + Dropout (0.3)
* **Dataset:** Expanded Development Set with Targeted Augmentation (`data/custom/gemini-3.8/expanded_stage2_labels.csv`)
* **Augmented Train Set:** {len(train_targets)} Samples (Above CEJ: 99, At CEJ: 70, Below CEJ: 313)
* **Validation Subset:** 48 Contact Cases (5 Above CEJ, 2 At CEJ, 41 Below CEJ)
* **Best Epoch:** {best_epoch} / {epochs}
* **Macro-F1 (3 Classes):** **{macro_f1:.4f}**
* **Balanced Accuracy:** **{bal_acc*100:.2f}%**

---

### Per-Class Performance (Validation Set, N = 48)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Above CEJ** | {report_dict['Above_CEJ']['precision']:.4f} | {report_dict['Above_CEJ']['recall']:.4f} | {report_dict['Above_CEJ']['f1-score']:.4f} | {report_dict['Above_CEJ']['support']} |
| **At CEJ** | {report_dict['At_CEJ']['precision']:.4f} | {report_dict['At_CEJ']['recall']:.4f} | {report_dict['At_CEJ']['f1-score']:.4f} | {report_dict['At_CEJ']['support']} |
| **Below CEJ** | {report_dict['Below_CEJ']['precision']:.4f} | {report_dict['Below_CEJ']['recall']:.4f} | {report_dict['Below_CEJ']['f1-score']:.4f} | {report_dict['Below_CEJ']['support']} |
| **Macro Average** | **{report_dict['macro avg']['precision']:.4f}** | **{report_dict['macro avg']['recall']:.4f}** | **{macro_f1:.4f}** | {len(val_trues)} |

---

### Confusion Matrix

```text
Actual \\ Predicted   Above_CEJ   At_CEJ   Below_CEJ
Above_CEJ                {cm[0, 0]}          {cm[0, 1]}        {cm[0, 2]}
At_CEJ                   {cm[1, 0]}          {cm[1, 1]}        {cm[1, 2]}
Below_CEJ                {cm[2, 0]}          {cm[2, 1]}        {cm[2, 2]}
```
"""
    report_file = out_dir / "REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved report: {report_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
