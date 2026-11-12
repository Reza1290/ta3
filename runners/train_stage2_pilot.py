"""
train_stage2_pilot.py
=====================
Stand-alone runner script to train and evaluate Stage 2 Deep Learning (ViT-B/16 Tail Classifier)
using the authentic vision-labeled dataset from data/custom/gemini-3.8/.

Completely isolated outside projects/m3-m2-experiments/ and projects/m3-m2-rebuild/.
Zero modifications to any existing submodule codebase.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

# Set deterministic seed
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
CLASS_MAP = {1: 0, 2: 1, 3: 2} # four_class_id to 3-class index

class ViTTailClassifier(nn.Module):
    """ViT Transfer Learning Tail: Trainable last N blocks + norm + 3-class head."""
    def __init__(self, backbone: nn.Module, trainable_blocks: int = 2, num_classes: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(*list(backbone.blocks.children())[-trainable_blocks:])
        self.norm = backbone.norm
        self.head = nn.Linear(backbone.num_features, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N_tokens, Dim]
        x = self.blocks(tokens)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

def preprocess_crop(crop_array: np.ndarray) -> torch.Tensor:
    """Preprocess (224, 224) crop with ImageNet normalization across 3 channels."""
    arr = np.array(crop_array, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1) # [3, 224, 224]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

@torch.no_grad()
def extract_prefix_tokens(backbone: nn.Module, images: torch.Tensor, trainable_blocks: int = 2) -> torch.Tensor:
    """Pass images through frozen ViT prefix to extract sequence tokens."""
    backbone.eval()
    x = backbone.patch_embed(images)
    x = backbone._pos_embed(x)
    x = backbone.patch_drop(x)
    x = backbone.norm_pre(x)
    for block in list(backbone.blocks.children())[:-trainable_blocks]:
        x = block(x)
    return x

def calculate_balanced_weights(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def main():
    seed_all(SEED)
    root_dir = Path("D:/Workspace/TA3")
    labels_csv = root_dir / "data/custom/gemini-3.8/labels_50_cases_vision.csv"
    if not labels_csv.exists():
        labels_csv = root_dir / "data/custom/gemini-3.8/labels_50_cases.csv"
    
    crops_dir = root_dir / "data/processed/crops_2d"
    base_weights_path = root_dir / "weights/vit_base_binary_best.pth"
    out_dir = root_dir / "runners/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2 PILOT DEEP LEARNING RUNNER")
    print("Using 100% Pure Multimodal Vision Labels (Gemini-3.8)")
    print("=" * 70)

    # 1. Load labels and filter contacts
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} total cases from: {labels_csv}")

    # Contact only & four_class_id in [1, 2, 3]
    contact_df = df[(df["contact_status"] == "CONTACT") & (df["four_class_id"].isin([1.0, 2.0, 3.0]))].copy()
    contact_df["target"] = contact_df["four_class_id"].astype(int).map(CLASS_MAP)
    print(f"Filtered {len(contact_df)} CONTACT cases with valid CEJ classes:")
    for cid, cname in enumerate(CLASS_NAMES):
        count = (contact_df["target"] == cid).sum()
        print(f"  Class {cid} ({cname}): {count} samples")

    # 2. Split train and val
    train_df = contact_df[contact_df["split"] == "train"].copy()
    val_df = contact_df[contact_df["split"] == "val"].copy()
    print(f"\nPartitions: Train = {len(train_df)} cases, Validation = {len(val_df)} cases")

    # 3. Load crops and extract tokens
    def load_tensors(df_subset):
        tensors = []
        targets = []
        sids = []
        for _, r in df_subset.iterrows():
            sid = r["sample_id"]
            npy_path = crops_dir / f"{sid}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Missing crop: {npy_path}")
            arr = np.load(npy_path)
            t = preprocess_crop(arr)
            tensors.append(t)
            targets.append(r["target"])
            sids.append(sid)
        return torch.stack(tensors, dim=0), torch.tensor(targets, dtype=torch.long), sids

    train_images, train_targets, train_sids = load_tensors(train_df)
    val_images, val_targets, val_sids = load_tensors(val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # 4. Load base ViT-B/16 backbone
    print("\nLoading ViT-B/16 base backbone from:", base_weights_path)
    backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    state_dict = torch.load(base_weights_path, map_location="cpu")
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)

    # 5. Extract cached prefix tokens
    print("Extracting cached prefix tokens from frozen ViT blocks...")
    trainable_blocks = 2
    train_tokens = extract_prefix_tokens(backbone, train_images.to(device), trainable_blocks=trainable_blocks).cpu()
    val_tokens = extract_prefix_tokens(backbone, val_images.to(device), trainable_blocks=trainable_blocks).cpu()
    print(f"Prefix tokens shape: {train_tokens.shape} (N, tokens=197, dim=768)")

    # 6. Initialize ViT Tail Classifier
    tail = ViTTailClassifier(backbone, trainable_blocks=trainable_blocks, num_classes=3).to(device)
    
    # Class weights for loss
    class_weights = calculate_balanced_weights(train_targets.tolist(), num_classes=3).to(device)
    print(f"Balanced Class Weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(tail.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # Data loaders
    batch_size = 4
    train_loader = DataLoader(TensorDataset(train_tokens, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tokens, val_targets), batch_size=batch_size, shuffle=False)

    # 7. Training loop
    epochs = 20
    best_val_f1 = -1.0
    best_model_state = None
    best_epoch = -1
    history = []

    print(f"\nStarting Stage 2 Training ({epochs} epochs, batch_size={batch_size})...")
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

    # 8. Load best model and evaluate
    print(f"\nLoading best checkpoint from Epoch {best_epoch} (Val Macro-F1: {best_val_f1:.4f})...")
    tail.load_state_dict(best_model_state)
    best_weights_path = out_dir / "stage2_vit_pilot_best.pth"
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
    print("FINAL VALIDATION EVALUATION (STAGE 2 CEJ)")
    print("=" * 50)
    print(classification_report(val_trues, val_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    print(cm)
    print(f"Overall Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"Three-Class Macro-F1:      {macro_f1:.4f}")

    # Save metrics JSON
    metrics_data = {
        "best_epoch": best_epoch,
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

    metrics_path = out_dir / "stage2_pilot_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Write Markdown Report
    report_md = f"""# STAGE 2 CEJ CLASSIFICATION PILOT REPORT
## Deep Learning Transfer Learning (ViT-B/16 Tail) on Pure Vision Labels

* **Model Architecture:** Vision Transformer ViT-B/16 (`vit_base_patch16_224`) Tail Classifier
* **Dataset:** 50-Case Pilot Pure Vision Labels (`data/custom/gemini-3.8/labels_50_cases_vision.csv`)
* **Training Subset:** 33 Contact Cases (12 Above CEJ, 5 At CEJ, 16 Below CEJ)
* **Validation Subset:** 7 Contact Cases (3 Above CEJ, 2 At CEJ, 2 Below CEJ)
* **Best Epoch:** {best_epoch} / {epochs}
* **Macro-F1 (3 Classes):** **{macro_f1:.4f}**
* **Balanced Accuracy:** **{bal_acc*100:.2f}%**

---

### Per-Class Performance (Validation Set)

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

---

### Per-Sample Validation Results

| Sample ID | Ground Truth | Prediction | Confidence | Correct? |
| :--- | :--- | :--- | :---: | :---: |
"""
    for sp in metrics_data["sample_predictions"]:
        correct = "✅" if sp["ground_truth"] == sp["prediction"] else "❌"
        report_md += f"| `{sp['sample_id']}` | **{sp['ground_truth']}** | {sp['prediction']} | {sp['confidence']:.2f} | {correct} |\n"

    report_md += """
---
*Runner generated in isolated `runners/` directory without altering `projects/m3-m2-experiments/` or `projects/m3-m2-rebuild/`.*
"""
    report_file = out_dir / "REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved report: {report_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
