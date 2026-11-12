# Isolated Stage 2 Deep Learning Pilot Runners

This directory (`D:\Workspace\TA3\runners/`) hosts independent, stand-alone execution scripts for research experiments that test new labeling paradigms without modifying submodules (`projects/m3-m2-experiments/` and `projects/m3-m2-rebuild/`).

## Scripts:
- `train_stage2_pilot.py`: Trains a ViT-B/16 tail classifier for Stage 2 (Above CEJ, At CEJ, Below CEJ) on the authentic vision-labeled dataset (`data/custom/gemini-3.8/`).

## Outputs:
- Saved to `runners/outputs/`:
  - `stage2_vit_pilot_best.pth`: Best model checkpoint weights.
  - `stage2_pilot_metrics.json`: Detailed performance metrics and sample predictions.
  - `REPORT.md`: Comprehensive evaluation report with confusion matrix.
