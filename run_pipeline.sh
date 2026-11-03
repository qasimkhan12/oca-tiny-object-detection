#!/usr/bin/env bash
# Full O+CA pipeline on VisDrone (needs a GPU; roughly 1-2 days in total on one RTX-class card).
# Ultralytics downloads and converts VisDrone automatically the first time VisDrone.yaml is used.
set -e

DATA=VisDrone.yaml
K=2

# 1) Baseline: YOLOv8l from scratch on full images
python train.py --data $DATA --name baseline --device 0

# 2) Warm-up model (15 epochs) used only to compute Grad-CAM++ heatmaps
python train.py --data $DATA --name warmup --epochs 15 --device 0

# 3) Grad-CAM++ attention crops -> O+CA dataset
python build_oca_dataset.py --data $DATA --weights runs/detect/warmup/weights/last.pt --k $K --out datasets/oca_k$K

# 4) Proposed method: train from scratch on O+CA (same hyper-parameters as the baseline)
python train.py --data datasets/oca_k$K/data_oca.yaml --name oca_k$K --device 0

# 5) Compare both on the ORIGINAL, uncropped validation set
python evaluate.py --data $DATA \
    --weights runs/detect/baseline/weights/best.pt runs/detect/oca_k$K/weights/best.pt \
    --out results/visdrone_metrics.json
