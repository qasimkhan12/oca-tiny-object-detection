# Attention-Guided O+CA Training for Tiny Object Detection (Grad-CAM++ × YOLOv8)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qasimkhan12/oca-tiny-object-detection/blob/main/OCA_Tiny_Object_Detection.ipynb)

An **unofficial reimplementation** of an attention-guided, data-centric training framework for detecting tiny objects (pedestrians, bicycles, motorbikes…) in drone imagery.

> Based on: Swetha Manikandan, *An Attention-Guided Heatmap Clustering Framework for Tiny Object Detection Using Grad-CAM++*, M.S. thesis, Dept. of Computer Science and Engineering, Yuan Ze University, Taiwan, December 2025. Advisor: Prof. Qazi Mazhar Ul Haq.
>
> All credit for the method goes to the original author. This repository is my independent implementation of the method from its written description.

<!-- After running build_oca_dataset.py, put one image from datasets/oca/visuals/ here: -->
<!-- ![Grad-CAM++ heatmap and selected crops](images/example_heatmap.jpg) -->

## The idea in one paragraph

Tiny objects in aerial images cover only a few pixels and get lost when a 2000-pixel image is shrunk to 640 pixels for training. Instead of changing the detector, this method changes **what the detector sees during training**. A warm-up YOLOv8 model produces **class-specific Grad-CAM++ heatmaps** showing where it "looks" for each class. The most activated regions are cut out as **zoomed-in crops** and added to the training set next to the original images. The result is a hybrid dataset called **O+CA (Original + Cropped Attention)**. The final detector is a standard, unmodified YOLOv8l, so **inference speed is unchanged**.

## Pipeline

```
Full images ──► (1) warm-up YOLOv8l, 15 epochs
                       │
                       ▼
               (2) Grad-CAM++ heatmap per predicted class (conf > 0.25)
                       │  binarise at 0.6 → connected components → peaks
                       ▼
               (3) top-k crops per image (k = 2)
                       │  keep crops containing ground truth, drop overlaps (IoU > 0.7)
                       ▼
               (4) O+CA dataset = original images + crops (resized to 640)
                       │
                       ▼
               (5) train YOLOv8l from scratch on O+CA → standard single-pass inference
```

## Repository structure

| File | What it does |
|---|---|
| `OCA_Tiny_Object_Detection.ipynb` | **Start here.** Runs the whole experiment on a free Google Colab GPU, with plots |
| `oca/gradcam_pp.py` | Class-specific Grad-CAM++ for an unmodified Ultralytics YOLOv8 model |
| `oca/crops.py` | Heatmap thresholding, peak selection, ground-truth filtering, IoU deduplication, label projection |
| `build_oca_dataset.py` | Runs steps 2–4 over the whole training set and writes `data_oca.yaml` |
| `train.py` | Training with the thesis's unified configuration (baseline, warm-up and O+CA runs) |
| `evaluate.py` | COCO-style AP, AP_S/M/L and tiny-object AP (<16, 16–32, 32–64 px) on **uncropped** validation images |
| `run_pipeline.sh` | The full experiment, start to finish |

## Quick start

### Option 1: Google Colab (no installation)

Click the **Open in Colab** button at the top, select **Runtime → Change runtime type → T4 GPU**, and run the cells from top to bottom. Start with `RUN_MODE = "smoke_test"` (~5 minutes) to check that everything works, then switch to `"quick"`. Checkpoints are saved to Google Drive, so training resumes after a disconnect.

### Option 2: Your own machine with a GPU

```bash
pip install -r requirements.txt
bash run_pipeline.sh          # needs a GPU; VisDrone is downloaded automatically
```

Or step by step:

```bash
python train.py --data VisDrone.yaml --name baseline --device 0
python train.py --data VisDrone.yaml --name warmup --epochs 15 --device 0
python build_oca_dataset.py --data VisDrone.yaml --weights runs/detect/warmup/weights/last.pt --k 2 --out datasets/oca_k2
python train.py --data datasets/oca_k2/data_oca.yaml --name oca_k2 --device 0
python evaluate.py --data VisDrone.yaml --weights runs/detect/baseline/weights/best.pt runs/detect/oca_k2/weights/best.pt
```

**Quick test without a GPU** (tiny model, a few images, a few minutes):

```bash
python train.py --data coco8.yaml --name warmup --model yolov8n.yaml --epochs 3 --device cpu
python build_oca_dataset.py --data coco8.yaml --weights runs/detect/warmup/weights/last.pt --tau-cls 0.01 --out datasets/oca_test
```

## Training configuration (from the thesis, Table 3.2)

| Setting | Value |
|---|---|
| Detector | YOLOv8l, trained from scratch (no COCO weights) |
| Epochs | 100 (warm-up model: 15) |
| Optimizer | AdamW, lr 1e-3, cosine decay, 3 warm-up epochs, weight decay 5e-4 |
| Batch size | 16 |
| Augmentation | Mosaic, MixUp, scale jitter, flip, affine, HSV |
| Grad-CAM++ | class threshold 0.25, heatmap threshold 0.6, final backbone C2f layer |
| Crops | k = 2 per image, IoU dedup 0.7, resized to 640×640 |

## Implementation choices and assumptions

The thesis leaves some details open or describes them in more than one way. These are the choices made here. All of them can be changed with command-line flags.

1. **Crop window size** is not stated. Default: 512×512 pixels of the original image (`--crop-size`), resized to 640. That roughly magnifies tiny objects 3–4× compared with downscaling the full image.
2. **Input resolution:** the thesis mentions both 640 and 1024 for full images. A single `--imgsz 640` is used for all training samples.
3. **Mixing ratio:** the thesis describes both a 1:1 original:crop ratio per batch and uniform sampling over the combined set (Table 3.1). Default: uniform; `--balanced` approximates 1:1 by repeating the originals.
4. **Class score for Grad-CAM++:** the sum of class-*c* confidences over all predictions above the 0.25 threshold.
5. **Grad-CAM++ weights** use the standard formulation of Chattopadhay et al. (2018), including its factor of 2 and the ReLU on gradients.
6. **Label projection:** a box is kept in a crop if at least 50% of its area is inside (`--min-visible`).
7. **Redundancy removal** uses IoU only. The SSIM-based filter mentioned once in the thesis is not implemented.
8. **MixUp probability** is not stated. Default: 0.1.

## Results

Evaluated on the original, **uncropped** VisDrone2019-DET validation set (548 images):

| Model | AP | AP50 | AP75 | AP_S | AP<16 | AP16-32 | AP32-64 |
|---|---|---|---|---|---|---|---|
| YOLOv8l baseline | _run evaluate.py_ | | | | | | |
| YOLOv8l + O+CA (k=2) | | | | | | | |

### A note on evaluation

Evaluation here always uses full, uncropped validation images. Attention crops are selected around labelled objects, so scoring a model on crops (or letting validation images leak into crop generation) would inflate the results. Published VisDrone-DET results typically fall well below the AP values reported in the thesis, so independent runs may show smaller improvements. The numbers in the table above are the ones measured with this code.

`evaluate.py` uses the official COCO evaluator (pycocotools), so its AP is usually a few points lower than the mAP printed by Ultralytics during training, because the two use different interpolation and NMS settings. Compare models using the same evaluator.

## Limitations and next steps

- [ ] Run the k = 1, 2, 3 ablation and the EigenCAM comparison
- [ ] Cross-dataset evaluation on UAVDT
- [ ] Adaptive k based on object density (suggested as future work in the thesis)
- [ ] Try a higher-resolution target layer (e.g. the stride-8 neck layer) for sharper heatmaps on tiny objects

## License and acknowledgements

- Method: Swetha Manikandan, Yuan Ze University (thesis cited above).
- Detector: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed under AGPL-3.0. Because this project uses Ultralytics, it is also released under **AGPL-3.0**.
- Grad-CAM++: A. Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks," WACV 2018.
- Dataset: VisDrone2019-DET (P. Zhu et al.).
