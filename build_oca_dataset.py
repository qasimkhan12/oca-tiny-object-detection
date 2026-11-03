"""
Step 2 of the pipeline: build the O+CA (Original + Cropped Attention) training set.

For every training image:
  warm-up detector -> class-specific Grad-CAM++ heatmaps -> top-k attention crops
  -> crops saved as new training images with projected labels.
The originals and the crops are then listed together in one training file.

Example:
    python build_oca_dataset.py --data VisDrone.yaml --weights runs/detect/warmup/weights/last.pt --k 2
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

from oca.crops import read_yolo_labels, select_crops
from oca.gradcam_pp import YOLOGradCAMPlusPlus, overlay


def list_images(train_spec) -> list[str]:
    """Resolve the 'train' entry of a dataset yaml (dir, list of dirs, or .txt) to image paths."""
    specs = train_spec if isinstance(train_spec, list) else [train_spec]
    files = []
    for s in specs:
        p = Path(s)
        if p.is_dir():
            files += [str(f) for f in sorted(p.rglob("*")) if f.suffix[1:].lower() in IMG_FORMATS]
        elif p.suffix == ".txt":
            parent = p.parent
            for line in p.read_text().splitlines():
                line = line.strip()
                if line:
                    f = Path(line)
                    files.append(str(f if f.is_absolute() else (parent / f).resolve()))
    return files


def save_visual(img, heatmaps, crops, path):
    """Heatmap overlay with the selected crop windows drawn on top (for the README)."""
    hm = np.max(np.stack(list(heatmaps.values())), axis=0) if heatmaps else np.zeros(img.shape[:2])
    vis = overlay(img, hm)
    for c, _ in crops:
        cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), (255, 255, 255), max(2, img.shape[1] // 300))
    cv2.imwrite(str(path), vis)


def main(a):
    data = check_det_dataset(a.data)  # resolves paths (and downloads VisDrone if its yaml says so)
    images = list_images(data["train"])
    if a.limit:
        images = images[: a.limit]
    print(f"{len(images)} training images found")

    out = Path(a.out).resolve()
    img_dir, lbl_dir, vis_dir = out / "images" / "crops", out / "labels" / "crops", out / "visuals"
    for d in (img_dir, lbl_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    cam = YOLOGradCAMPlusPlus(YOLO(a.weights), target_layer=a.layer, imgsz=a.imgsz)
    crop_paths, per_class, n_vis = [], Counter(), 0

    for img_path, lbl_path in tqdm(zip(images, img2label_paths(images)), total=len(images), desc="Grad-CAM++ crops"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo_labels(lbl_path, w, h)
        if len(gt) == 0:
            continue

        heatmaps = cam.class_heatmaps(img, tau_cls=a.tau_cls)
        crops = select_crops(heatmaps, gt, w, h, k=a.k, crop_size=a.crop_size,
                             tau=a.tau, iou_thr=a.iou, min_visible=a.min_visible)

        for j, (c, labels) in enumerate(crops):
            patch = img[c.y1:c.y2, c.x1:c.x2]
            scale = a.crop_out / max(patch.shape[:2])
            patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            name = f"{Path(img_path).stem}_cam{j}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), patch)
            np.savetxt(lbl_dir / f"{name}.txt", labels, fmt="%d %.6f %.6f %.6f %.6f")
            crop_paths.append(str(img_dir / f"{name}.jpg"))
            per_class[data["names"][c.cls]] += 1

        if n_vis < a.vis and crops:
            save_visual(img, heatmaps, crops, vis_dir / f"{Path(img_path).stem}.jpg")
            n_vis += 1

    # --- O+CA training list ----------------------------------------------------
    originals = [str(Path(p).resolve()) for p in images]
    if a.balanced and crop_paths:
        # repeat originals so full images and crops appear ~1:1 (thesis Section 3.3)
        reps = max(1, round(len(crop_paths) / len(originals)))
        originals = originals * reps
    train_list = originals + crop_paths
    random.Random(0).shuffle(train_list)
    (out / "train_oca.txt").write_text("\n".join(train_list) + "\n")

    oca_yaml = {"path": str(out), "train": str(out / "train_oca.txt"), "val": data["val"],
                "nc": data["nc"], "names": data["names"]}
    (out / "data_oca.yaml").write_text(yaml.safe_dump(oca_yaml, sort_keys=False, allow_unicode=True))

    stats = {"original_images": len(images), "attention_crops": len(crop_paths),
             "oca_training_samples": len(train_list), "crops_by_source_class": dict(per_class),
             "settings": vars(a)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "settings"}, indent=2))
    print(f"\nO+CA dataset config written to {out / 'data_oca.yaml'}")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset yaml, e.g. VisDrone.yaml")
    p.add_argument("--weights", required=True, help="warm-up detector weights (15 epochs in the thesis)")
    p.add_argument("--out", default="datasets/oca", help="output folder for crops and the O+CA yaml")
    p.add_argument("--k", type=int, default=2, help="crops per image (thesis: 1, 2 or 3; best k=2)")
    p.add_argument("--tau-cls", type=float, default=0.25, help="class confidence threshold for heatmaps")
    p.add_argument("--tau", type=float, default=0.6, help="heatmap binarisation threshold")
    p.add_argument("--iou", type=float, default=0.7, help="IoU threshold for removing redundant crops")
    p.add_argument("--crop-size", type=int, default=512, help="crop window in ORIGINAL pixels (assumption)")
    p.add_argument("--crop-out", type=int, default=640, help="crops are resized to this size (thesis: 640)")
    p.add_argument("--min-visible", type=float, default=0.5, help="min fraction of a box inside a crop to keep it")
    p.add_argument("--layer", type=int, default=8, help="target layer (8 = final backbone C2f in YOLOv8)")
    p.add_argument("--imgsz", type=int, default=640, help="input size for the Grad-CAM++ forward pass")
    p.add_argument("--balanced", action="store_true", help="repeat originals for a ~1:1 original:crop ratio")
    p.add_argument("--vis", type=int, default=20, help="number of heatmap visualisations to save")
    p.add_argument("--limit", type=int, default=0, help="only process the first N images (for quick tests)")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
