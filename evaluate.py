"""
Evaluate one or more trained detectors on the ORIGINAL (uncropped) validation set
with COCO-style metrics, including the fine-grained tiny-object ranges from the thesis
(Table 4.5: <16x16, 16-32, 32-64 pixels).

Example:
    python evaluate.py --data VisDrone.yaml \
        --weights runs/detect/baseline/weights/best.pt runs/detect/oca_k2/weights/best.pt

Important: always evaluate on full, uncropped validation images. Evaluating on attention
crops would leak ground-truth information (crops are chosen around labelled objects) and
inflate the scores.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, img2label_paths

from build_oca_dataset import list_images
from oca.crops import read_yolo_labels

AREAS = {  # label: (min_area, max_area) in pixels^2 of the original image
    "AP_S": (0, 32 ** 2), "AP_M": (32 ** 2, 96 ** 2), "AP_L": (96 ** 2, 1e10),
    "AP<16": (0, 16 ** 2), "AP16-32": (16 ** 2, 32 ** 2), "AP32-64": (32 ** 2, 64 ** 2),
}


def build_coco_gt(images, names):
    gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in names.items()]}
    ann_id = 1  # pycocotools treats id 0 as "unmatched", so ids must start at 1
    for img_id, (img_path, lbl_path) in enumerate(zip(images, img2label_paths(images))):
        h, w = cv2.imread(img_path).shape[:2]
        gt["images"].append({"id": img_id, "file_name": img_path, "width": w, "height": h})
        for cls, x1, y1, x2, y2 in read_yolo_labels(lbl_path, w, h):
            bw, bh = x2 - x1, y2 - y1
            gt["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": int(cls),
                                      "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    return coco


def predict(model, images, imgsz, max_det):
    dets = []
    for img_id, path in enumerate(tqdm(images, desc="predict", leave=False)):
        r = model.predict(path, imgsz=imgsz, conf=0.001, iou=0.6, max_det=max_det, verbose=False)[0]
        for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            dets.append({"image_id": img_id, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    return dets


def coco_metrics(coco_gt, dets, max_det):
    if not dets:  # model detected nothing: report zeros-as-nan explicitly instead of an empty entry
        print("WARNING: this model made no detections at all. Its training probably failed.")
        return {k: float("nan") for k in ["AP", "AP50", "AP75", *AREAS]} | {"per_class_AP": {}, "no_detections": True}
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 100, max_det]
    ev.params.areaRng = [[0, 1e10]] + [list(v) for v in AREAS.values()]
    ev.params.areaRngLbl = ["all"] + list(AREAS)
    ev.evaluate()
    ev.accumulate()

    prec = ev.eval["precision"]  # [IoU thresholds, recall, classes, areas, maxDets]

    def ap(p):
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    m = -1  # largest maxDets
    out = {"AP": ap(prec[:, :, :, 0, m]), "AP50": ap(prec[0, :, :, 0, m]), "AP75": ap(prec[5, :, :, 0, m])}
    for i, label in enumerate(AREAS, start=1):
        out[label] = ap(prec[:, :, :, i, m])
    names = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
    out["per_class_AP"] = {names[cid]: ap(prec[:, :, k, 0, m]) for k, cid in enumerate(ev.params.catIds)}
    return out


def main(a):
    data = check_det_dataset(a.data)
    images = list_images(data["val"])
    if a.limit:
        images = images[: a.limit]
    coco_gt = build_coco_gt(images, data["names"])

    results = {}
    for w in a.weights:
        name = Path(w).parent.parent.name if Path(w).parent.name == "weights" else Path(w).stem
        print(f"\nEvaluating {name} on {len(images)} uncropped validation images ...")
        results[name] = coco_metrics(coco_gt, predict(YOLO(w), images, a.imgsz, a.max_det), a.max_det)

    cols = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AP<16", "AP16-32", "AP32-64"]
    lines = ["| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r.get(c, float('nan')):.3f}" for c in cols) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    Path(a.out).with_suffix(".md").write_text(table + "\n")
    print(f"\nSaved {a.out} and {Path(a.out).with_suffix('.md')}")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-det", type=int, default=500, help="VisDrone scenes are dense; 500 is customary")
    p.add_argument("--limit", type=int, default=0, help="only evaluate the first N images (quick test)")
    p.add_argument("--out", default="results/metrics.json")
    return p


if __name__ == "__main__":
    main(get_parser().parse_args())
