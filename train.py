"""
Train YOLOv8 with the unified configuration from the thesis (Table 3.2).

Three uses, all with the SAME hyper-parameters so results are comparable:
    python train.py --data VisDrone.yaml --name baseline                 # full images only, 100 epochs
    python train.py --data VisDrone.yaml --name warmup --epochs 15       # warm-up model used for Grad-CAM++
    python train.py --data datasets/oca/data_oca.yaml --name oca_k2      # proposed O+CA training

The detector architecture is never modified: the only thing that changes between
runs is the training data.

--resume   continues an interrupted run (useful on Google Colab, where sessions disconnect).
           If the run already finished, it is skipped.
--no-amp   trains in full precision (FP32). Slower, but avoids the numerical overflow (NaN)
           that mixed precision can occasionally cause.

Collapse watchdog: if validation mAP50 drops to zero after the model had started learning,
or is still zero after 5 epochs, training stops and a COLLAPSED.txt file is written in the
run folder. A collapsed run should be deleted or renamed and trained again, not resumed.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

MAP50 = "metrics/mAP50(B)"


def collapse_watchdog(min_epoch: int = 5):
    """Ultralytics callback that stops a run whose validation mAP50 has collapsed to zero."""
    best = {"map50": 0.0}

    def on_fit_epoch_end(trainer):
        m = (trainer.metrics or {}).get(MAP50)
        if m is None:
            return
        m, epoch = float(m), trainer.epoch + 1
        best["map50"] = max(best["map50"], m)
        dead_after_learning = best["map50"] >= 0.01 and m < 1e-4
        never_learned = epoch >= min_epoch and best["map50"] < 1e-4
        if dead_after_learning or never_learned:
            reason = (f"mAP50 fell to {m:.4f} at epoch {epoch} after reaching {best['map50']:.4f}"
                      if dead_after_learning else f"mAP50 still {m:.4f} after {epoch} epochs")
            msg = (f"TRAINING COLLAPSED: {reason}.\n"
                   "The model has stopped learning (usually a numerical overflow, NaN). "
                   "Delete or rename this run folder and train again with --no-amp.")
            print("\n" + "!" * 80 + "\n" + msg + "\n" + "!" * 80 + "\n")
            Path(trainer.save_dir, "COLLAPSED.txt").write_text(msg + "\n")
            trainer.stop = True

    return on_fit_epoch_end


def main(a):
    run_dir = Path(a.project) / a.name
    last = run_dir / "weights" / "last.pt"

    if (run_dir / "COLLAPSED.txt").exists():
        raise SystemExit(f"{run_dir} collapsed earlier (see COLLAPSED.txt). Rename or delete it, then train again.")

    if a.resume and last.exists():
        print(f"Resuming {last}")
        model = YOLO(str(last))
        model.add_callback("on_fit_epoch_end", collapse_watchdog())
        try:
            model.train(resume=True)
        except AssertionError as e:  # Ultralytics raises this when the run is already complete
            print(f"Nothing to resume ({e}). Skipping.")
        return

    # "yolov8l.yaml" = architecture only, random initialisation (trained from scratch, no COCO weights)
    model = YOLO(a.model)
    model.add_callback("on_fit_epoch_end", collapse_watchdog())
    model.train(
        data=a.data,
        project=a.project,
        name=a.name,
        exist_ok=True,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        pretrained=False,
        amp=not a.no_amp,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=5e-4,
        # augmentation pipeline: Mosaic, MixUp, scale jitter, flip, affine, HSV
        mosaic=1.0,
        mixup=a.mixup,
        scale=0.5,
        fliplr=0.5,
        degrees=0.0,
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        seed=a.seed,
        workers=a.workers,
        save_period=a.save_period,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True, help="run name, e.g. baseline / warmup / oca_k2")
    p.add_argument("--model", default="yolov8l.yaml", help="use yolov8n.yaml for a quick test")
    p.add_argument("--project", default="runs/detect", help="folder where runs are saved")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--mixup", type=float, default=0.1, help="MixUp probability (value not given in thesis)")
    p.add_argument("--device", default=None, help="0 for first GPU, cpu for CPU")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-period", type=int, default=-1, help="also save a checkpoint every N epochs")
    p.add_argument("--resume", action="store_true", help="continue this run if a checkpoint exists")
    p.add_argument("--no-amp", action="store_true", help="train in full precision (FP32) to avoid NaN overflow")
    main(p.parse_args())
