"""
Class-specific Grad-CAM++ for an (unmodified) Ultralytics YOLOv8 detector.

Thesis reference: Section 3.3, Eq. (3.7)-(3.8).
- Activations A_k are taken from the final C2f block of the YOLOv8 backbone
  (layer index 8 in yolov8*.yaml), as described in the thesis.
- The class score y_c is the sum of class-c confidences over all predictions
  whose confidence exceeds tau_cls (default 0.25).
- Grad-CAM++ weights follow the standard formulation of Chattopadhay et al. (2018),
  using the usual closed-form approximation of the 2nd/3rd-order derivatives.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class YOLOGradCAMPlusPlus:
    def __init__(self, yolo, target_layer: int = 8, imgsz: int = 640, device: str | None = None):
        """
        Args:
            yolo: an ultralytics.YOLO object (e.g. YOLO("runs/warmup/weights/last.pt")).
            target_layer: index of the layer whose activations are used (8 = last backbone C2f in YOLOv8).
            imgsz: square size the image is resized to before the forward pass.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = yolo.model.to(self.device).float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)  # we only need gradients w.r.t. activations
        self.names = yolo.names
        self.imgsz = imgsz

        self._acts = None
        self._grads = None
        layer = self.model.model[target_layer]
        layer.register_forward_hook(self._save_activation)

    # -- hooks -------------------------------------------------------------
    def _save_activation(self, module, inp, out):
        self._acts = out
        out.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._grads = grad

    # -- helpers -----------------------------------------------------------
    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR->RGB, HWC->CHW
        x = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(self.device)
        return x.requires_grad_(True)  # makes the graph differentiable without touching weights

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class scores of shape (num_predictions, num_classes)."""
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):  # defensive: some versions return dicts
            out = out.get("one2many", out.get("pred", next(iter(out.values()))))
            if isinstance(out, (list, tuple)):
                out = out[0]
        # YOLOv8 eval output: (1, 4 + nc, N) -> boxes (xywh) followed by sigmoid class scores
        return out[0, 4:, :].T

    @staticmethod
    def _gradcam_pp(acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        """Grad-CAM++ (Eq. 3.7-3.8). acts/grads: (1, K, h, w)."""
        g2, g3 = grads.pow(2), grads.pow(3)
        sum_a = acts.sum(dim=(2, 3), keepdim=True)
        alpha = g2 / (2 * g2 + sum_a * g3 + 1e-7)
        alpha = torch.where(grads != 0, alpha, torch.zeros_like(alpha))
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))[0]
        return cam.detach().cpu().numpy()

    # -- public API --------------------------------------------------------
    def class_heatmaps(self, img_bgr: np.ndarray, tau_cls: float = 0.25) -> dict[int, np.ndarray]:
        """
        Compute one Grad-CAM++ heatmap per class that is confidently predicted in the image.

        Returns:
            {class_id: heatmap} with each heatmap min-max normalised to [0, 1] and resized
            to the ORIGINAL image size (H, W).
        """
        H, W = img_bgr.shape[:2]
        heatmaps: dict[int, np.ndarray] = {}

        x = self._preprocess(img_bgr)
        scores = self._forward(x)                     # (N, nc)
        conf = scores.detach()

        for c in range(scores.shape[1]):
            mask = conf[:, c] > tau_cls
            if not mask.any():
                continue                              # Algorithm 1, line 3: only classes with y_c > tau_cls
            y_c = scores[mask, c].sum()
            self.model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            y_c.backward(retain_graph=True)

            cam = self._gradcam_pp(self._acts, self._grads)
            cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
            lo, hi = cam.min(), cam.max()
            if hi - lo < 1e-8:
                continue
            heatmaps[c] = (cam - lo) / (hi - lo)

        return heatmaps


def overlay(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Colour overlay of a [0,1] heatmap on an image, for figures in the README."""
    color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(color, alpha, img_bgr, 1 - alpha, 0)
