"""Image preprocessing for the ONNX models, aligned with the Ultralytics SAM pipeline.

- SAM / EdgeSAM / SlimSAM / SAM2: aspect-ratio preserving letterbox (scale by the min
  ratio, pad right/bottom with 114) to the square encoder input, exactly like
  ``ultralytics.data.augment.LetterBox(imgsz, auto=False, center=False)``.
- SAM3: direct stretch to the target size, like ``LetterBox(..., scale_fill=True)``
  used by the SAM3 predictors.
"""

from __future__ import annotations

import cv2
import numpy as np

SAM_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
SAM_STD = np.array([[58.395, 57.12, 57.375]], np.float32)


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr[..., ::-1].copy()


def preprocess_sam_letterbox(image_bgr: np.ndarray, img_size: int = 1024) -> tuple[np.ndarray, float]:
    """SAM / EdgeSAM / SlimSAM / SAM2 encoder input: letterbox + mean/std normalize.

    Returns ((1,3,img_size,img_size) fp32 tensor, ratio r) where r is the uniform
    scale applied to the original image (prompt coordinates must be multiplied by r).
    """
    h, w = image_bgr.shape[:2]
    r = min(img_size / h, img_size / w)
    new_unpad = (round(w * r), round(h * r))
    im = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
    im = cv2.copyMakeBorder(im, 0, round(dh + 0.1), 0, round(dw + 0.1), cv2.BORDER_CONSTANT, value=(114,) * 3)
    im = _to_rgb(im)
    im = im.astype(np.float32)
    im = (im - SAM_MEAN) / SAM_STD
    return im.transpose(2, 0, 1)[None].astype(np.float32), r


def preprocess_sam3(image_bgr: np.ndarray, target: int = 1008) -> np.ndarray:
    """SAM3 encoder input: stretch to target square, normalized to [-1, 1].

    Returns (1,3,target,target) fp32.
    """
    im = cv2.resize(image_bgr, (target, target), interpolation=cv2.INTER_LINEAR)
    im = _to_rgb(im)
    im = im.astype(np.float32)
    im = (im - 127.5) / 127.5
    return im.transpose(2, 0, 1)[None].astype(np.float32)
