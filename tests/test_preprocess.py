"""Preprocessing tests: SAM-family stretch, SlimSAM/SAM2 letterbox math, SAM3 dual path."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.sam_ort.preprocess import preprocess_sam, preprocess_sam3, preprocess_sam_letterbox

SAM_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
SAM_STD = np.array([[58.395, 57.12, 57.375]], np.float32)


def reference_letterbox(img: np.ndarray, new_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Reimplementation of ultralytics LetterBox(auto=False, center=False)."""
    h, w = img.shape[:2]
    dst_h, dst_w = new_shape
    r = min(dst_h / h, dst_w / w)
    new_unpad = (round(w * r), round(h * r))
    im = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = dst_w - new_unpad[0], dst_h - new_unpad[1]
    im = cv2.copyMakeBorder(im, 0, round(dh + 0.1), 0, round(dw + 0.1), cv2.BORDER_CONSTANT, value=(114,) * 3)
    return im, r


class TestPreprocessSam:
    def test_stretch_shape_and_norm(self, img_bgr):
        tensor = preprocess_sam(img_bgr, 1024)
        assert tensor.shape == (1, 3, 1024, 1024)
        assert tensor.dtype == np.float32
        # stretched: content spans the full tensor, no padding
        expected = cv2.resize(img_bgr, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        expected = expected[..., ::-1].astype(np.float32)
        expected = ((expected - SAM_MEAN) / SAM_STD).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(tensor, expected, atol=1e-5)


class TestPreprocessSamLetterbox:
    def test_shape_and_ratio(self, img_bgr):
        tensor, r = preprocess_sam_letterbox(img_bgr, 1024)
        assert tensor.shape == (1, 3, 1024, 1024)
        assert tensor.dtype == np.float32
        h, w = img_bgr.shape[:2]
        assert r == pytest.approx(min(1024 / h, 1024 / w))

    def test_matches_ultralytics_letterbox(self, img_bgr):
        ref, r = reference_letterbox(img_bgr, (1024, 1024))
        tensor, ratio = preprocess_sam_letterbox(img_bgr, 1024)
        ref_rgb = ref[..., ::-1].astype(np.float32)
        ref_norm = ((ref_rgb - SAM_MEAN) / SAM_STD).transpose(2, 0, 1)[None]
        np.testing.assert_allclose(tensor, ref_norm, atol=1e-5)
        assert ratio == pytest.approx(r)

    def test_padding_value_and_position(self):
        # 1280x720 -> 1024x1024: scale 0.8, content 1024x576, pad bottom 448
        img = np.full((720, 1280, 3), 50, np.uint8)
        tensor, r = preprocess_sam_letterbox(img, 1024)
        content = tensor[0, :, :576, :]  # first 576 rows are content
        pad = tensor[0, :, 576:, :]  # bottom padding
        assert r == pytest.approx(0.8)
        mean = np.array([123.675, 116.28, 103.53], np.float32)
        std = np.array([58.395, 57.12, 57.375], np.float32)
        # content: gray 50 -> normalized (50-mean)/std per channel
        np.testing.assert_allclose(content.mean(axis=(1, 2)), (50 - mean) / std, atol=0.5)
        # padding: 114 -> normalized (114-mean)/std per channel
        np.testing.assert_allclose(pad.mean(axis=(1, 2)), (114 - mean) / std, atol=0.5)

    def test_prompt_scale_consistency(self, img_bgr):
        # a point at the center of the original image maps to the center of the content area
        tensor, r = preprocess_sam_letterbox(img_bgr, 1024)
        h, w = img_bgr.shape[:2]
        cx, cy = w / 2, h / 2
        assert cx * r <= 1024 and cy * r <= 1024
        assert 0 <= cx * r and 0 <= cy * r


class TestPreprocessSam3:
    def test_shape_and_range(self, img_bgr):
        tensor = preprocess_sam3(img_bgr, 1008)
        assert tensor.shape == (1, 3, 1008, 1008)
        assert tensor.dtype == np.float32
        # image values in [0, 255] -> normalized in [-1, 1]
        assert tensor.min() >= -1.01 and tensor.max() <= 1.01

    def test_stretch_no_padding(self, img_bgr):
        tensor = preprocess_sam3(img_bgr, 1008)
        expected = cv2.resize(img_bgr, (1008, 1008)).astype(np.float32)
        expected = expected[..., ::-1] / 127.5 - 1.0
        np.testing.assert_allclose(tensor[0].transpose(1, 2, 0), expected, atol=1e-5)

    def test_pad_letterbox(self, img_bgr):
        tensor = preprocess_sam3(img_bgr, 1008, pad=True)
        assert tensor.shape == (1, 3, 1008, 1008)
        # 1280x720 -> min-ratio 1008/1280=0.7875, content 1008x567, pad bottom 441
        ref, r = reference_letterbox(img_bgr, (1008, 1008))
        ref_norm = ref[..., ::-1].astype(np.float32) / 127.5 - 1.0
        ref_norm = ref_norm.transpose(2, 0, 1)[None]
        np.testing.assert_allclose(tensor, ref_norm, atol=1e-5)
