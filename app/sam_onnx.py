import copy
import hashlib

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from app.logger import ZLogger
from app.ztypes import (
    PromptType,
    SAM2EncodedInput,
    SamOnnxEncodedInput,
    SamOnnxPrompt,
    SamOnnxResult,
)


class SamOnnxModel:
    """Segmentation model using SegmentAnything"""

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        cv_interpolation: int = cv2.INTER_LINEAR,
    ) -> None:
        self.img_size: int = 1024
        self.input_size = (1024, 1024)
        self.logger = ZLogger("SamOnnxModel")
        self.img = None
        self._cache: dict[str, SamOnnxEncodedInput] = {}
        self.cv_interpolation = cv_interpolation

        # Load models
        providers: list[str] = ort.get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        # providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        if providers:
            self.logger.info(f"Available providers for ONNXRuntime: {providers}")
        else:
            self.logger.warning("No available providers for ONNXRuntime")
        # providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        cuda_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        providers_encoder = [
            ("CUDAExecutionProvider", cuda_provider_options),
            # "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers_decoder = [
            "CPUExecutionProvider",
        ]
        self.logger.info(f"Using encoder: {encoder_path}, decoder: {decoder_path}")
        self.encoder = ort.InferenceSession(
            encoder_path,
            sess_options,
            providers=providers_encoder,
        )
        self.encoder_input_name: str = self.encoder.get_inputs()[0].name
        self.decoder = ort.InferenceSession(decoder_path, providers=providers_decoder)
        self.decoder_output_names: list[str] = [
            output.name for output in self.decoder.get_outputs()
        ]

    def ensure_image_shape(self, img: NDArray):
        """Ensure image shape is (N, C, H, W)"""
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
        if img.ndim == 3:
            img = np.expand_dims(img, 0)
        return img

    def add_encoded_input(self, key: str, inp: SamOnnxEncodedInput):
        if key not in self._cache:
            self._cache[key] = inp

    def key_cached(self, key: str):
        return key in self._cache

    def get_encoded_input(self, key: str):
        return self._cache.get(key, None)

    def get_input_points(self, prompt: list[SamOnnxPrompt]):
        """Get input points"""
        points = []
        labels = []
        for _, mark in enumerate(prompt):
            if mark.type_ == PromptType.POINT:
                points.append(mark.point)
                labels.append(mark.label)
            elif mark.type_ == PromptType.RECTANGLE:
                points.append([mark.point[0], mark.point[1]])  # top left
                points.append([mark.point[2], mark.point[3]])  # type: ignore bottom right
                labels.append(2)
                labels.append(3)
            else:
                raise NotImplementedError
        points, labels = (
            np.array(points, dtype=np.float32),
            np.array(labels, dtype=np.float32).ravel(),
        )
        return points, labels

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def transform_point_labels(
        self,
        points: NDArray,
        labels: NDArray,
        original_size: tuple[int, int],  # (H, W)
        resized_size: tuple[int, int],  # (H, W)
    ):
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        assert points.ndim == 2 and labels.ndim == 1
        if (labels == 2).astype(np.int8).sum() == 0:
            onnx_coord = np.concatenate(
                [points, np.array([[0.0, 0.0]])],
                axis=0,
            )[None, ...]
            onnx_label = np.append(labels, -1)[None, :].astype(np.float32)
        else:
            onnx_coord = points[None, ...]
            onnx_label = labels[None, :].astype(np.float32)
        coords = copy.deepcopy(onnx_coord).astype(np.float32)
        coords[..., 0] = coords[..., 0] * (resized_size[1] / original_size[1])
        coords[..., 1] = coords[..., 1] * (resized_size[0] / original_size[0])
        onnx_coord = coords.astype("float32")
        return onnx_coord, onnx_label

    def postprocess_mask(
        self,
        mask: np.ndarray,  # H, W
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
    ):
        # H, W -> H, W, C
        assert mask.ndim == 2, f"{mask.shape=}"
        # print(f"{mask.shape=}, {original_size=}, {resized_size=}")
        mask[mask < 0] = 0
        mask = mask.astype(np.uint8)
        if mask.shape == original_size:
            return mask
        mask = cv2.resize(
            mask,
            self.input_size,
            interpolation=self.cv_interpolation,
        )
        mask = mask[: resized_size[0], : resized_size[1]]
        mask = cv2.resize(
            mask,
            (original_size[1], original_size[0]),
            interpolation=self.cv_interpolation,
        )
        return mask

    def run_encoder(
        self,
        img: NDArray,
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
    ) -> SamOnnxEncodedInput:
        """Run encoder"""
        encoder_inputs = {self.encoder_input_name: img}
        image_embedding: np.ndarray = self.encoder.run(None, encoder_inputs)[0]  # type: ignore
        res = SamOnnxEncodedInput(
            image_embedding=image_embedding,
            original_size=original_size,
            resized_size=resized_size,
        )
        return res

    def encode(self, cv_image: NDArray) -> SamOnnxEncodedInput:
        """
        Calculate embedding and metadata for a single image.
        """
        md5 = hashlib.md5(cv_image.tobytes()).hexdigest()
        res = self._cache.get(md5, None)
        if res is not None:
            return res

        h, w, c = cv_image.shape
        if h > w:
            nh = self.input_size[0]
            nw = int(self.input_size[0] / h * w)
        else:
            nw = self.input_size[1]
            nh = int(self.input_size[1] / w * h)
        cv_image = cv2.resize(cv_image, (nw, nh), interpolation=self.cv_interpolation)

        if nh < nw:
            cv_image = np.pad(cv_image, ((0, self.input_size[0] - nh), (0, 0), (0, 0)))
        else:
            cv_image = np.pad(cv_image, ((0, 0), (0, self.input_size[1] - nw), (0, 0)))

        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([[58.395, 57.12, 57.375]])
        cv_image = (cv_image - mean) / std

        cv_image = np.transpose(cv_image, (2, 0, 1))[None, ...]
        cv_image = self.ensure_image_shape(cv_image)
        res = self.run_encoder(cv_image, (h, w), (nh, nw))
        self._cache[md5] = res
        return res

    def decode(
        self,
        einput: SamOnnxEncodedInput,
        prompt: list[SamOnnxPrompt],
    ) -> list[SamOnnxResult]:
        # (N, 2), (N,)
        input_points, input_labels = self.get_input_points(prompt)

        onnx_coord, onnx_label = self.transform_point_labels(
            input_points,
            input_labels,
            einput.original_size,
            einput.resized_size,
        )
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": einput.image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(einput.original_size, dtype=np.float32),
        }
        results = self.run_decoder(
            decoder_inputs,
            einput.original_size,
            einput.resized_size,
        )
        return results

    def run_decoder(
        self,
        decoder_inputs: dict[str, NDArray],
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
        postprocess: bool = True,
    ) -> list[SamOnnxResult]:
        outputs: list[np.ndarray] = self.decoder.run(None, decoder_inputs)  # type: ignore
        assert len(outputs) >= 2

        # masks: (B, N, 256, 256)
        # scores: (B, N)
        masks, scores = outputs[0], outputs[1]
        # only use the first result
        masks, scores = masks[0], scores[0]
        assert isinstance(masks, np.ndarray) and isinstance(scores, np.ndarray)
        # masks: (N, 256, 256)
        # scores: (N,)
        assert masks.shape[0] == scores.shape[0], (
            f"masks.shape: {masks.shape}, scores.shape: {scores.shape}"
        )
        assert masks.ndim == 3 and scores.ndim == 1, (
            f"masks.shape: {masks.shape}, scores.shape: {scores.shape}"
        )

        results: list[SamOnnxResult] = []
        for idx in range(len(scores)):
            results.append(
                SamOnnxResult(
                    mask=self.postprocess_mask(
                        masks[idx],
                        original_size,
                        resized_size,
                    )
                    if postprocess
                    else masks[idx],
                    score=scores[idx],
                )
            )
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results

    def predict(
        self,
        img: NDArray,
        prompts: list[SamOnnxPrompt],
        return_all_masks: bool = False,
    ) -> SamOnnxResult | list[SamOnnxResult]:
        img_encoded = self.encode(cv_image=img)
        results = self.decode(img_encoded, prompts)
        return results if return_all_masks else results[0]


class EdgeSam(SamOnnxModel):
    def decode(
        self,
        einput: SamOnnxEncodedInput,
        prompt: list[SamOnnxPrompt],
    ) -> list[SamOnnxResult]:
        """Run decoder"""
        # (N, 2), (N,)
        input_points, input_labels = self.get_input_points(prompt)

        onnx_coord, onnx_label = self.transform_point_labels(
            input_points,
            input_labels,
            einput.original_size,
            einput.resized_size,
        )

        decoder_inputs = {
            "image_embeddings": einput.image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
        }
        results = self.run_decoder(
            decoder_inputs,
            einput.original_size,
            einput.resized_size,
        )

        return results


class SAM2(SamOnnxModel):
    def run_encoder(
        self,
        img: NDArray,
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
    ) -> SAM2EncodedInput:
        """Run encoder"""
        encoder_inputs = {self.encoder_input_name: img}
        (
            high_res_feats_0,
            high_res_feats_1,
            image_embedding,
        ) = self.encoder.run(None, encoder_inputs)  # type: ignore
        res = SAM2EncodedInput(
            image_embedding,  # type: ignore
            original_size=original_size,
            resized_size=resized_size,
            high_res_feats_0=high_res_feats_0,  # type: ignore
            high_res_feats_1=high_res_feats_1,  # type: ignore
        )
        return res

    def run_decoder(
        self,
        decoder_inputs: dict[str, NDArray],
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
        postprocess: bool = True,
    ) -> list[SamOnnxResult]:
        outputs: list[np.ndarray] = self.decoder.run(None, decoder_inputs)  # type: ignore
        # masks: (B, N, 256, 256)
        # scores: (B, N)
        masks, scores = outputs[0], outputs[1]
        # only use the first result
        masks, scores = masks[0], scores[0]
        assert isinstance(masks, np.ndarray) and isinstance(scores, np.ndarray)
        # masks: (N, 256, 256)
        # scores: (N,)
        assert masks.shape[0] == scores.shape[0]
        assert masks.ndim == 3 and scores.ndim == 1

        results: list[SamOnnxResult] = []
        for idx in range(len(scores)):
            results.append(
                SamOnnxResult(
                    mask=self.postprocess_mask(
                        masks[idx],
                        original_size,
                        resized_size,
                    )
                    if postprocess
                    else masks[idx],
                    score=scores[idx],
                )
            )
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results

    def decode(
        self,
        einput: SAM2EncodedInput,
        prompt: list[SamOnnxPrompt],
    ) -> list[SamOnnxResult]:
        """Run decoder"""
        assert einput.high_res_feats_0 is not None
        assert einput.high_res_feats_1 is not None

        # (N, 2), (N,)
        input_points, input_labels = self.get_input_points(prompt)

        onnx_coord, onnx_label = self.transform_point_labels(
            input_points,
            input_labels,
            einput.original_size,
            einput.resized_size,
        )
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embed": einput.image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "high_res_feats_0": einput.high_res_feats_0,
            "high_res_feats_1": einput.high_res_feats_1,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
        }
        results = self.run_decoder(
            decoder_inputs,
            einput.original_size,
            einput.resized_size,
        )

        return results


class SlimSAM(SamOnnxModel):
    ...
