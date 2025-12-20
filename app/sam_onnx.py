import copy
import hashlib
from collections import OrderedDict

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from app.logger import ZLogger
from app.ztypes import (
    PromptType,
    SAM2EncodedInput,
    SAM3EncodedInput,
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
        img_size: int = 1024,
        cache_size: int = 241,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> None:
        self.img_size: int = img_size
        self.input_size = (img_size, img_size)
        self.logger = ZLogger("SamOnnxModel")
        self.img = None
        self._cache: OrderedDict[str, SamOnnxEncodedInput] = OrderedDict()
        self.cv_interpolation = cv_interpolation
        self.cache_size: int = cache_size
        self.mean = np.array(mean or [123.675, 116.28, 103.53])
        self.std = np.array(std or [[58.395, 57.12, 57.375]])

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
        self.decoder_output_names: list[str] = [output.name for output in self.decoder.get_outputs()]

    def ensure_image_shape(self, img: NDArray):
        """Ensure image shape is (N, C, H, W)"""
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
        if img.ndim == 3:
            img = np.expand_dims(img, 0)
        return img

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        new_img = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        new_img = (new_img - self.mean) / self.std
        new_img = np.transpose(new_img, (2, 0, 1))[None, ...]
        new_img = self.ensure_image_shape(new_img)
        return new_img

    def add_encoded_input(self, key: str, inp: SamOnnxEncodedInput):
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
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
        # self.logger.debug(f"{mask.shape=}, {original_size=}, {resized_size=}")
        # self.logger.debug(f"{mask.min()=}, {mask.max()=}")
        mask[mask < 0] = 0
        mask[mask > 0] += 255 - mask.max()
        mask = mask.astype(np.uint8)
        cv2.imwrite("mask.png", mask)
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
        res = self.get_encoded_input(md5)
        if res is not None:
            return res

        # N, C, H, W
        h, w = cv_image.shape[:2]
        new_img = self.preprocess_image(cv_image)
        nh, nw = new_img.shape[2:]
        res = self.run_encoder(new_img, (h, w), (nh, nw))

        self.add_encoded_input(md5, res)

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
        assert masks.shape[0] == scores.shape[0], f"masks.shape: {masks.shape}, scores.shape: {scores.shape}"
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
    ) -> list[SamOnnxResult]:
        img_encoded = self.encode(cv_image=img)
        results = self.decode(img_encoded, prompts)
        return results


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


class SlimSAM(SamOnnxModel): ...


class SAM3(SamOnnxModel):
    """SAM3 ONNX Model Implementation"""

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        cv_interpolation: int = cv2.INTER_LINEAR,
        img_size: int = 1008,
        cache_size: int = 241,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            cv_interpolation=cv_interpolation,
            img_size=img_size,
            cache_size=cache_size,
            mean=mean,
            std=std,
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess: resize to target size and normalize"""
        # Resize image to target size
        resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Normalize: [0,255] -> [-1,1]
        normalized = resized.astype(np.float32) / 127.5 - 1.0

        # Convert to NCHW format
        tensor = normalized.transpose(2, 0, 1)[np.newaxis]
        return tensor

    def xyxy_to_cxcywh_norm(
        self,
        boxes: list[tuple[float, float, float, float]],
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Convert xyxy to cxcywh"""
        # N, 4 => N, (x0, y0, x1, y1)
        x0, y0, x1, y1 = np.array(boxes, dtype=np.float32).T
        result = np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis=1)
        result = result / np.array([img_w, img_h, img_w, img_h])
        return result

    def nms_filter(
        self,
        boxes: list[tuple[float, float, float, float]],
        iou_threshold: float = 0.5,
    ) -> np.ndarray:
        """Apply NMS to filter boxes"""
        # boxes: list[(x1, y1, x2, y2)]
        if len(boxes) == 0:
            return np.array([])
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        # Sort contours by area in descending order
        indices = np.argsort(areas)[::-1]
        keep_indices = np.zeros(len(boxes), dtype=bool)

        while len(indices) > 0:
            # Get the index of the largest remaining contour
            current = indices[0]
            keep_indices[current] = True

            if len(indices) == 1:
                break

            # Get the rest of the indices
            rest_indices = indices[1:]
            # Calculate IoU between the current contour and all remaining contours
            current_box = boxes[current]
            ious = []

            for idx in rest_indices:
                box = boxes[idx]

                # Calculate intersection
                x1 = max(current_box[0], box[0])
                y1 = max(current_box[1], box[1])
                x2 = min(current_box[2], box[2])
                y2 = min(current_box[3], box[3])

                # Calculate intersection area
                if x2 <= x1 or y2 <= y1:
                    intersection = 0
                else:
                    intersection = (x2 - x1) * (y2 - y1)

                # Calculate union area
                union = areas[current] + areas[idx] - intersection
                # Calculate IoU
                iou = intersection / union if union > 0 else 0
                ious.append(iou)

            # Keep only contours with IoU <= threshold
            indices = [rest_indices[i] for i, iou in enumerate(ious) if iou <= iou_threshold]

        return keep_indices

    def decode(
        self,
        einput: SAM3EncodedInput,
        prompt: list[SamOnnxPrompt],
    ) -> list[SamOnnxResult]:
        """Decode features to generate masks"""
        assert einput.image_embedding is not None  # fpn_feat_0
        assert einput.fpn_feat_1 is not None
        assert einput.fpn_feat_2 is not None
        assert einput.fpn_pos_2 is not None

        boxes_array = (
            self.xyxy_to_cxcywh_norm(
                [p.point for p in prompt if len(p.point) == 4],
                einput.original_size[1],
                einput.original_size[0],
            )
            .astype(np.float32)
            .reshape(1, -1, 4)
        )
        input_boxes_labels = np.array(
            [p.label for p in prompt if len(p.point) == 4],
            dtype=np.int64,
        ).reshape(1, -1)

        self.logger.info(f"{boxes_array=}, {input_boxes_labels=}")

        decoder_inputs = {
            "fpn_feat_0": einput.image_embedding,  # [batch, 256, 288, 288]    FLOAT
            "fpn_feat_1": einput.fpn_feat_1,  # [batch, 256, 144, 144]    FLOAT
            "fpn_feat_2": einput.fpn_feat_2,  # [batch, 256, 72, 72]      FLOAT
            "fpn_pos_2": einput.fpn_pos_2,  # [batch, 256, 72, 72]      FLOAT
            "input_boxes": boxes_array,  # [batch, num_boxes, 4]     FLOAT
            "input_boxes_labels": input_boxes_labels,  # [batch, num_boxes]        INT64
        }

        results = self.run_decoder(
            decoder_inputs,
            einput.original_size,
            einput.resized_size,
        )

        return results

    def run_encoder(
        self,
        img: NDArray,
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
    ) -> SAM3EncodedInput:
        outputs = self.encoder.run(None, {"images": img})
        return SAM3EncodedInput(
            image_embedding=outputs[0],  # fpn_feat_0 [B, 256, 288, 288] # type: ignore
            original_size=original_size,
            resized_size=resized_size,
            fpn_feat_1=outputs[1],  # fpn_feat_1 [B, 256, 144, 144] # type: ignore
            fpn_feat_2=outputs[2],  # fpn_feat_2 [B, 256, 72, 72] # type: ignore
            fpn_pos_2=outputs[3],  # fpn_pos_2 [B, 256, 72, 72] # type: ignore
        )

    def run_decoder(
        self,
        decoder_inputs: dict[str, np.ndarray],
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
        postprocess: bool = True,
    ) -> list[SamOnnxResult]:
        # pred_masks, pred_boxes, pred_logits, presence_logits
        # Outputs:
        #     pred_masks            [batch, 200, 288, 288]    FLOAT
        #     pred_boxes            [batch, 200, 4]           FLOAT
        #     pred_logits           [batch, 200]              FLOAT
        #     presence_logits       [batch, 1]                FLOAT
        outputs: list[np.ndarray] = self.decoder.run(None, decoder_inputs)  # type: ignore
        assert len(outputs) == 4

        self.logger.info(f"{outputs[0].shape=}, {outputs[2].shape=}, {outputs[3].shape=}")

        masks, boxes, pred_logits, presence_logits = (
            outputs[0][0],
            outputs[1][0],
            outputs[2][0],
            outputs[3][0, 0],
        )

        presence_score = 1 / (1 + np.exp(-presence_logits))
        scores = (1 / (1 + np.exp(-pred_logits))) * presence_score
        keep = scores > self.conf_threshold

        self.logger.debug(f"{scores=}")

        h, w = original_size
        masks = masks[keep]
        boxes = boxes[keep]
        scores = scores[keep]

        # Apply NMS
        keep = self.nms_filter(boxes, self.iou_threshold)
        masks = masks[keep]
        boxes = boxes[keep]
        scores = scores[keep]

        self.logger.info(f"{scores.shape=}")
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = np.clip(boxes, 0, [[w, h, w, h]])

        # Resize masks: 288x288 -> original size
        results: list[SamOnnxResult] = []
        for i, mask in enumerate(masks):
            mask_resized = cv2.resize(mask, (w, h), interpolation=self.cv_interpolation)
            mask_binary = mask_resized > 0
            results.append(
                SamOnnxResult(
                    mask=mask_binary.astype(np.float32),
                    score=scores[i],
                    box=(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]),
                )
            )

        # Sort by score
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results
