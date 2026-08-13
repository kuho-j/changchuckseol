from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np
import cv2
if TYPE_CHECKING:
    import onnxruntime as ort

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

class EmbeddingProcessor():
    def __init__(self, model_path : Path) -> None:
        '''
        create an onnx session
        '''

        # 지연 임포트
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime이 필요합니다. requirements.txt를 설치해 주세요.") from exc
        
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('cv2가 필요합니다.') from exc

        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX 모델을 찾을 수 없습니다: {model_path}")
        
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


    def preprocess_mobilenet(self, image_rgb: np.ndarray) -> np.ndarray:
        """Apply the ImageNet MobileNetV3 resize, center-crop, and normalization."""

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("카메라 이미지는 (height, width, 3) RGB 형식이어야 합니다.")

        height, width = image_rgb.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("카메라 이미지가 비어 있습니다.")

        # torchvision ImageNet weights: Resize(shorter_side=256), CenterCrop(224).
        scale = 256 / min(height, width)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(
            image_rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_CUBIC,
        )
        top = (resized_height - 224) // 2
        left = (resized_width - 224) // 2
        cropped = resized[top : top + 224, left : left + 224]

        normalized = cropped.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


    def create_embedding(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run one RGB image through the ONNX embedding model."""
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: self.preprocess_mobilenet(image_rgb)})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)
