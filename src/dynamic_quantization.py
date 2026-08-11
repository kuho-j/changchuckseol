"""MobileNetV3-Small ONNX export and dynamic-quantization utility.

The default export is an embedding model because the project identifies clothes
by similarity search.  It returns the 576-dimensional tensor immediately
before MobileNetV3-Small's classification head.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


IMAGE_SIZE = 224
OPSET_VERSION = 18


class MobileNetV3SmallEmbedding(nn.Module):
    """Expose MobileNetV3-Small's pooled feature vector (shape: N x 576)."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, image: Tensor) -> Tensor:
        features = self.features(image)
        return torch.flatten(self.avgpool(features), start_dim=1)


def build_model(*, imagenet_weights: bool, embedding: bool) -> nn.Module:
    """Create an evaluation-mode MobileNetV3-Small model.

    Set ``imagenet_weights=False`` for an offline pipeline test.  For useful
    similarity embeddings, use the ImageNet-pretrained weights (or load a
    later fine-tuned checkpoint before export).
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if imagenet_weights else None
    backbone = mobilenet_v3_small(weights=weights)
    model: nn.Module = MobileNetV3SmallEmbedding(backbone) if embedding else backbone
    return model.eval()


def export_onnx(model: nn.Module, path: Path) -> None:
    """Export a fixed 224x224 RGB input model with a dynamic batch dimension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output_name = "embedding" if isinstance(model, MobileNetV3SmallEmbedding) else "logits"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            path,
            input_names=["image"],
            output_names=[output_name],
            dynamic_axes={"image": {0: "batch"}, output_name: {0: "batch"}},
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(path))


def quantize_dynamic_model(source: Path, destination: Path) -> None:
    """Create an int8-weight ONNX model for CPU execution.

    Explicitly including ``Conv`` is important for MobileNetV3, whose work is
    predominantly convolutional rather than linear layers.
    """
    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    onnx.checker.check_model(str(destination))


def smoke_test(float_model: Path, quantized_model: Path) -> tuple[float, float]:
    """Run both models and return their maximum absolute output difference."""
    image = np.random.default_rng(0).random((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    float_output = ort.InferenceSession(str(float_model), providers=["CPUExecutionProvider"]).run(None, {"image": image})[0]
    quantized_output = ort.InferenceSession(str(quantized_model), providers=["CPUExecutionProvider"]).run(None, {"image": image})[0]
    return float(np.max(np.abs(float_output - quantized_output))), float(np.mean(np.abs(float_output - quantized_output)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and dynamically quantize MobileNetV3-Small.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--no-imagenet-weights", action="store_true", help="Do not download pretrained weights.")
    parser.add_argument("--classification", action="store_true", help="Export 1,000 ImageNet logits instead of 576-D embeddings.")
    args = parser.parse_args()

    suffix = "classification" if args.classification else "embedding"
    float_model = args.output_dir / f"mobilenetv3_small_{suffix}.onnx"
    quantized_model = args.output_dir / f"mobilenetv3_small_{suffix}_dynamic_int8.onnx"
    model = build_model(imagenet_weights=not args.no_imagenet_weights, embedding=not args.classification)
    export_onnx(model, float_model)
    quantize_dynamic_model(float_model, quantized_model)
    max_error, mean_error = smoke_test(float_model, quantized_model)

    print(f"float model:     {float_model} ({float_model.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"dynamic int8:    {quantized_model} ({quantized_model.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"output error:    max={max_error:.6f}, mean={mean_error:.6f}")


if __name__ == "__main__":
    main()
