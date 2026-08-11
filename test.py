#!/usr/bin/env python3
"""카메라로 촬영한 의류 이미지의 MobileNet 임베딩을 DB에 등록한다.

예시:
    python test.py --name "검정 티셔츠" --category dark
    python test.py --garment-id 1 --image-save captures/shirt.jpg
    python test.py --search-only --threshold 0.85

Raspberry Pi에서는 system Python에 설치된 ``picamera2``를 사용할 수 있도록
README의 안내처럼 해당 Python 환경에서 실행한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from garment_db import GarmentDB, VALID_CATEGORIES

DEFAULT_MODEL = Path("models/mobilenetv3_small_embedding.onnx")
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess_mobilenet(image_rgb: np.ndarray) -> np.ndarray:
    """Apply the ImageNet MobileNetV3 resize, center-crop, and normalization."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV가 필요합니다. requirements.txt를 설치해 주세요.") from exc

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


def create_embedding(model_path: Path, image_rgb: np.ndarray) -> np.ndarray:
    """Run one RGB image through the ONNX embedding model."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime이 필요합니다. requirements.txt를 설치해 주세요.") from exc

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX 모델을 찾을 수 없습니다: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: preprocess_mobilenet(image_rgb)})[0]
    return np.asarray(output, dtype=np.float32).reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="카메라 이미지의 의류 임베딩을 DB에 저장")
    garment = parser.add_mutually_exclusive_group()
    garment.add_argument("--garment-id", type=int, help="기존 의류 ID")
    garment.add_argument("--name", help="새로 등록할 의류 이름")
    parser.add_argument(
        "--category",
        choices=sorted(VALID_CATEGORIES),
        help="새 의류의 분류 (--name 사용 시 필수)",
    )
    parser.add_argument("--db", type=Path, default=Path("garments.db"), help="SQLite DB 경로")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="ONNX 임베딩 모델 경로")
    parser.add_argument("--camera-num", type=int, default=0, help="Picamera2 카메라 번호")
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 480),
        help="캡처 해상도 (기본값: 640 480)",
    )
    parser.add_argument("--image-save", type=Path, help="원본 캡처 이미지 저장 경로")
    parser.add_argument("--embedding-save", type=Path, help="임베딩 .npy 저장 경로")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="DB 유사도 검색만 하고 새 임베딩은 저장하지 않음",
    )
    parser.add_argument("--top-k", type=int, default=5, help="출력할 유사 의류 수 (기본값: 5)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="동일 의류로 판단할 코사인 유사도 임계값 (기본값: 0.85)",
    )
    args = parser.parse_args()
    if not args.search_only and args.garment_id is None and args.name is None:
        parser.error("저장하려면 --garment-id 또는 --name과 --category를 지정해야 합니다.")
    if args.name is not None and args.category is None:
        parser.error("--name 사용 시 --category도 지정해야 합니다.")
    if args.category is not None and args.name is None:
        parser.error("--category는 --name과 함께 사용합니다.")
    if min(args.size) <= 0:
        parser.error("--size 값은 양수여야 합니다.")
    if args.top_k <= 0:
        parser.error("--top-k는 1 이상이어야 합니다.")
    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold는 -1.0 이상 1.0 이하여야 합니다.")
    return args


def main() -> None:
    args = parse_args()

    # picamera2는 Raspberry Pi의 system Python에만 있을 수 있으므로 여기서 import한다.
    from src.capture import capture_still, save_image

    image_rgb = capture_still(camera_num=args.camera_num, size=tuple(args.size))
    if args.image_save:
        saved_image = save_image(image_rgb, args.image_save)
        print(f"캡처 이미지 저장: {saved_image}")

    embedding = create_embedding(args.model, image_rgb)
    if args.embedding_save:
        args.embedding_save.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embedding_save, embedding)
        print(f"임베딩 파일 저장: {args.embedding_save}")

    db = GarmentDB(args.db)
    # 새 임베딩을 추가하기 전에 검색해야 방금 저장할 자기 자신과 매칭되지 않는다.
    similar = db.find_similar(embedding, top_k=args.top_k)
    if not similar:
        print("유사도 검색: 등록된 임베딩이 없습니다.")
    else:
        print("유사도 검색 결과:")
        for result in similar:
            print(
                f"  id={result['garment_id']}, name={result['name']}, "
                f"category={result['category']}, similarity={result['similarity']:.4f}"
            )

        matched = [result for result in similar if result["similarity"] >= args.threshold]
        if matched:
            best = matched[0]
            print(
                f"임계값 {args.threshold:.2f} 이상 일치: {best['name']} "
                f"(id={best['garment_id']}, similarity={best['similarity']:.4f})"
            )
        else:
            print(f"임계값 {args.threshold:.2f} 이상의 일치 의류가 없습니다.")

    if args.search_only:
        return

    garment_id = args.garment_id
    if garment_id is None:
        garment_id = db.add_garment(args.name, args.category)
        print(f"의류 등록: id={garment_id}, name={args.name}, category={args.category}")

    embedding_id = db.add_embedding(garment_id, embedding)
    print(f"임베딩 저장 완료: garment_id={garment_id}, embedding_id={embedding_id}, dimension={embedding.size}")


if __name__ == "__main__":
    main()
