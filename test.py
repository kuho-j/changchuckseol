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
from src.process import EmbeddingProcessor

DEFAULT_MODEL = Path("models/mobilenetv3_small_embedding.onnx")

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
    processor = EmbeddingProcessor(args.model)

    # picamera2는 Raspberry Pi의 system Python에만 있을 수 있으므로 여기서 import한다.
    from src.capture import capture_still, save_image

    image_rgb = capture_still(camera_num=args.camera_num, size=tuple(args.size))
    if args.image_save:
        saved_image = save_image(image_rgb, args.image_save)
        print(f"캡처 이미지 저장: {saved_image}")

    embedding = processor.create_embedding(image_rgb)
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
