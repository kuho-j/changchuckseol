# 프로젝트 개요
이 프로젝트의 목표는 사용자의 옷을 분류하는 것.

사용자가 특정 옷을 등록하면, 그 옷을 다시 넣었을 때 자동으로 분류되게 한다.

등록 시 색이 진한 외출복 / 색이 연한 외출복 / 내복을 선택할 수 있다.

카메라 모듈을 이용하여 받은 이미지를 ai에 넣되,
ai의 classification layer 이전 층의 출력값을 DB에 저장한다.

이를 이용한 유사도 검색으로 옷을 분류하는 방식을 사용할 것이다.

# 스택
raspberrypi 5를 기반으로 빌드할 것이다.
pre-trained된 경량 ai 모델을 이용할 것이다. (MobileNet V3-Small / EfficientNet-Lite 등) (fine-tunning 할 것인지는 나중에 생각...)
DB는 ...
검색은 ...

# 구현
옷 감지 : TFO 센서인 VL53L0를 사용한다. 
컨베이어 벨트 시작 부분과 끝 부분에 센서를 달아
사용자가 옷을 올려 놓으면 컨베이어 벨트 끝까지 옷을 보낸다.

컨베이어 벨트 : DC 모터를 사용한다.
모터 드라이버에 연결할 것이며, 센서에 맞추어 옷을 보낸다.

카메라 : 옷이 컨베이어 벨트 끝에 오면, 카메라 모듈을 이용하여 옷의 사진을 찍는다.

이후 : ...

## MobileNetV3-Small ONNX 동적 양자화 (초기 테스트)

개발 PC에서 `onnx`와 `onnxruntime`를 설치한 뒤 아래를 실행한다.

```bash
python -m src.model
```

기본 동작은 ImageNet 사전학습 MobileNetV3-Small을 내려받아, 분류기 직전의
`576`차원 feature embedding을 ONNX로 내보낸 후 동적 int8 양자화한다.
출력 파일은 다음과 같다.

- `models/mobilenetv3_small_embedding.onnx`: FP32 기준 모델
- `models/mobilenetv3_small_embedding_dynamic_int8.onnx`: Raspberry Pi CPU용 동적 int8 테스트 모델

첫 실행에서 사전학습 가중치를 받을 수 없는 환경이라면 아래처럼 구조와 양자화
파이프라인만 점검할 수 있다. 이 모델의 embedding은 학습되지 않았으므로 실제
유사도 검색에는 사용하면 안 된다.

```bash
python -m src.model --no-imagenet-weights
```

분류 logits 전체가 필요하면 `--classification`을 더한다. 카메라 입력은 RGB로
변환하고 `float32`, `NCHW(1, 3, 224, 224)` 형태로 만들며, ImageNet 사전학습
가중치 사용 시 해당 weights의 resize/normalize 전처리를 동일하게 적용한다.

동적 양자화는 빠르게 용량·호환성을 확인하기 위한 첫 단계다. Raspberry Pi에서
실측 지연시간과 embedding 유사도(또는 검색 정확도)를 FP32와 비교하고, 정확도나
성능이 부족하면 대표 의류 이미지로 static QDQ INT8 양자화로 전환한다.

# 주의 사항
raspberry pi 5에 기본으로 들어 있는 라이브러리는 venv가 아닌 raspberry pi의 것을 이용함. (picamera 등)
