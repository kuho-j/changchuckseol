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

# 주의 사항
raspberry pi 5에 기본으로 들어 있는 라이브러리는 venv가 아닌 raspberry pi의 것을 이용함. (picamera 등)